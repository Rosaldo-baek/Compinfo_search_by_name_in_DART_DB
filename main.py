import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
import re
import openai
import unicodedata


def download_corpcode_zip(api_key: str) -> bytes:
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP 오류: {response.status_code}\n{response.text}")
    return response.content


def xml_to_dataframe(zip_binary: bytes) -> pd.DataFrame:
    zip_bytes = io.BytesIO(zip_binary)
    with zipfile.ZipFile(zip_bytes) as zf:
        xml_name = zf.namelist()[0]
        with zf.open(xml_name) as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            rows = []
            for corp in root.findall("list"):
                rows.append(
                    {
                        "corp_code": corp.findtext("corp_code"),
                        "corp_name": corp.findtext("corp_name"),
                        "corp_eng_name": corp.findtext("corp_eng_name"),
                        "stock_code": corp.findtext("stock_code"),
                        "modify_date": corp.findtext("modify_date"),
                    }
                )
    df = pd.DataFrame(rows)
    return df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()

def basic_clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    # 1) 전각/반각, 기타 호환 문자 통일
    s = unicodedata.normalize("NFKC", str(name)).strip()

    # 2) 법적 형태 제거
    patterns = [
        "(주)", "㈜", "주식회사", "유한회사", "(유)",
        "주식 회 사", "유 한 회 사"
    ]
    for p in patterns:
        s = s.replace(p, "")

    # 3) 괄호 안 '영문 포함' 부분 제거
    #    ex) 에이치이공일(H201Co.Ltd) -> 에이치이공일
    #    여러 개 있을 수 있으니 한 번에 처리
    s = re.sub(r"\([A-Za-z0-9][^)]*\)", "", s)

    # 4) 괄호 주변/다중 공백 정리
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def drop_english_in_parentheses(name: str) -> str:
    """
    '터미네이터다이아몬드코리아(TerminatorDiamondProductsKoreaINC.)'
    -> '터미네이터다이아몬드코리아' 로 잘라내는 전용 후처리
    """
    if not isinstance(name, str):
        return ""

    s = unicodedata.normalize("NFKC", name).strip()

    # 앞부분: 한글/숫자/공백
    # 괄호 안: 영문자가 하나 이상 포함된 어떤 내용
    # 예) 에이치이공일(H201Co.Ltd) 등도 커버
    s = re.sub(r"([가-힣0-9\s]+)\([A-Za-z0-9].*\)", r"\1", s)

    return s.strip()


def clean_names_with_openai(raw_names, openai_api_key: str):
    """
    OpenAI로 기업명 정제.
    - 입력: 원본 기업명 리스트
    - 출력: 정제된 기업명 리스트
    """

    # 0) 먼저 NFKC 정규화해서 전각/반각 통일
    #    예: "（주）와이월드제모" -> "(주)와이월드제모"
    norm_raw = [unicodedata.normalize("NFKC", str(n)).strip() for n in raw_names]

    # OpenAI Key 없으면 기본 정제만 사용
    if not openai_api_key:
        return [basic_clean_name(n) for n in norm_raw]

    # 프롬프트용 문자열 구성 (정규화된 값 기준)
    joined = "\n".join(f"{i+1}. {name}" for i, name in enumerate(norm_raw))

    prompt = f"""
다음은 한국 기업명 목록임.

- '주식회사', '(주)', '㈜', '유한회사', '(유)' 등 법적 형태는 제거
- 괄호 안 브랜드명, 설명은 유지하되, 불필요한 공백은 최소화
- 한글 표기는 원문 유지, 번역하지 말 것
- 기업명이 한글명(영어명)과 같은 경우 한글명만 남김.  예) '에이치이공일(H201Co.Ltd)' 이라면 '에이치이공일'만 남김
- 각 줄 하나의 기업명, 입력과 같은 순서로 출력
- 번호가 있다면 제거 후 기업명만 남길 것

기업명 목록:
{joined}

위 기업명을 정제해서, 각 줄에 '정제된 기업명'만 한 줄씩 출력하시오.
다른 설명 텍스트는 출력하지 말 것.
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",  # openai==0.28.1 기준
            messages=[
                {"role": "system", "content": "당신은 한국 기업명 정제 도우미입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,api_key = openai_api_key
        )
        content = resp.choices[0].message["content"]
    except Exception as e:
        # 에러 나면 정규화 + 기본 정제로 fallback
        print(f"OpenAI 오류: {e}")
        return [basic_clean_name(n) for n in norm_raw]

    # 1) 응답 라인 나누기
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    # 2) 번호 제거
    cleaned_lines = []
    for line in lines:
        # "1. ", "1) ", "1 - " 등 앞 번호 제거
        line = re.sub(r"^\d+[\).\s-]*", "", line).strip()
        cleaned_lines.append(line)

    # 3) 줄 수와 입력 수를 맞추는 로직
    if len(cleaned_lines) == len(norm_raw):
        result = cleaned_lines
    elif len(cleaned_lines) == 1:
        # 한 줄만 나오면 전체에 복제
        result = cleaned_lines * len(norm_raw)
    else:
        print("clean_names_with_openai: 라인 수 불일치, fallback 사용")
        return [basic_clean_name(n) for n in norm_raw]

    # 4-1) 먼저 괄호 안 영문 제거 후
    result = [drop_english_in_parentheses(x) for x in result]

    # 4-2) 기본 클리닝까지 적용해서 반환
    return [basic_clean_name(x) for x in result]

API_BASE_URL = "https://opendart.fss.or.kr/api/company.json"


def get_company_info(corp_code: str, api_key: str) -> dict:
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code
    }

    try:
        res = requests.get(API_BASE_URL, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        return {
            "corp_code": corp_code,
            "status": "999",
            "message": f"request_error: {e}"
        }

    status = data.get("status")
    if status != "000":
        return {
            "corp_code": corp_code,
            "status": status,
            "message": data.get("message")
        }

    return {
        "corp_code": corp_code,
        "status": status,
        "message": data.get("message"),
        "corp_name": data.get("corp_name"),
        "corp_name_eng": data.get("corp_name_eng"),
        "stock_name": data.get("stock_name"),
        "stock_code": data.get("stock_code"),
        "ceo_nm": data.get("ceo_nm"),
        "corp_cls": data.get("corp_cls"),
        "jurir_no": data.get("jurir_no"),
        "bizr_no": data.get("bizr_no"),
        "adres": data.get("adres"),
        "hm_url": data.get("hm_url"),
        "ir_url": data.get("ir_url"),
        "phn_no": data.get("phn_no"),
        "fax_no": data.get("fax_no"),
        "induty_code": data.get("induty_code"),
        "est_dt": data.get("est_dt"),
        "acc_mt": data.get("acc_mt"),
    }


def main():
    st.title("DART corpCode / 기업개황 매칭 도구")

    if "df_corp" not in st.session_state:
        st.session_state["df_corp"] = None

    st.sidebar.header("설정")
    dart_api_key = st.sidebar.text_input(
        "DART API Key",
        type="password",
    )
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
    )
    sheet_name_input = st.sidebar.text_input(
        "업로드 엑셀 시트명 (비우면 첫 시트)",
        value=""
    )

    st.write("1단계: corpCode 다운로드 → 2단계: 엑셀 업로드 및 기업개황 조회")

    # 1) corpCode
    st.subheader("1. DART corpCode 다운로드")
    if st.button("corpCode 가져오기"):
        if not dart_api_key:
            st.warning("DART API Key를 먼저 입력해 주셔야 합니다.")
            return

        try:
            with st.spinner("DART에서 corpCode ZIP 다운로드 중"):
                zip_data = download_corpcode_zip(dart_api_key)
            with st.spinner("XML → DataFrame 변환 중"):
                df_corp = xml_to_dataframe(zip_data)

            df_corp["corp_name_norm"] = df_corp["corp_name"].astype(str).apply(basic_clean_name)
            st.session_state["df_corp"] = df_corp

            st.success("corpCode 데이터 로드 완료")
            st.write(f"총 기업 수: {len(df_corp):,}개")
            st.dataframe(df_corp.head(50))

            excel_bytes = df_to_excel_bytes(df_corp)
            st.download_button(
                label="corp_code 전체 목록 엑셀 다운로드",
                data=excel_bytes,
                file_name="corp_code.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"corpCode 처리 중 오류: {e}")

    # 2) 엑셀 업로드 및 기업개황
    st.subheader("2. 기업 리스트 엑셀 업로드 및 기업개황 조회")

    df_corp = st.session_state.get("df_corp")
    uploaded_file = st.file_uploader(
        "기업 리스트 엑셀 업로드 (기업명 컬럼 필요)",
        type=["xlsx", "xls"]
    )

    if uploaded_file is not None:
        sheet_arg = sheet_name_input if sheet_name_input.strip() else 0
        try:
            list_data = pd.read_excel(uploaded_file, sheet_name=sheet_arg, dtype="str")
        except Exception as e:
            st.error(f"엑셀 읽기 오류: {e}")
            return

        st.write("업로드 데이터 미리보기")
        st.dataframe(list_data.head(20))

        if "기업명" not in list_data.columns:
            st.error("'기업명' 컬럼이 필요합니다.")
            return

        if df_corp is None:
            st.warning("먼저 1단계에서 corpCode를 내려받으셔야 합니다.")
            return

        if st.button("기업명 정제 및 DART 기업개황 조회 실행"):
            if not dart_api_key:
                st.warning("DART API Key가 필요합니다.")
                return

            with st.spinner("기업명 정제 중"):
                list_data["기업명_raw"] = list_data["기업명"].astype(str)
                list_data["기업명_basic"] = list_data["기업명_raw"].apply(basic_clean_name)
                cleaned_names = clean_names_with_openai(
                    list_data["기업명_basic"].tolist(),
                    openai_api_key=openai_api_key
                )
                list_data["기업명_clean"] = cleaned_names

            st.write("정제된 기업명 예시")
            st.dataframe(
                list_data[["기업명", "기업명_basic", "기업명_clean"]].head(20)
            )

            corp_slim = df_corp[["corp_code", "corp_name", "corp_name_norm"]].drop_duplicates()
            merged = list_data.merge(
                corp_slim,
                left_on="기업명_clean",
                right_on="corp_name_norm",
                how="left"
            )

            st.write("corp_code 매칭 결과 예시")
            st.dataframe(
                merged[["기업명", "기업명_clean", "corp_code", "corp_name"]].head(20)
            )

            valid_codes = merged["corp_code"].dropna().astype(str).unique()
            st.write(f"매칭된 기업 수: {len(valid_codes):,}개")

            if len(valid_codes) == 0:
                st.warning("매칭된 corp_code가 없습니다.")
                return

            results = []
            progress_bar = st.progress(0.0)
            for i, code in enumerate(valid_codes, start=1):
                info = get_company_info(code, api_key=dart_api_key)
                results.append(info)
                progress_bar.progress(i / len(valid_codes))

            company_df = pd.DataFrame(results)

            final = merged.merge(
                company_df,
                on="corp_code",
                how="left",
                suffixes=("", "_dart")
            )

            st.success("기업개황 조회 및 매칭 완료")
            st.write("최종 결과 미리보기")
            st.dataframe(final.head(50))

            final_excel_bytes = df_to_excel_bytes(final)
            st.download_button(
                label="최종 결과 엑셀 다운로드",
                data=final_excel_bytes,
                file_name="기업개황_매칭결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()

