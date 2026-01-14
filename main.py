import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
import re
import unicodedata
import time

# =========================================================
# 0) DART API 엔드포인트
# =========================================================
API_COMPANY_URL = "https://opendart.fss.or.kr/api/company.json"
API_FNLTT_ALL_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
API_EMP_URL = "https://opendart.fss.or.kr/api/empSttus.json"


# =========================================================
# 1) 공통 유틸: 멀티시트 엑셀 bytes 생성
# =========================================================
def dfs_to_excel_bytes(sheet_df_map: dict) -> bytes:
    """
    여러 DataFrame을 하나의 엑셀 파일(멀티시트)로 묶어서 bytes로 반환
    - sheet_df_map: {"sheet_name": df, ...}
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheet_df_map.items():
            safe_name = str(sheet_name)[:31]  # Excel 시트명 31자 제한
            df.to_excel(writer, index=False, sheet_name=safe_name)
    output.seek(0)
    return output.read()


# =========================================================
# 2) corpCode 다운로드/파싱
# =========================================================
def download_corpcode_zip(api_key: str) -> bytes:
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}
    response = requests.get(url, params=params, timeout=30)

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

    return pd.DataFrame(rows)


# =========================================================
# 3) 기업명 정제 로직
# =========================================================
def basic_clean_name(name: str) -> str:
    """
    기업명 기본 정제
    - 법적 형태 제거
    - 괄호 안 영문/숫자로 시작하는 부가정보 제거
    - 공백 정리
    """
    if not isinstance(name, str):
        return ""

    s = unicodedata.normalize("NFKC", str(name)).strip()

    patterns = ["(주)", "㈜", "주식회사", "유한회사", "(유)", "주식 회 사", "유 한 회 사"]
    for p in patterns:
        s = s.replace(p, "")

    # 예: 에이치이공일(H201Co.Ltd) -> 에이치이공일
    s = re.sub(r"\([A-Za-z0-9][^)]*\)", "", s)

    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def drop_english_in_parentheses(name: str) -> str:
    """
    한글명(영문명) 형태면 한글명만 남김
    예) 터미네이터다이아몬드코리아(Terminator...) -> 터미네이터다이아몬드코리아
    """
    if not isinstance(name, str):
        return ""

    s = unicodedata.normalize("NFKC", name).strip()
    s = re.sub(r"([가-힣0-9\s]+)\([A-Za-z0-9].*\)", r"\1", s)
    return s.strip()


def clean_names_with_openai_batched(raw_names, openai_api_key: str, batch_size: int = 80):
    """
    OpenAI로 기업명 정제(선택) + 배치 처리
    - OpenAI 키 없거나 openai 미설치면 basic_clean_name만 수행
    - 원코드(openai.ChatCompletion.create) 방식 유지
    """
    norm_raw = [unicodedata.normalize("NFKC", str(n)).strip() for n in raw_names]

    if not openai_api_key:
        return [basic_clean_name(n) for n in norm_raw]

    try:
        import openai
    except Exception:
        return [basic_clean_name(n) for n in norm_raw]

    openai.api_key = openai_api_key

    results = []

    for start in range(0, len(norm_raw), batch_size):
        batch = norm_raw[start:start + batch_size]

        joined = "\n".join(f"{i+1}. {name}" for i, name in enumerate(batch))

        prompt = f"""
다음은 한국 기업명 목록임.

- '주식회사', '(주)', '㈜', '유한회사', '(유)' 등 법적 형태는 제거
- 괄호 안 브랜드명, 설명은 유지하되, 불필요한 공백은 최소화
- 한글 표기는 원문 유지, 번역하지 말 것
- 기업명이 한글명(영어명)과 같은 경우 한글명만 남김. 예) '에이치이공일(H201Co.Ltd)' -> '에이치이공일'
- 각 줄 하나의 기업명, 입력과 같은 순서로 출력
- 번호가 있다면 제거 후 기업명만 남길 것

기업명 목록:
{joined}

정제 결과를 각 줄에 '정제된 기업명'만 출력하시오. 다른 설명 텍스트 출력 금지.
"""

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 한국 기업명 정제 도우미입니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = resp.choices[0].message["content"]
            lines = [line.strip() for line in content.splitlines() if line.strip()]

            cleaned_lines = []
            for line in lines:
                line = re.sub(r"^\d+[\).\s-]*", "", line).strip()
                cleaned_lines.append(line)

            # 라인 수 불일치 시 batch fallback
            if len(cleaned_lines) != len(batch):
                cleaned_lines = [basic_clean_name(n) for n in batch]

        except Exception:
            cleaned_lines = [basic_clean_name(n) for n in batch]

        cleaned_lines = [drop_english_in_parentheses(x) for x in cleaned_lines]
        cleaned_lines = [basic_clean_name(x) for x in cleaned_lines]

        results.extend(cleaned_lines)

    return results


# =========================================================
# 4) DART API 조회 함수들
# =========================================================
def get_company_info(corp_code: str, api_key: str) -> dict:
    corp_code_8 = str(corp_code).zfill(8)
    params = {"crtfc_key": api_key, "corp_code": corp_code_8}

    try:
        res = requests.get(API_COMPANY_URL, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        return {"corp_code": corp_code_8, "status": "999", "message": f"request_error: {e}"}

    if data.get("status") != "000":
        return {"corp_code": corp_code_8, "status": data.get("status"), "message": data.get("message")}

    return {
        "corp_code": corp_code_8,
        "status": "000",
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


def get_fnltt_singl_acnt_all(corp_code: str, bsns_year: str, reprt_code: str, fs_div: str, api_key: str) -> pd.DataFrame:
    corp_code_8 = str(corp_code).zfill(8)

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code_8,
        "bsns_year": str(bsns_year),
        "reprt_code": str(reprt_code),
        "fs_div": str(fs_div),
    }

    try:
        res = requests.get(API_FNLTT_ALL_URL, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
    except Exception:
        return pd.DataFrame()

    if data.get("status") != "000":
        return pd.DataFrame()

    rows = data.get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["req_corp_code"] = corp_code_8
    df["req_bsns_year"] = str(bsns_year)
    df["req_reprt_code"] = str(reprt_code)
    df["req_fs_div"] = str(fs_div)
    return df


def get_emp_status(corp_code: str, bsns_year: str, reprt_code: str, api_key: str) -> pd.DataFrame:
    corp_code_8 = str(corp_code).zfill(8)

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code_8,
        "bsns_year": str(bsns_year),
        "reprt_code": str(reprt_code),
    }

    try:
        res = requests.get(API_EMP_URL, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
    except Exception:
        return pd.DataFrame()

    if data.get("status") != "000":
        return pd.DataFrame()

    rows = data.get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["req_corp_code"] = corp_code_8
    df["req_bsns_year"] = str(bsns_year)
    df["req_reprt_code"] = str(reprt_code)
    return df


# =========================================================
# 5) Streamlit main
# =========================================================
def main():
    st.title("DART 기업개황 + 재무제표 + 직원현황 매칭 도구")

    # -----------------------------
    # 세션 상태
    # -----------------------------
    if "df_corp" not in st.session_state:
        st.session_state["df_corp"] = None

    # -----------------------------
    # 사이드바 설정
    # -----------------------------
    st.sidebar.header("설정")

    dart_api_key = st.sidebar.text_input("DART API Key", type="password")
    openai_api_key = st.sidebar.text_input("OpenAI API Key (선택)", type="password")
    sheet_name_input = st.sidebar.text_input("업로드 엑셀 시트명 (비우면 첫 시트)", value="")

    st.sidebar.markdown("---")
    st.sidebar.subheader("추가 조회 옵션")

    bsns_year = st.sidebar.text_input("사업연도", value="2024")

    # 보고서 코드 한글 선택 → 내부 코드 자동 매핑
    reprt_code_map = {
        "사업보고서": "11011",
        "반기보고서": "11012",
        "1분기보고서": "11013",
        "3분기보고서": "11014",
    }
    reprt_label = st.sidebar.selectbox(
        "보고서 종류",
        options=list(reprt_code_map.keys()),
        index=0,
        help="보고서 종류를 선택하면 내부 코드(11011~11014)가 자동 적용됩니다.",
    )
    reprt_code = reprt_code_map[reprt_label]

    # 재무제표 구분 한글 선택 → 내부 코드 자동 매핑
    fs_div_map = {
        "개별재무제표": "OFS",
        "연결재무제표": "CFS",
    }
    fs_div_label = st.sidebar.selectbox(
        "재무제표 구분",
        options=list(fs_div_map.keys()),
        index=0,
        help="재무제표 구분을 선택하면 내부 코드(OFS/CFS)가 자동 적용됩니다.",
    )
    fs_div = fs_div_map[fs_div_label]

    DO_FNLTT = st.sidebar.checkbox("재무제표 조회 포함", value=True)
    DO_EMP = st.sidebar.checkbox("직원현황 조회 포함", value=True)

    # 고급 옵션: 기본 숨김 (자동화 목적)
    with st.sidebar.expander("고급 옵션 (필요 시만 조정)", expanded=False):
        openai_batch_size = st.number_input(
            "OpenAI 정제 배치 크기",
            min_value=10,
            max_value=200,
            value=80,
            step=10,
            help="기업명 수가 많으면 배치 크기를 키우면 호출 횟수는 줄지만 실패 리스크가 증가할 수 있습니다.",
        )
        sleep_sec = st.number_input(
            "API 호출 간 sleep (초)",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.1,
            help="DART API 과호출 방지 목적의 대기시간입니다. 0으로 하면 더 빠르지만 제한 리스크가 있습니다.",
        )

    st.write("1단계: corpCode 다운로드 → 2단계: 엑셀 업로드/매칭 → 3단계: 기업개황+재무+직원 조회")

    # =========================================================
    # 1) corpCode 다운로드
    # =========================================================
    st.subheader("1) DART corpCode 다운로드")

    if st.button("corpCode 가져오기"):
        if not dart_api_key:
            st.warning("DART API Key를 먼저 입력해 주셔야 합니다.")
            return

        try:
            with st.spinner("DART에서 corpCode ZIP 다운로드 중..."):
                zip_data = download_corpcode_zip(dart_api_key)

            with st.spinner("XML → DataFrame 변환 중..."):
                df_corp = xml_to_dataframe(zip_data)

            df_corp["corp_name_norm"] = df_corp["corp_name"].astype(str).apply(basic_clean_name)
            st.session_state["df_corp"] = df_corp

            st.success("corpCode 데이터 로드 완료")
            st.write(f"총 기업 수: {len(df_corp):,}개")
            st.dataframe(df_corp.head(50))

            excel_bytes = dfs_to_excel_bytes({"corp_code": df_corp})
            st.download_button(
                label="corp_code 전체 목록 엑셀 다운로드",
                data=excel_bytes,
                file_name="corp_code.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"corpCode 처리 중 오류: {e}")
            return

    # =========================================================
    # 2) 기업 리스트 업로드 + 매칭 + DART 조회
    # =========================================================
    st.subheader("2) 기업 리스트 업로드 및 조회")

    df_corp = st.session_state.get("df_corp")

    uploaded_file = st.file_uploader(
        "기업 리스트 엑셀 업로드 (필수 컬럼: '기업명')",
        type=["xlsx", "xls"]
    )

    if uploaded_file is None:
        return

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

    if not dart_api_key:
        st.warning("DART API Key가 필요합니다.")
        return

    if st.button("기업명 정제 → corp_code 매칭 → 기업개황/재무/직원 조회 실행"):
        # ---------------------------
        # (A) 기업명 정제
        # ---------------------------
        with st.spinner("기업명 정제 중..."):
            list_data["기업명_raw"] = list_data["기업명"].astype(str)
            list_data["기업명_basic"] = list_data["기업명_raw"].apply(basic_clean_name)

            cleaned_names = clean_names_with_openai_batched(
                list_data["기업명_basic"].tolist(),
                openai_api_key=openai_api_key,
                batch_size=int(openai_batch_size),
            )
            list_data["기업명_clean"] = cleaned_names

        st.write("정제 결과 예시")
        st.dataframe(list_data[["기업명", "기업명_basic", "기업명_clean"]].head(20))

        # ---------------------------
        # (B) corp_code 매칭
        # ---------------------------
        corp_slim = df_corp[["corp_code", "corp_name", "corp_name_norm"]].drop_duplicates()
        merged = list_data.merge(
            corp_slim,
            left_on="기업명_clean",
            right_on="corp_name_norm",
            how="left"
        )

        st.write("corp_code 매칭 결과 예시")
        st.dataframe(merged[["기업명", "기업명_clean", "corp_code", "corp_name"]].head(30))

        valid_codes = merged["corp_code"].dropna().astype(str).unique().tolist()
        st.write(f"매칭된 corp_code 수: {len(valid_codes):,}개")

        if len(valid_codes) == 0:
            st.warning("매칭된 corp_code가 없습니다. (기업명 정제 규칙/매칭키를 조정해야 합니다.)")

            bytes_no_match = dfs_to_excel_bytes({"company_match": merged})
            st.download_button(
                label="매칭 실패 결과 다운로드",
                data=bytes_no_match,
                file_name="기업개황_매칭결과_NO_MATCH.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            return

        # ---------------------------
        # (C) DART 조회
        # ---------------------------
        st.info("DART 조회 진행 중입니다. 기업 수가 많으면 시간이 걸릴 수 있습니다.")

        company_rows = []
        fnltt_frames = []
        emp_frames = []

        progress = st.progress(0.0)
        status_text = st.empty()

        for i, code in enumerate(valid_codes, start=1):
            corp_code_8 = str(code).zfill(8)
            status_text.write(f"조회 중: {i}/{len(valid_codes)} (corp_code={corp_code_8})")

            # 1) 기업개황
            company_rows.append(get_company_info(code, api_key=dart_api_key))

            # 2) 재무제표
            if DO_FNLTT:
                df_fnltt = get_fnltt_singl_acnt_all(
                    corp_code=code,
                    bsns_year=bsns_year,
                    reprt_code=reprt_code,
                    fs_div=fs_div,
                    api_key=dart_api_key
                )
                if not df_fnltt.empty:
                    fnltt_frames.append(df_fnltt)

            # 3) 직원현황
            if DO_EMP:
                df_emp = get_emp_status(
                    corp_code=code,
                    bsns_year=bsns_year,
                    reprt_code=reprt_code,
                    api_key=dart_api_key
                )
                if not df_emp.empty:
                    emp_frames.append(df_emp)

            # 과호출 방지 sleep
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))

            progress.progress(i / len(valid_codes))

        company_df = pd.DataFrame(company_rows)
        fnltt_df = pd.concat(fnltt_frames, ignore_index=True) if fnltt_frames else pd.DataFrame()
        emp_df = pd.concat(emp_frames, ignore_index=True) if emp_frames else pd.DataFrame()

        st.success("DART 조회 완료")
        st.write(f"기업개황 수집: {len(company_df):,}행")
        if DO_FNLTT:
            st.write(f"재무제표 수집: {len(fnltt_df):,}행")
        if DO_EMP:
            st.write(f"직원현황 수집: {len(emp_df):,}행")

        # ---------------------------
        # (D) 최종 병합 + 멀티시트 엑셀 다운로드
        # ---------------------------
        final_company = merged.merge(
            company_df,
            on="corp_code",
            how="left",
            suffixes=("", "_dart")
        )

        st.subheader("최종 결과 미리보기")
        st.dataframe(final_company.head(50))

        sheet_map = {"company_match": final_company}

        if DO_FNLTT:
            if not fnltt_df.empty:
                sheet_map["fnltt_raw"] = fnltt_df
            else:
                sheet_map["fnltt_raw"] = pd.DataFrame([{"note": "재무제표 데이터 없음"}])

        if DO_EMP:
            if not emp_df.empty:
                sheet_map["emp_status_raw"] = emp_df
            else:
                sheet_map["emp_status_raw"] = pd.DataFrame([{"note": "직원현황 데이터 없음"}])

        out_bytes = dfs_to_excel_bytes(sheet_map)

        st.download_button(
            label="최종 결과 엑셀 다운로드 (멀티시트)",
            data=out_bytes,
            file_name=f"최종결과_{bsns_year}_{reprt_code}_{fs_div}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
