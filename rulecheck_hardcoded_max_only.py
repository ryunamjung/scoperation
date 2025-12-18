from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Dict, List, Set

import pandas as pd
import streamlit as st


# ============================================================
# ✅ 하드코딩 RULES (각 파일의 case_n 최대값과 동일한 행만)
# ============================================================
RULES: Dict[str, dict] = {
    # (사용자 제공 RULES 그대로 두시면 됩니다)
    # ... 생략하지 말고 현재 갖고 있는 RULES 전체를 여기 붙여넣으세요 ...
}

# -----------------------------
# 공통: 복붙 파서
# -----------------------------
EXPECTED_COLS = [
    "선택", "처방코드", "청구코드", "처방명", "항목", "종별가산", "단가", "종별가산단가",
    "1회투", "Tms/Tot Q", "일수", "금액", "급비", "급비지정", "포괄", "완화", "원외", "무료", "처방일자", "항목명"
]

SECTION_ROW_PATTERN = re.compile(r"^\s*\[\s*.+?\s*\]\s*$")  # [ 진찰료 ] 같은 행


def _clean_lines(raw: str) -> str:
    lines: List[str] = []
    for ln in raw.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if not ln.strip():
            continue
        if SECTION_ROW_PATTERN.match(ln.strip()):
            continue
        lines.append(ln.lstrip("\t"))
    return "\n".join(lines)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    rename_map = {
        "처방 코드": "처방코드",
        "청구 코드": "청구코드",
        "처방코드 ": "처방코드",
        "청구코드 ": "청구코드",
        "처 방 코 드": "처방코드",
        "청 구 코 드": "청구코드",
        "처방코드(내부)": "처방코드",
        "청구코드(EDI)": "청구코드",
    }
    return df.rename(columns=rename_map)


def parse_clipboard_tsv(raw: str) -> pd.DataFrame:
    cleaned = _clean_lines(raw)
    if not cleaned.strip():
        return pd.DataFrame(columns=EXPECTED_COLS)

    df = pd.read_csv(
        io.StringIO(cleaned),
        sep="\t",
        dtype=str,
        engine="python",
        keep_default_na=False,
    )
    df = _normalize_columns(df)

    # 헤더 없을 때 재시도
    if ("처방코드" not in df.columns) and ("청구코드" not in df.columns):
        df2 = pd.read_csv(
            io.StringIO(cleaned),
            sep="\t",
            header=None,
            dtype=str,
            engine="python",
            keep_default_na=False,
        )
        df2 = df2.iloc[:, : len(EXPECTED_COLS)]
        df2.columns = EXPECTED_COLS[: df2.shape[1]]
        df = df2
    else:
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[EXPECTED_COLS].copy()

    # 날짜
    df["처방일자"] = df["처방일자"].astype(str).str.strip()
    df["처방일자_dt"] = pd.to_datetime(df["처방일자"], format="%Y%m%d", errors="coerce")

    # 섹션행 제거
    mask_section = df["처방코드"].astype(str).str.strip().str.match(r"^\[.+\]$")
    df = df.loc[~mask_section].copy()

    # 코드 둘 다 비어있는 합계행 제거
    mask_no_codes = (df["처방코드"].astype(str).str.strip() == "") & (df["청구코드"].astype(str).str.strip() == "")
    df = df.loc[~mask_no_codes].copy()

    # 기본 trim
    for c in ["항목", "처방코드", "청구코드", "처방명", "급비", "처방일자"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    return df


# -----------------------------
# 점검 로직
# -----------------------------
def applied_base_codes_by_date(df_case: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    날짜별 적용 기준코드 목록
    - 기준코드 판정: 항목=0801 행에서 RULES[base_code]['base_col'] 값에 base_code가 등장하면 적용
    """
    out: Dict[str, Set[str]] = {}
    if df_case is None or df_case.empty:
        return out

    d = df_case.copy()
    if "처방일자" not in d.columns:
        d["처방일자"] = ""
    if "항목" not in d.columns:
        d["항목"] = ""
    d["처방일자"] = d["처방일자"].astype(str).str.strip()
    d["항목"] = d["항목"].astype(str).str.strip()

    for rx_date, g in d.groupby("처방일자", dropna=False):
        codes_for_date: Set[str] = set()
        for base_code, rule in RULES.items():
            base_col = rule.get("base_col", "청구코드")
            if base_col not in g.columns:
                continue
            base_vals = set(g.loc[g["항목"] == "0801", base_col].astype(str).str.strip().tolist())
            base_vals.discard("")
            if base_code in base_vals:
                codes_for_date.add(base_code)
        if codes_for_date:
            out[str(rx_date)] = codes_for_date

    return out


def build_check_table(
    df_case: pd.DataFrame,
    rx_date: str,
    base_code: str,
    item: str,
    check_col: str,
    show_only_missing: bool,
) -> pd.DataFrame:
    """
    item: "0401" or "0801"
    check_col: "청구코드" or "처방코드" (처방 내 존재 여부 판단)
    """
    rule = RULES.get(base_code, {})
    rules_list = rule.get("rules_0401", []) if item == "0401" else rule.get("rules_0801", [])
    if not rules_list:
        return pd.DataFrame(columns=["✓", "코드", "청구코드", "처방코드", "코드명", "단가", "급비", "case_n"])

    g = df_case.copy()
    g["처방일자"] = g.get("처방일자", "").astype(str).str.strip()
    g["항목"] = g.get("항목", "").astype(str).str.strip()
    g[check_col] = g.get(check_col, "").astype(str).str.strip()

    dg = g[g["처방일자"] == str(rx_date)].copy()

    obs = set(dg.loc[dg["항목"] == item, check_col].astype(str).str.strip().tolist())
    obs.discard("")

    rows: List[dict] = []
    for r in rules_list:
        code = str(r.get("코드", "")).strip()
        is_present = (code in obs) if code else False
        rows.append(
            {
                "✓": is_present,
                "코드": code,
                "청구코드": str(r.get("청구코드", "")).strip(),
                "처방코드": str(r.get("처방코드", "")).strip(),
                "코드명": str(r.get("코드명", "")).strip(),
                "단가": str(r.get("단가", "")).strip(),
                "급비": str(r.get("급비", "")).strip(),
                "case_n": int(r.get("case_n", 0) or 0),
            }
        )

    out = pd.DataFrame(rows)
    if show_only_missing:
        out = out[out["✓"] == False].copy()

    out = out.sort_values(["✓", "case_n", "코드"], ascending=[True, False, True]).reset_index(drop=True)
    return out


def summarize_result(check_0401: pd.DataFrame, check_0801: pd.DataFrame) -> dict:
    def _cnt(df: pd.DataFrame):
        if df is None or df.empty:
            return (0, 0)
        total = int(len(df))
        ok = int(df["✓"].sum()) if "✓" in df.columns else 0
        miss = total - ok
        return total, miss

    t40, m40 = _cnt(check_0401)
    t80, m80 = _cnt(check_0801)
    return {
        "0401_total": t40,
        "0401_missing": m40,
        "0801_total": t80,
        "0801_missing": m80,
        "total_missing": m40 + m80,
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="규칙 누락 점검(하드코딩)", layout="wide")
st.title("규칙결과 하드코딩 → 처방 복붙만으로 0401/0801 점검 (각 파일 case_n 최대값 규칙만)")

with st.sidebar:
    st.subheader("점검 옵션")
    check_col = st.radio("처방에서 ‘있다/없다’ 판단 컬럼", ["청구코드", "처방코드"], index=0)
    show_only_missing = st.toggle("누락만 보기", value=False)
    st.divider()
    st.caption("※ 기준코드는 '항목=0801'에서 base_col에 등장해야 적용됩니다.")
    st.caption("※ RULES는 '각 파일의 case_n 최대값'과 동일한 행만 포함합니다.")

st.subheader("처방 복붙")
if "rx_raw" not in st.session_state:
    st.session_state["rx_raw"] = ""

cbtn, _ = st.columns([1, 5])
with cbtn:
    if st.button("🧹 입력창 비우기", use_container_width=True):
        st.session_state["rx_raw"] = ""
        st.rerun()

raw = st.text_area("표 그대로 붙여넣기(탭 구분)", height=220, key="rx_raw")

if not raw.strip():
    st.info("처방을 복붙하면 자동으로 기준코드를 판정하고 체크리스트를 보여줍니다.")
    st.stop()

df_case = parse_clipboard_tsv(raw)

# 날짜별 적용 기준코드 판정
applied_by_date = applied_base_codes_by_date(df_case)
applied_codes_all = sorted({c for s in applied_by_date.values() for c in s})

# ✅ 기준코드 목록(이번 복붙 적용 여부 색표시)
with st.expander("기준코드 목록 (이번 복붙에서 적용된 기준코드 색표시)", expanded=True):
    rows = []
    for base_code, rule in sorted(RULES.items(), key=lambda x: x[0]):
        rows.append(
            {
                "기준코드": base_code,
                "base_col": rule.get("base_col", ""),
                "case_n_max": rule.get("case_n_max", ""),
                "0401규칙수": len(rule.get("rules_0401", [])),
                "0801규칙수": len(rule.get("rules_0801", [])),
                "이번복붙_적용여부": (base_code in applied_codes_all),
            }
        )
    df_list = (
        pd.DataFrame(rows)
        .sort_values(["이번복붙_적용여부", "기준코드"], ascending=[False, True])
        .reset_index(drop=True)
    )

    def _hl(row):
        return ["background-color:#d1fae5"] * len(row) if bool(row.get("이번복붙_적용여부")) else [""] * len(row)

    st.dataframe(df_list.style.apply(_hl, axis=1), use_container_width=True)
    st.caption("이번 복붙에서 적용된 기준코드: " + (", ".join(applied_codes_all) if applied_codes_all else "(없음)"))

if not applied_by_date:
    st.warning("이번 복붙에서는 어떤 기준코드도(항목=0801 기준) 발견되지 않아 규칙을 적용하지 않았습니다.")
    st.stop()

st.divider()
st.subheader("점검 결과 (날짜 × 기준코드)")

# 날짜별 섹션
for rx_date in sorted(applied_by_date.keys()):
    codes = sorted(applied_by_date[rx_date])
    st.markdown(f"### 처방일자: {rx_date}  |  적용 기준코드: {', '.join(codes)}")

    for base_code in codes:
        colL, colR = st.columns(2)

        # 표시용(누락만 보기 옵션 적용)
        view_0401 = build_check_table(df_case, rx_date, base_code, "0401", check_col, show_only_missing)
        view_0801 = build_check_table(df_case, rx_date, base_code, "0801", check_col, show_only_missing)

        # 요약용(항상 전체 기준)
        full_0401 = build_check_table(df_case, rx_date, base_code, "0401", check_col, False)
        full_0801 = build_check_table(df_case, rx_date, base_code, "0801", check_col, False)
        summary = summarize_result(full_0401, full_0801)

        # ✅ 결론(요약)
        if summary["total_missing"] == 0:
            st.success(f"✅ 기준코드 {base_code}: 누락 없음 (0401 {summary['0401_total']}개 / 0801 {summary['0801_total']}개)")
        else:
            st.error(
                f"⚠️ 기준코드 {base_code}: 누락 {summary['total_missing']}개 "
                f"(0401 누락 {summary['0401_missing']}/{summary['0401_total']}, "
                f"0801 누락 {summary['0801_missing']}/{summary['0801_total']})"
            )

        with colL:
            st.markdown(f"**0401 체크리스트 — 기준코드 {base_code}**")
            st.dataframe(
                view_0401,
                use_container_width=True,
                column_config={
                    "✓": st.column_config.CheckboxColumn("✓", help="현재 처방(해당 날짜)에 존재하면 체크"),
                    "case_n": st.column_config.NumberColumn("case_n"),
                },
            )

        with colR:
            st.markdown(f"**0801 체크리스트 — 기준코드 {base_code}**")
            st.dataframe(
                view_0801,
                use_container_width=True,
                column_config={
                    "✓": st.column_config.CheckboxColumn("✓", help="현재 처방(해당 날짜)에 존재하면 체크"),
                    "case_n": st.column_config.NumberColumn("case_n"),
                },
            )

    st.divider()

# -----------------------------
# 다운로드 (openpyxl 없으면 CSV로 자동 대체)
# -----------------------------
st.subheader("다운로드")

out_rows = []
for rx_date, codes in applied_by_date.items():
    for base_code in codes:
        for item in ["0401", "0801"]:
            tbl = build_check_table(df_case, rx_date, base_code, item, check_col, False)
            if tbl.empty:
                continue
            tbl2 = tbl.copy()
            tbl2.insert(0, "항목", item)
            tbl2.insert(0, "기준코드", base_code)
            tbl2.insert(0, "처방일자", rx_date)
            out_rows.append(tbl2)

if not out_rows:
    st.info("다운로드할 체크리스트가 없습니다.")
    st.stop()

out_df = pd.concat(out_rows, ignore_index=True)

# 1) xlsx 시도
can_xlsx = True
try:
    import openpyxl  # noqa: F401
except Exception:
    can_xlsx = False

if can_xlsx:
    x = io.BytesIO()
    with pd.ExcelWriter(x, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="checklist")

    st.download_button(
        "📥 체크리스트 다운로드(Excel .xlsx)",
        data=x.getvalue(),
        file_name=f"체크리스트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
    st.warning("⚠️ openpyxl이 설치되어 있지 않아 XLSX 대신 CSV로 다운로드합니다. (requirements에 openpyxl 추가하세요)")
    st.download_button(
        "📥 체크리스트 다운로드(CSV)",
        data=csv_bytes,
        file_name=f"체크리스트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
