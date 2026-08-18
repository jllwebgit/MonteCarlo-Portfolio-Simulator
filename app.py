# -*- coding: utf-8 -*-
"""
モンテカルロ・ポートフォリオ・シミュレーター
ローカル実行専用（streamlit run app.py）

複数銘柄（アセット）の期待リターン・ボラティリティ・コスト・銘柄間相関、
および積立・取崩（キャッシュフロー）を入力し、モンテカルロ法で
将来の資産評価額の分布をシミュレーションする。

金額はすべて「万円」単位で入力・表示する。
設定（銘柄・相関・積立取崩・初期投資額など）はJSONファイルとして保存し、
サイドバーからアップロードして復元できる。
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="モンテカルロ・ポートフォリオ・シミュレーター", layout="wide")

st.title("📈 モンテカルロ・ポートフォリオ・シミュレーター")
st.caption(
    "複数銘柄間の相関・積立/取崩を考慮したモンテカルロ法により、"
    "将来の資産推移を確率的にシミュレーションします。"
    "本ツールはローカル環境での利用を想定しています（ネットワーク公開は非推奨）。金額はすべて万円単位です。"
)

ASSET_COLS = ["銘柄名", "投資金額(万円)", "投資比率(%)", "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]
ASSET_NUMERIC_COLS = ["投資金額(万円)", "投資比率(%)", "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]
CASHFLOW_COLS = ["種別", "金額(万円/年)", "開始年", "終了年"]
CASH_NAME = "円現預金"  # 生活防衛資金など、比率ではなく残額で維持する待機資金の銘柄名

# ============================================================
# 初期値（デフォルト銘柄セット・相関）
#
# 数値は「円建て・為替ヘッジなし」を前提とした参考値です。
#   - 期待リターン/ボラティリティ: GPIF「第5期中期目標期間における基本ポートフォリオ」
#     （2025年4月適用、名目期待リターン: 国内株式4〜7.5%・外国株式4.6〜8.1%・
#     国内債券-0.3〜3.2%・外国債券1.4〜4.9%、国内株式リスク約19%）等の公表値を参考に、
#     REIT・ゴールド・コモディティ・為替（USD/JPY）は一般的な長期統計を参考にした概算値です。
#     米ドル現預金（3.75%）・円現預金（1.0%）の期待リターンは指定値です。
#   - コスト: 各ファンドの信託報酬の概算値です（2026年8月時点、変更されている可能性があります）。
# 実際の投資判断の際は、必ず最新の目論見書・運用報告書でご確認のうえ、表内の数値を
# ご自身の見立てに合わせて修正してください（本ツールは将来の成果を保証するものではありません）。
# ============================================================
DEFAULT_ASSET_NAMES = [
    "全世界株式（オール・カントリー）",
    "国内株式（日経平均）",
    "米国株式（S&P500）",
    "国内債券インデックス",
    "先進国債券インデックス（除く日本）",
    "国内リートインデックス",
    "先進国リートインデックス（除く日本）",
    "ゴールド（為替ヘッジなし）",
    "コモディティインデックス",
    "453A：iシェアーズ 米国債20年超 プレミアムインカム ETF",
    "563A：グローバルX NASDAQ100・デイリー・カバード・コール ETF",
    "米ドル現預金",
    CASH_NAME,
]
DEFAULT_RETURNS = [6.5, 5.5, 7.0, 1.0, 3.0, 5.0, 5.5, 5.0, 3.5, 5.5, 7.0, 3.75, 1.0]
DEFAULT_VOLS = [17.0, 19.0, 18.0, 4.0, 12.0, 20.0, 21.0, 16.0, 18.0, 16.0, 19.0, 10.0, 0.0]
DEFAULT_COSTS = [0.06, 0.15, 0.08, 0.13, 0.10, 0.17, 0.22, 0.45, 0.55, 0.605, 0.28, 0.0, 0.0]
CASH_DEFAULT_RETURN = DEFAULT_RETURNS[DEFAULT_ASSET_NAMES.index(CASH_NAME)]


def build_default_assets_df() -> pd.DataFrame:
    """円現預金を除く銘柄の初期テーブル（投資金額はすべて0から開始）。
    円現預金は別枠（残額）で扱うため、このテーブルには含めない。"""
    rows = []
    for i, name in enumerate(DEFAULT_ASSET_NAMES):
        if name == CASH_NAME:
            continue
        rows.append(
            {
                "銘柄名": name,
                "投資金額(万円)": 0.0,
                "投資比率(%)": 0.0,
                "期待リターン(%)": DEFAULT_RETURNS[i],
                "ボラティリティ(%)": DEFAULT_VOLS[i],
                "コスト(%)": DEFAULT_COSTS[i],
            }
        )
    return pd.DataFrame(rows, columns=ASSET_COLS)


# 銘柄間相関（円建てベース）の参考値。キーに無い組み合わせは0（無相関）扱い。
# 円現預金はボラティリティ0のため、ここでの値の有無に関わらず計算結果には影響しません。
_N = DEFAULT_ASSET_NAMES
DEFAULT_CORR_PAIRS = {
    (_N[0], _N[1]): 0.75,   # 全世界株 - 国内株
    (_N[0], _N[2]): 0.97,   # 全世界株 - 米国株
    (_N[0], _N[3]): 0.05,   # 全世界株 - 国内債券
    (_N[0], _N[4]): 0.35,   # 全世界株 - 先進国債券
    (_N[0], _N[5]): 0.55,   # 全世界株 - 国内リート
    (_N[0], _N[6]): 0.65,   # 全世界株 - 先進国リート
    (_N[0], _N[7]): 0.10,   # 全世界株 - ゴールド
    (_N[0], _N[8]): 0.25,   # 全世界株 - コモディティ
    (_N[0], _N[9]): 0.20,   # 全世界株 - 453A(米国債カバコ)
    (_N[0], _N[10]): 0.85,  # 全世界株 - 563A(NAS100カバコ)
    (_N[0], _N[11]): 0.45,  # 全世界株 - 米ドル現預金

    (_N[1], _N[2]): 0.55,   # 国内株 - 米国株
    (_N[1], _N[3]): 0.00,   # 国内株 - 国内債券
    (_N[1], _N[4]): 0.15,   # 国内株 - 先進国債券
    (_N[1], _N[5]): 0.60,   # 国内株 - 国内リート
    (_N[1], _N[6]): 0.40,   # 国内株 - 先進国リート
    (_N[1], _N[7]): 0.00,   # 国内株 - ゴールド
    (_N[1], _N[8]): 0.20,   # 国内株 - コモディティ
    (_N[1], _N[9]): 0.10,   # 国内株 - 453A
    (_N[1], _N[10]): 0.50,  # 国内株 - 563A
    (_N[1], _N[11]): 0.20,  # 国内株 - 米ドル現預金

    (_N[2], _N[3]): 0.00,   # 米国株 - 国内債券
    (_N[2], _N[4]): 0.35,   # 米国株 - 先進国債券
    (_N[2], _N[5]): 0.40,   # 米国株 - 国内リート
    (_N[2], _N[6]): 0.65,   # 米国株 - 先進国リート
    (_N[2], _N[7]): 0.10,   # 米国株 - ゴールド
    (_N[2], _N[8]): 0.25,   # 米国株 - コモディティ
    (_N[2], _N[9]): 0.20,   # 米国株 - 453A
    (_N[2], _N[10]): 0.88,  # 米国株 - 563A
    (_N[2], _N[11]): 0.55,  # 米国株 - 米ドル現預金

    (_N[3], _N[4]): 0.15,   # 国内債券 - 先進国債券
    (_N[3], _N[5]): 0.25,   # 国内債券 - 国内リート
    (_N[3], _N[6]): 0.10,   # 国内債券 - 先進国リート
    (_N[3], _N[7]): 0.10,   # 国内債券 - ゴールド
    (_N[3], _N[8]): -0.05,  # 国内債券 - コモディティ
    (_N[3], _N[9]): 0.10,   # 国内債券 - 453A
    (_N[3], _N[10]): 0.00,  # 国内債券 - 563A
    (_N[3], _N[11]): 0.00,  # 国内債券 - 米ドル現預金

    (_N[4], _N[5]): 0.15,   # 先進国債券 - 国内リート
    (_N[4], _N[6]): 0.35,   # 先進国債券 - 先進国リート
    (_N[4], _N[7]): 0.20,   # 先進国債券 - ゴールド
    (_N[4], _N[8]): 0.05,   # 先進国債券 - コモディティ
    (_N[4], _N[9]): 0.60,   # 先進国債券 - 453A
    (_N[4], _N[10]): 0.30,  # 先進国債券 - 563A
    (_N[4], _N[11]): 0.55,  # 先進国債券 - 米ドル現預金

    (_N[5], _N[6]): 0.45,   # 国内リート - 先進国リート
    (_N[5], _N[7]): 0.05,   # 国内リート - ゴールド
    (_N[5], _N[8]): 0.15,   # 国内リート - コモディティ
    (_N[5], _N[9]): 0.15,   # 国内リート - 453A
    (_N[5], _N[10]): 0.35,  # 国内リート - 563A
    (_N[5], _N[11]): 0.15,  # 国内リート - 米ドル現預金

    (_N[6], _N[7]): 0.10,   # 先進国リート - ゴールド
    (_N[6], _N[8]): 0.20,   # 先進国リート - コモディティ
    (_N[6], _N[9]): 0.30,   # 先進国リート - 453A
    (_N[6], _N[10]): 0.60,  # 先進国リート - 563A
    (_N[6], _N[11]): 0.45,  # 先進国リート - 米ドル現預金

    (_N[7], _N[8]): 0.45,   # ゴールド - コモディティ
    (_N[7], _N[9]): 0.20,   # ゴールド - 453A
    (_N[7], _N[10]): 0.10,  # ゴールド - 563A
    (_N[7], _N[11]): 0.30,  # ゴールド - 米ドル現預金

    (_N[8], _N[9]): 0.05,   # コモディティ - 453A
    (_N[8], _N[10]): 0.20,  # コモディティ - 563A
    (_N[8], _N[11]): 0.25,  # コモディティ - 米ドル現預金

    (_N[9], _N[10]): 0.25,  # 453A - 563A
    (_N[9], _N[11]): 0.65,  # 453A - 米ドル現預金

    (_N[10], _N[11]): 0.50, # 563A - 米ドル現預金
    # 円現預金（_N[10]）はどの銘柄とも 0（ボラティリティ0のため設定しても結果に影響しません）
}


def build_default_corr(names_list) -> pd.DataFrame:
    """既知の標準銘柄名同士は参考相関値を、それ以外は無相関(0)をセットした相関行列を返す。"""
    df = pd.DataFrame(np.eye(len(names_list)), index=names_list, columns=names_list, dtype=float)
    for (a, b), v in DEFAULT_CORR_PAIRS.items():
        if a in df.index and b in df.columns:
            df.loc[a, b] = v
            df.loc[b, a] = v
    return df


# ============================================================
# 汎用ヘルパー
# ============================================================
def is_named(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return str(x).strip() != ""


def dedupe_names(names_list):
    seen = {}
    out = []
    for nm in names_list:
        nm = str(nm).strip()
        if nm in seen:
            seen[nm] += 1
            out.append(f"{nm}#{seen[nm]}")
        else:
            seen[nm] = 1
            out.append(nm)
    return out


def fmt_man(x, digits=0):
    return f"{x:,.{digits}f} 万円"


# ============================================================
# 設定の保存・読込（シナリオ全体をJSONで保存/復元）
# ============================================================
def build_scenario_json() -> str:
    """現在の設定（銘柄・相関・積立取崩・初期投資額など）をまとめてJSON文字列にする。"""
    assets_export = st.session_state.get("assets_full_export", build_default_assets_df()).copy()
    assets_export = assets_export.rename(
        columns={
            "銘柄名": "name",
            "投資金額(万円)": "amount",
            "投資比率(%)": "ratio",
            "期待リターン(%)": "expected_return",
            "ボラティリティ(%)": "volatility",
            "コスト(%)": "cost",
        }
    )

    corr_state = st.session_state.get("corr_df")
    corr_export = None
    if corr_state is not None and not corr_state.empty:
        corr_export = {
            "names": list(corr_state.columns),
            "matrix": corr_state.to_numpy(dtype=float).tolist(),
        }

    cashflow_state = st.session_state.get("cashflow_df", pd.DataFrame(columns=CASHFLOW_COLS))
    cashflow_export = cashflow_state.rename(
        columns={"種別": "type", "金額(万円/年)": "amount", "開始年": "start_year", "終了年": "end_year"}
    ).to_dict(orient="records")

    data = {
        "version": 1,
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "initial_investment": st.session_state.get("initial_investment_input", 2000.0),
        "years": st.session_state.get("years_input", 30),
        "target_amount": st.session_state.get("target_amount_input", 4000.0),
        "n_sims": st.session_state.get("n_sims_input", 10000),
        "rebalance": st.session_state.get("rebalance_input", True),
        "seed": st.session_state.get("seed_input", 0),
        "cash_expected_return": st.session_state.get("cash_return_input", CASH_DEFAULT_RETURN),
        "assets": assets_export.to_dict(orient="records"),
        "correlation": corr_export,
        "cashflow": cashflow_export,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def apply_scenario(data: dict) -> None:
    """アップロードされたJSONの内容をセッションへ反映する（ウィジェット生成前に呼ぶこと）。"""
    st.session_state["initial_investment_input"] = float(data.get("initial_investment", 2000.0))
    st.session_state["years_input"] = int(data.get("years", 30))
    st.session_state["target_amount_input"] = float(data.get("target_amount", 4000.0))
    st.session_state["n_sims_input"] = int(data.get("n_sims", 10000))
    st.session_state["rebalance_input"] = bool(data.get("rebalance", True))
    st.session_state["seed_input"] = int(data.get("seed", 0))
    st.session_state["cash_return_input"] = float(data.get("cash_expected_return", CASH_DEFAULT_RETURN))

    # 3つのdata_editorのbaselineを強制的に作り直させる（そうしないと、アップロードした内容が
    # 画面に反映されない）。各editorのbaseline構築条件を「不一致」にするため、いったん削除する。
    st.session_state.pop("assets_editor_baseline", None)
    st.session_state.pop("_assets_baseline_key", None)
    st.session_state.pop("corr_editor_baseline", None)
    st.session_state.pop("cashflow_editor_baseline", None)

    # data_editorはkeyが同じままだと、baselineを差し替えても内部に溜まった編集差分
    # （行の追加・削除・編集）を新しいbaselineに上書き適用してしまい、特に③積立・取崩の
    # ように行数が変わる場合に読込内容が正しく反映されないことがある（Streamlitの既知の挙動）。
    # 読込のたびに世代カウンタを進め、各editorのkeyに含めることで、読込時は必ず新規の
    # data_editorとして生成させ、古い編集差分が引き継がれないようにする。
    st.session_state["_scenario_gen"] = st.session_state.get("_scenario_gen", 0) + 1

    rows = []
    for a in data.get("assets") or []:
        rows.append(
            {
                "銘柄名": str(a.get("name", "")),
                "投資金額(万円)": float(a.get("amount", 0.0)),
                "投資比率(%)": float(a.get("ratio", 0.0)),
                "期待リターン(%)": float(a.get("expected_return", 0.0)),
                "ボラティリティ(%)": float(a.get("volatility", 0.0)),
                "コスト(%)": float(a.get("cost", 0.0)),
            }
        )
    if rows:
        st.session_state.assets_df = pd.DataFrame(rows, columns=ASSET_COLS)

    corr = data.get("correlation")
    if corr and corr.get("names") and corr.get("matrix"):
        try:
            names_loaded = [str(n) for n in corr["names"]]
            mat = np.array(corr["matrix"], dtype=float)
            if mat.shape == (len(names_loaded), len(names_loaded)):
                st.session_state.corr_df = pd.DataFrame(mat, index=names_loaded, columns=names_loaded)
        except Exception:
            pass

    cf_rows = []
    for c in data.get("cashflow") or []:
        cf_rows.append(
            {
                "種別": str(c.get("type", "積立")),
                "金額(万円/年)": float(c.get("amount", 0.0)),
                "開始年": int(c.get("start_year", 1)),
                "終了年": int(c.get("end_year", 30)),
            }
        )
    if cf_rows:
        st.session_state.cashflow_df = pd.DataFrame(cf_rows, columns=CASHFLOW_COLS)


with st.sidebar:
    st.header("💾 設定の保存・読込")
    uploaded_scenario = st.file_uploader(
        "保存したファイル（JSON）をアップロードして読み込む", type=["json"], key="scenario_uploader"
    )
    if uploaded_scenario is not None:
        file_sig = f"{uploaded_scenario.name}:{uploaded_scenario.size}"
        if st.session_state.get("_last_loaded_scenario_sig") != file_sig:
            try:
                loaded_data = json.load(uploaded_scenario)
                apply_scenario(loaded_data)
                st.session_state["_last_loaded_scenario_sig"] = file_sig
                st.success("設定を読み込みました。")
                st.rerun()
            except Exception as e:
                st.error(f"読み込みに失敗しました: {e}")
    st.caption(
        "銘柄（投資金額・比率含む）・相関・積立取崩・初期投資額などをまとめて復元します。"
        "保存は画面下部の「💾 現在の設定を保存」から行えます。"
    )

# ============================================================
# サイドバー：基本設定
# ============================================================
with st.sidebar:
    st.header("基本設定（金額は万円単位）")
    initial_investment = st.number_input(
        "初期投資額（万円）", min_value=0.0, value=2000.0, step=10.0, format="%.0f", key="initial_investment_input"
    )
    years = st.number_input(
        "投資年数（年）", min_value=1, max_value=100, value=30, step=1, key="years_input"
    )
    target_amount = st.number_input(
        "目標金額（万円）",
        min_value=0.0,
        value=4000.0,
        step=10.0,
        format="%.0f",
        key="target_amount_input",
        help="「目標額到達確率」の算出に使用します。",
    )
    n_sims = st.number_input(
        "シミュレーション回数", min_value=100, max_value=200_000, value=10_000, step=1000, key="n_sims_input"
    )
    rebalance = st.checkbox("毎年リバランスする（投資比率を維持）", value=True, key="rebalance_input")
    seed = st.number_input(
        "乱数シード（0 = 毎回ランダム）", min_value=0, value=0, step=1, key="seed_input",
        help="同じ条件で再現性のある結果を得たい場合は 1 以上を指定してください。",
    )
    st.caption("※ コストは各銘柄の期待リターンから毎年差し引かれる前提です（簡易モデル）。")

years = int(years)

# ============================================================
# ① 銘柄設定
# ============================================================
st.subheader("① 銘柄設定")

if "assets_df" not in st.session_state:
    st.session_state.assets_df = build_default_assets_df()

input_mode = st.radio(
    "入力単位",
    ["投資金額（万円）で入力", "投資比率（%）で入力"],
    horizontal=True,
    key="asset_input_mode",
)
amt_editable = input_mode.startswith("投資金額")
active_col = "投資金額(万円)" if amt_editable else "投資比率(%)"
inactive_col = "投資比率(%)" if amt_editable else "投資金額(万円)"
view_cols = ["銘柄名", active_col, "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]
_scenario_gen = st.session_state.get("_scenario_gen", 0)
editor_key = f"assets_editor_{'amt' if amt_editable else 'ratio'}_{_scenario_gen}"

st.caption(
    f"{active_col} の列を編集してください。"
    f"「{CASH_NAME}」は生活防衛資金など、比率ではなく金額で維持する待機資金として下に別枠で表示します。"
)

# 不具合の真因（訂正）: キーのローテーションではなく、「data_editorが返した値（編集結果）を、
# 同じキーのdata_editorへ次回もそのまま渡し直していたこと」自体が原因だった。
# data_editorはキーごとに内部で編集差分を自動的に積み上げて管理しているため、
# こちらが「その差分適用済みの結果」を毎回valueとして渡し直すと、差分が二重に解釈され、
# 直後の別セルへの編集を取りこぼすことがある（Streamlitの既知のアンチパターン）。
#
# 対策: このwidgetに渡す value（baseline）は「入力モードが変わった時」だけ作り直し、
# それ以外のあらゆる再実行では絶対に書き換えない。編集結果はdata_editorの戻り値からのみ読み取り、
# 計算・保存用の別データ（assets_full 等）にのみ反映する。
if st.session_state.get("_assets_baseline_key") != editor_key:
    source = st.session_state.assets_df.copy()
    if active_col not in source.columns:
        if inactive_col in source.columns:
            if amt_editable:
                source[active_col] = source[inactive_col] / 100.0 * initial_investment
            else:
                source[active_col] = source[inactive_col] / initial_investment * 100.0 if initial_investment > 0 else 0.0
        else:
            source[active_col] = 0.0
    st.session_state["assets_editor_baseline"] = source[view_cols].reset_index(drop=True)
    st.session_state["_assets_baseline_key"] = editor_key

assets_edit = st.data_editor(
    st.session_state["assets_editor_baseline"],
    num_rows="dynamic",
    use_container_width=True,
    key=editor_key,
    column_config={
        "銘柄名": st.column_config.TextColumn(required=True),
        active_col: st.column_config.NumberColumn(format="%.1f" if amt_editable else "%.2f"),
        "期待リターン(%)": st.column_config.NumberColumn(format="%.2f"),
        "ボラティリティ(%)": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
        "コスト(%)": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
    },
)

# NaNを残さずクリーニング（コピー上で行う。baseline自体は一切書き換えない）
assets_clean = assets_edit.copy()
assets_clean["銘柄名"] = assets_clean["銘柄名"].fillna("").astype(str)
for c in [active_col, "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]:
    assets_clean[c] = pd.to_numeric(assets_clean[c], errors="coerce").fillna(0.0).astype("float64")
assets_clean = assets_clean.reset_index(drop=True)

# 表示・保存・計算用のフル版（投資金額・投資比率の両方を保持）。baselineには一切書き戻さない。
assets_full = assets_clean.copy()
if amt_editable:
    assets_full["投資比率(%)"] = (
        assets_full["投資金額(万円)"] / initial_investment * 100.0 if initial_investment > 0 else 0.0
    )
else:
    assets_full["投資金額(万円)"] = assets_full["投資比率(%)"] / 100.0 * initial_investment
assets_full = assets_full[ASSET_COLS]

st.session_state.assets_df = assets_full  # 次のモード切替・保存用（このwidgetのbaselineには使わない）
st.session_state["assets_full_export"] = assets_full  # JSON保存用（フル版、円現預金は含まない）

assets_df = assets_full

# 円現預金は別枠：他の銘柄の投資金額合計を初期投資額から差し引いた残額として表示専用で算出する
other_sum = float(assets_df["投資金額(万円)"].sum())
cash_amount = max(initial_investment - other_sum, 0.0)
cash_ratio = cash_amount / initial_investment * 100.0 if initial_investment > 0 else 0.0

cc1, cc2, cc3 = st.columns([2, 1, 1])
with cc1:
    cash_return = st.number_input(
        f"{CASH_NAME}の期待リターン（%）",
        value=float(st.session_state.get("cash_return_input", CASH_DEFAULT_RETURN)),
        step=0.05,
        format="%.2f",
        key="cash_return_input",
    )
with cc2:
    st.metric(f"{CASH_NAME}（自動計算・待機資金）", fmt_man(cash_amount))
with cc3:
    st.metric(f"{CASH_NAME}の投資比率", f"{cash_ratio:.1f}%")

if other_sum > initial_investment + 1e-9:
    st.warning(
        f"「{CASH_NAME}」以外の銘柄の合計投資額（{fmt_man(other_sum)}）が初期投資額（{fmt_man(initial_investment)}）"
        f"を超えています。{CASH_NAME}は0として計算し、実際の投資金額合計は初期投資額を超過した状態でシミュレーションします。"
    )

# 銘柄名が入力されている行（円現預金以外）＋円現預金（残額）を、以降の計算対象とする
valid_mask = assets_df["銘柄名"].apply(is_named)
assets_noncash_valid = assets_df[valid_mask].reset_index(drop=True)
cash_row = pd.DataFrame(
    [
        {
            "銘柄名": CASH_NAME,
            "投資金額(万円)": cash_amount,
            "投資比率(%)": cash_ratio,
            "期待リターン(%)": cash_return,
            "ボラティリティ(%)": 0.0,
            "コスト(%)": 0.0,
        }
    ]
)
assets_valid = pd.concat([assets_noncash_valid, cash_row], ignore_index=True)

names = dedupe_names(assets_valid["銘柄名"].tolist())

# ============================================================
# ② 銘柄間相関
# ============================================================
st.subheader("② 銘柄間の相関")
st.caption("対角成分は1、行列は対称（相関係数は -1〜1）である必要があります。相関係数は円建てリターンを基準とした値です。")
st.caption(
    "💡 円現預金はボラティリティ0のため、相関係数を何に設定しても計算結果には影響しません（設定不要）。"
    "一方、米ドル現預金は円換算では実質的に為替（USD/JPY）そのものへのエクスポージャーであり、"
    "先進国債券・先進国リート・米国株式など為替ヘッジなしの外貨建て資産と連動するため、"
    "相関を設定する意味があります（初期値は参考値としてあらかじめ入力済みです）。"
)

if "corr_editor_baseline" not in st.session_state or list(st.session_state["corr_editor_baseline"].columns) != names:
    old = st.session_state.get("corr_df")
    new_corr = build_default_corr(names)
    if old is not None and not old.empty:
        common = [c for c in names if c in old.columns and c in old.index]
        for a in common:
            for b in common:
                try:
                    val = old.loc[a, b]
                    if isinstance(val, (pd.Series, pd.DataFrame)):
                        continue
                    if pd.notna(val):
                        new_corr.loc[a, b] = float(val)
                except Exception:
                    pass
    st.session_state["corr_editor_baseline"] = new_corr

corr_edit = st.data_editor(
    st.session_state["corr_editor_baseline"],
    use_container_width=True,
    key=f"corr_editor_{_scenario_gen}",
    column_config={
        nm: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05, format="%.2f")
        for nm in names
    },
)

# NaN・範囲外を補正した「解決済み」値は計算専用（corr_df）に保存し、baselineには書き戻さない
# （baselineを毎回書き換えると、直後の編集を取りこぼす不具合の原因になるため）
corr_clean = corr_edit.apply(pd.to_numeric, errors="coerce").fillna(0.0)
corr_clean = corr_clean.clip(lower=-1.0, upper=1.0)
# pandasのCopy-on-Write設定下では .values / to_numpy() が読み取り専用配列を返すことがあるため、
# 書き込み可能な配列として明示的にコピーしてから対角成分を設定する
corr_arr = np.array(corr_clean.to_numpy(dtype=float), copy=True)
np.fill_diagonal(corr_arr, 1.0)
corr_clean = pd.DataFrame(corr_arr, index=corr_clean.index, columns=corr_clean.columns)
st.session_state.corr_df = corr_clean
corr_df = corr_clean

# ============================================================
# ③ 積立・取崩
# ============================================================
st.subheader("③ 積立・取崩")
st.caption(
    "毎年の積立や取崩しを複数行登録できます。開始年〜終了年（投資年数を1年目とする）の間、"
    "毎年その金額を加算/減算します。金額は万円単位、その年の運用開始前に反映されます。"
)

if "cashflow_editor_baseline" not in st.session_state:
    uploaded_cf = st.session_state.get("cashflow_df")
    if uploaded_cf is not None and not uploaded_cf.empty:
        st.session_state["cashflow_editor_baseline"] = uploaded_cf.reset_index(drop=True)
    else:
        st.session_state["cashflow_editor_baseline"] = pd.DataFrame(
            {
                "種別": ["積立"],
                "金額(万円/年)": [0.0],
                "開始年": [1],
                "終了年": [years],
            }
        )

cashflow_edit = st.data_editor(
    st.session_state["cashflow_editor_baseline"],
    num_rows="dynamic",
    use_container_width=True,
    key=f"cashflow_editor_{_scenario_gen}",
    column_config={
        "種別": st.column_config.SelectboxColumn(options=["積立", "取崩"], required=True),
        "金額(万円/年)": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
        "開始年": st.column_config.NumberColumn(min_value=1, max_value=100, step=1, format="%d"),
        "終了年": st.column_config.NumberColumn(min_value=1, max_value=100, step=1, format="%d"),
    },
)

# 解決済み値は計算専用（cashflow_df）に保存し、baselineには書き戻さない
cashflow_clean = cashflow_edit.copy()
cashflow_clean["種別"] = cashflow_clean["種別"].fillna("積立")
cashflow_clean.loc[~cashflow_clean["種別"].isin(["積立", "取崩"]), "種別"] = "積立"
cashflow_clean["金額(万円/年)"] = pd.to_numeric(cashflow_clean["金額(万円/年)"], errors="coerce").fillna(0.0)
cashflow_clean["開始年"] = pd.to_numeric(cashflow_clean["開始年"], errors="coerce").fillna(1).astype(int)
cashflow_clean["終了年"] = pd.to_numeric(cashflow_clean["終了年"], errors="coerce").fillna(years).astype(int)
st.session_state.cashflow_df = cashflow_clean
cashflow_df = cashflow_clean


def build_cashflow_array(cf_df: pd.DataFrame, T: int) -> np.ndarray:
    flow = np.zeros(T)
    for _, row in cf_df.iterrows():
        amt = float(row["金額(万円/年)"])
        if amt == 0:
            continue
        start = max(1, int(row["開始年"]))
        end = min(T, int(row["終了年"]))
        if start > end:
            continue
        sign = 1.0 if row["種別"] == "積立" else -1.0
        flow[start - 1:end] += sign * amt
    return flow


# ============================================================
# 💾 現在の設定を保存
# ============================================================
st.subheader("💾 現在の設定を保存")
st.caption(
    "銘柄（投資金額・比率含む）・相関・積立取崩・初期投資額・投資年数・目標金額などをまとめてJSONファイルに"
    "保存できます。次回起動時はサイドバーの「保存したファイル（JSON）をアップロードして読み込む」から"
    "このファイルを選ぶと、すぐに今の状態を復元できます。"
)
st.download_button(
    "💾 現在の設定をJSONファイルとしてダウンロード",
    data=build_scenario_json(),
    file_name=f"portfolio_scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
    mime="application/json",
)

# ============================================================
# ④ 実行
# ============================================================
run = st.button("▶ シミュレーション実行", type="primary")

if run:
    amounts_raw = assets_valid["投資金額(万円)"].to_numpy(dtype=float)
    if amounts_raw.sum() <= 0:
        st.error("投資金額の合計が0です。① 銘柄設定で投資金額（または投資比率）を入力してください。")
        st.stop()
    weights = amounts_raw / amounts_raw.sum()
    actual_initial = float(amounts_raw.sum())  # 実際の投資金額合計（万円）
    mu = (
        assets_valid["期待リターン(%)"].to_numpy(dtype=float)
        - assets_valid["コスト(%)"].to_numpy(dtype=float)
    ) / 100.0
    sigma = assets_valid["ボラティリティ(%)"].to_numpy(dtype=float) / 100.0

    corr = corr_df.to_numpy(dtype=float, copy=True)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sigma, sigma) * corr

    rng = np.random.default_rng(None if seed == 0 else int(seed))

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(cov)
        eigval_clipped = np.clip(eigval, 1e-10, None)
        cov = eigvec @ np.diag(eigval_clipped) @ eigvec.T
        L = np.linalg.cholesky(cov)

    M = int(n_sims)
    T = int(years)
    N = len(weights)

    z = rng.standard_normal(size=(M, T, N))
    correlated = z @ L.T
    asset_returns = mu + correlated  # (M, T, N)

    flow_arr = build_cashflow_array(cashflow_df, T)  # 万円/年, 長さT
    has_cashflow = bool(np.any(flow_arr != 0))

    asset_values = np.zeros((M, N))
    asset_values[:, :] = amounts_raw[None, :]
    path_list = [asset_values.sum(axis=1)]

    for t in range(T):
        net_flow = flow_arr[t]
        if net_flow >= 0:
            asset_values = asset_values + net_flow * weights[None, :]
        else:
            total = asset_values.sum(axis=1, keepdims=True)
            total_safe = np.where(total > 1e-9, total, 1.0)
            proportion = asset_values / total_safe
            asset_values = asset_values + net_flow * proportion
        asset_values = np.maximum(asset_values, 0.0)

        growth = np.maximum(1.0 + asset_returns[:, t, :], 0.0)
        asset_values = asset_values * growth

        if rebalance:
            total_after = asset_values.sum(axis=1, keepdims=True)
            asset_values = total_after * weights[None, :]

        asset_values = np.maximum(asset_values, 0.0)
        path_list.append(asset_values.sum(axis=1))

    paths_full = np.stack(path_list, axis=1)  # (M, T+1) 万円単位
    final_values = paths_full[:, -1]

    running_max = np.maximum.accumulate(paths_full, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_max > 0, (paths_full - running_max) / running_max, 0.0)
    max_drawdown = drawdowns.min(axis=1)

    principal_base = actual_initial + float(flow_arr.sum())  # 累計投入元本（万円）

    expected_return_simple = float(
        np.dot(
            weights,
            (
                assets_valid["期待リターン(%)"].to_numpy(dtype=float)
                - assets_valid["コスト(%)"].to_numpy(dtype=float)
            ),
        )
    )

    # ポートフォリオ全体のリスク（年率標準偏差、入力値ベースの分析的な計算）
    portfolio_variance = float(weights @ cov @ weights)
    portfolio_risk_pct = float(np.sqrt(max(portfolio_variance, 0.0))) * 100.0

    # シャープレシオ = (期待リターン - 無リスク金利) / リスク。無リスク金利は円現預金の期待リターンを使用
    risk_free_rate_pct = float(cash_return)
    if portfolio_risk_pct > 1e-9:
        sharpe_ratio = (expected_return_simple - risk_free_rate_pct) / portfolio_risk_pct
    else:
        sharpe_ratio = float("nan")

    if not has_cashflow and actual_initial > 0:
        with np.errstate(invalid="ignore"):
            cagr = np.where(final_values > 0, (final_values / actual_initial) ** (1.0 / T) - 1.0, -1.0)
        mean_cagr = float(np.mean(cagr))
    else:
        mean_cagr = None

    mean_final = float(np.mean(final_values))
    median_final = float(np.median(final_values))
    # 極端な外れ値を避けるため、絶対的な最小値・最大値ではなく、全試行の99%が収まる範囲
    # （下位1%タイル〜上位99%タイル）を「最低額・最高額」として使用する
    low_final = float(np.percentile(final_values, 1.0))
    high_final = float(np.percentile(final_values, 99.0))
    # 最頻値（連続値のため、ヒストグラムで最も度数の多い区間の中央値を推定値として使用）
    hist_counts, hist_edges = np.histogram(final_values, bins=60)
    mode_bin = int(np.argmax(hist_counts))
    mode_final = float((hist_edges[mode_bin] + hist_edges[mode_bin + 1]) / 2.0)

    prob_loss = float(np.mean(final_values < principal_base)) if principal_base > 0 else float("nan")
    prob_target = float(np.mean(final_values >= target_amount)) if target_amount > 0 else float("nan")

    mean_mdd = float(np.mean(max_drawdown))
    median_mdd = float(np.median(max_drawdown))
    worst_mdd = float(np.percentile(max_drawdown, 5))

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------
    st.subheader("④ 結果")

    st.markdown("##### ポートフォリオ全体の前提（年率・入力値ベース）")
    p1, p2, p3 = st.columns(3)
    p1.metric("期待リターン（加重平均）", f"{expected_return_simple:.2f}%")
    p2.metric("リスク（標準偏差）", f"{portfolio_risk_pct:.2f}%")
    p3.metric(
        "シャープレシオ",
        f"{sharpe_ratio:.2f}" if np.isfinite(sharpe_ratio) else "—",
        help=f"無リスク金利として{CASH_NAME}の期待リターン（{risk_free_rate_pct:.2f}%）を使用して算出しています。",
    )

    st.markdown("##### シミュレーション結果（確率）")
    c1, c2, c3 = st.columns(3)
    if mean_cagr is not None:
        c1.metric("シミュレーション平均CAGR", f"{mean_cagr * 100:.2f}%")
    else:
        c1.metric("シミュレーション平均CAGR", "—（積立/取崩ありのため非表示）")
    c2.metric("元本割れ確率", f"{prob_loss * 100:.1f}%")
    c3.metric("目標額到達確率", f"{prob_target * 100:.1f}%")
    st.caption(f"※ 元本割れ確率は累計投入元本（初期投資額 + 積立 − 取崩 = {fmt_man(principal_base)}）との比較です。")

    st.markdown("##### 最終資産額の分布（万円）")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("平均値", fmt_man(mean_final))
    d2.metric("中央値", fmt_man(median_final))
    d3.metric("最頻値（推定）", fmt_man(mode_final))
    d4.metric("最低額（1%～範囲）", fmt_man(low_final))
    d5.metric("最高額（～99%範囲）", fmt_man(high_final))
    st.caption(
        "※ 最頻値はシミュレーション結果をヒストグラム化し、最も度数の多い区間の中央値を推定値としたものです。"
        "最低額・最高額は全試行のうち極端な外れ値0.5%ずつを除いた、確率99%が収まる範囲"
        "（下位0.5%タイル〜上位99.5%タイル）です。"
    )

    st.markdown("##### 最大ドローダウン")
    e1, e2, e3 = st.columns(3)
    e1.metric("平均", f"{mean_mdd * 100:.1f}%")
    e2.metric("中央値", f"{median_mdd * 100:.1f}%")
    e3.metric("悪化5%タイル", f"{worst_mdd * 100:.1f}%")

    # --------------------------------------------------------
    # ファンチャート（資産推移の分布）
    # --------------------------------------------------------
    st.markdown("#### 資産推移の分布（ファンチャート、単位：万円）")
    pct_levels = [5, 25, 50, 75, 95]
    percentiles = np.percentile(paths_full, pct_levels, axis=0)
    x = np.arange(0, T + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=percentiles[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=percentiles[0], fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
            line=dict(width=0), name="5-95%タイル",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=percentiles[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=percentiles[1], fill="tonexty", fillcolor="rgba(99,110,250,0.35)",
            line=dict(width=0), name="25-75%タイル",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=percentiles[2], line=dict(color="royalblue", width=2), name="中央値"))
    fig.add_hline(y=target_amount, line_dash="dash", line_color="green", annotation_text="目標金額")
    fig.add_hline(y=actual_initial, line_dash="dot", line_color="gray", annotation_text="初期投資額")
    if has_cashflow:
        fig.add_hline(y=principal_base, line_dash="dashdot", line_color="orange", annotation_text="累計投入元本")
    fig.update_layout(
        xaxis_title="経過年数",
        yaxis_title="資産評価額（万円）",
        yaxis=dict(tickformat=",.0f", separatethousands=True),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # 最終資産額のヒストグラム
    # --------------------------------------------------------
    st.markdown("#### 最終資産額の分布（単位：万円）")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=final_values, nbinsx=500, marker_color="royalblue"))
    fig2.add_vline(x=actual_initial, line_dash="dot", line_color="gray", annotation_text="初期投資額")
    fig2.add_vline(x=target_amount, line_dash="dash", line_color="green", annotation_text="目標金額")
    if has_cashflow:
        fig2.add_vline(x=principal_base, line_dash="dashdot", line_color="orange", annotation_text="累計投入元本")
    fig2.update_layout(
        xaxis_title="最終資産評価額（万円）",
        xaxis=dict(tickformat=",.0f", separatethousands=True),
        yaxis_title="頻度",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------------
    # 最大ドローダウンのヒストグラム
    # --------------------------------------------------------
    st.markdown("#### 最大ドローダウンの分布")
    st.caption("キャッシュフロー（積立/取崩）による資産評価額の増減も含みます。")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=max_drawdown * 100, nbinsx=500, marker_color="indianred"))
    fig3.update_layout(xaxis_title="最大ドローダウン（%）", yaxis_title="頻度", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("シミュレーション設定の詳細"):
        st.write(f"銘柄ごとの投資金額・比率（実際の投資金額合計: {fmt_man(actual_initial)}）")
        st.dataframe(pd.DataFrame({"銘柄名": names, "投資金額(万円)": amounts_raw, "比率(%)": weights * 100}))
        st.write("使用した相関行列（対称化・対角=1に補正済み）")
        st.dataframe(pd.DataFrame(corr, index=names, columns=names))
        st.write("積立・取崩スケジュール（年別ネットキャッシュフロー、万円）")
        st.dataframe(pd.DataFrame({"年": np.arange(1, T + 1), "ネットCF(万円)": flow_arr}))
        st.write(
            f"モデル: 年次ステップ, {'毎年リバランスあり' if rebalance else 'リバランスなし（バイ&ホールド）'}, "
            f"各銘柄は多変量正規分布に従う年次リターンを仮定。キャッシュフローは年始（当年の運用前）に反映。"
        )
else:
    st.info("① 銘柄設定・② 相関・③ 積立取崩を確認し、「▶ シミュレーション実行」ボタンを押してください。")
