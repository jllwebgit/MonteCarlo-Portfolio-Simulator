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

try:
    from scipy.optimize import minimize as _scipy_minimize

    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_minimize = None
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="モンテカルロ・ポートフォリオ・シミュレーター", layout="wide")

st.title("📈 モンテカルロ・ポートフォリオ・シミュレーター")
st.caption(
    "複数銘柄間の相関・積立/取崩を考慮したモンテカルロ法により、"
    "将来の資産推移を確率的にシミュレーションします。"
)
st.info(
    "📱 スマートフォンでご覧の場合：画面左上の「››」アイコンをタップするとサイドバーが開き、"
    "基本設定（投資年数・目標金額など）と設定の保存・読込メニューを操作できます。",
    icon="📱",
)

ASSET_COLS = ["銘柄名", "投資金額(万円)", "投資比率(%)", "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]
ASSET_NUMERIC_COLS = ["投資金額(万円)", "投資比率(%)", "期待リターン(%)", "ボラティリティ(%)", "コスト(%)"]
CASHFLOW_COLS = ["種別", "金額(万円/年)", "開始年", "終了年"]
CASH_NAME = "円現預金"  # 生活防衛資金など、比率ではなく残額で維持する待機資金の銘柄名

# ============================================================
# 初期値（デフォルト銘柄セット・相関）
#
# 数値は「円建て・為替ヘッジなし」を前提とした参考値です。
# 実際の投資判断の際は、必ず最新の目論見書・運用報告書でご確認のうえ、表内の数値を
# 見立てに合わせて修正してください（本ツールは将来の成果を保証するものではありません）。
# ============================================================
DEFAULT_ASSET_NAMES = [
    "全世界株式（オール・カントリー）",
    "国内株式（日経平均）",
    "国内株式(JPX日経400)",
    "米国株式（S&P500）",
    "国内債券インデックス",
    "先進国債券インデックス（除く日本）",
    "国内リートインデックス",
    "先進国リートインデックス（除く日本）",
    "ゴールド（為替ヘッジなし）",
    "コモディティインデックス",
    "SBI・iシェアーズ・インド株式インデックス・ファンド",
    "SBI-Man リキッド・トレンド・ファンド",
    "453A：iシェアーズ 米国債20年超 プレミアムインカム ETF",
    "563A：グローバルX NASDAQ100・デイリー・カバード・コール ETF",
    "米ドル現預金",
    CASH_NAME,
]
# 2026年10月時点の長期モンテカルロシミュレーション値。
DEFAULT_RETURNS = [
    6.5,  # 全世界株式
    6.3,  # 国内株式（日経平均）
    6.2,  # 国内株式(JPX日経400)
    6.5,  # 米国株式（S&P500）
    2.2,  # 国内債券
    4.0,  # 先進国債券（除く日本）
    4.5,  # 国内リート
    6.0,  # 先進国リート（除く日本）
    5.5,  # ゴールド
    4.5,  # コモディティ
    7.0,  # インド株式
    4.0,  # リキッド・トレンド
    5.2,  # 453A
    6.7,  # 563A
    4.0,  # 米ドル現預金（円換算）
    1.3,  # 円現預金
]
DEFAULT_VOLS = [
    16.0, # 全世界株式
    17.0, # 国内株式（日経平均）
    15.0, # 国内株式(JPX日経400)
    17.0, # S&P500
    3.5,  # 国内債券
    8.0,  # 先進国債券
    12.0, # 国内リート
    17.0, # 先進国リート
    17.0, # ゴールド
    20.0, # コモディティ
    20.0, # インド株式
    15.0, # リキッド・トレンド
    18.0, # 453A
    19.0, # 563A
    9.0,  # 米ドル現預金（円換算）
    0.0,  # 円現預金
]
DEFAULT_COSTS = [0.06, 0.15, 0.22, 0.08, 0.13, 0.10, 0.17, 0.22, 0.45, 0.55, 0.31, 0.99, 0.605, 0.28, 0.0, 0.0]
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
    (_N[0], _N[1]): 0.70,    # 全世界株 - 国内株（日経平均）
    (_N[0], _N[2]): 0.70,    # 全世界株 - 国内株（JPX日経400）
    (_N[0], _N[3]): 0.95,    # 全世界株 - 米国株
    (_N[0], _N[4]): 0.10,    # 全世界株 - 国内債券
    (_N[0], _N[5]): 0.40,    # 全世界株 - 先進国債券
    (_N[0], _N[6]): 0.55,    # 全世界株 - 国内リート
    (_N[0], _N[7]): 0.65,    # 全世界株 - 先進国リート
    (_N[0], _N[8]): 0.10,    # 全世界株 - ゴールド
    (_N[0], _N[9]): 0.25,    # 全世界株 - コモディティ
    (_N[0], _N[10]): 0.60,   # 全世界株 - インド株式
    (_N[0], _N[11]): 0.00,   # 全世界株 - トレンドフォロー
    (_N[0], _N[12]): 0.15,   # 全世界株 - 453A
    (_N[0], _N[13]): 0.82,   # 全世界株 - 563A
    (_N[0], _N[14]): 0.45,   # 全世界株 - 米ドル現預金

    (_N[1], _N[2]): 0.96,    # 国内株（日経平均） - 国内株（JPX日経400）
    (_N[1], _N[3]): 0.60,    # 国内株（日経平均） - 米国株
    (_N[1], _N[4]): 0.10,    # 国内株 - 国内債券
    (_N[1], _N[5]): 0.20,    # 国内株 - 先進国債券
    (_N[1], _N[6]): 0.60,    # 国内株 - 国内リート
    (_N[1], _N[7]): 0.40,    # 国内株 - 先進国リート
    (_N[1], _N[8]): 0.05,    # 国内株 - ゴールド
    (_N[1], _N[9]): 0.20,    # 国内株 - コモディティ
    (_N[1], _N[10]): 0.40,   # 国内株 - インド株式
    (_N[1], _N[11]): 0.00,   # 国内株 - トレンドフォロー
    (_N[1], _N[12]): 0.10,   # 国内株 - 453A
    (_N[1], _N[13]): 0.50,   # 国内株 - 563A
    (_N[1], _N[14]): 0.20,   # 国内株 - 米ドル現預金

    (_N[2], _N[3]): 0.58,    # JPX日経400 - 米国株
    (_N[2], _N[4]): 0.10,    # JPX日経400 - 国内債券
    (_N[2], _N[5]): 0.20,    # JPX日経400 - 先進国債券
    (_N[2], _N[6]): 0.58,    # JPX日経400 - 国内リート
    (_N[2], _N[7]): 0.40,    # JPX日経400 - 先進国リート
    (_N[2], _N[8]): 0.05,    # JPX日経400 - ゴールド
    (_N[2], _N[9]): 0.20,    # JPX日経400 - コモディティ
    (_N[2], _N[10]): 0.40,   # JPX日経400 - インド株式
    (_N[2], _N[11]): 0.00,   # JPX日経400 - トレンドフォロー
    (_N[2], _N[12]): 0.10,   # JPX日経400 - 453A
    (_N[2], _N[13]): 0.50,   # JPX日経400 - 563A
    (_N[2], _N[14]): 0.20,   # JPX日経400 - 米ドル現預金

    (_N[3], _N[4]): 0.10,    # 米国株 - 国内債券
    (_N[3], _N[5]): 0.40,    # 米国株 - 先進国債券
    (_N[3], _N[6]): 0.40,    # 米国株 - 国内リート
    (_N[3], _N[7]): 0.65,    # 米国株 - 先進国リート
    (_N[3], _N[8]): 0.10,    # 米国株 - ゴールド
    (_N[3], _N[9]): 0.25,    # 米国株 - コモディティ
    (_N[3], _N[10]): 0.55,   # 米国株 - インド株式
    (_N[3], _N[11]): 0.00,   # 米国株 - トレンドフォロー
    (_N[3], _N[12]): 0.15,   # 米国株 - 453A
    (_N[3], _N[13]): 0.82,   # 米国株 - 563A
    (_N[3], _N[14]): 0.55,   # 米国株 - 米ドル現預金

    (_N[4], _N[5]): 0.20,    # 国内債券 - 先進国債券
    (_N[4], _N[6]): 0.25,    # 国内債券 - 国内リート
    (_N[4], _N[7]): 0.10,    # 国内債券 - 先進国リート
    (_N[4], _N[8]): 0.10,    # 国内債券 - ゴールド
    (_N[4], _N[9]): -0.05,   # 国内債券 - コモディティ
    (_N[4], _N[10]): 0.10,   # 国内債券 - インド株式
    (_N[4], _N[11]): 0.05,   # 国内債券 - トレンドフォロー
    (_N[4], _N[12]): 0.15,   # 国内債券 - 453A
    (_N[4], _N[13]): 0.05,   # 国内債券 - 563A
    (_N[4], _N[14]): 0.00,   # 国内債券 - 米ドル現預金

    (_N[5], _N[6]): 0.15,    # 先進国債券 - 国内リート
    (_N[5], _N[7]): 0.35,    # 先進国債券 - 先進国リート
    (_N[5], _N[8]): 0.20,    # 先進国債券 - ゴールド
    (_N[5], _N[9]): 0.05,    # 先進国債券 - コモディティ
    (_N[5], _N[10]): 0.20,   # 先進国債券 - インド株式
    (_N[5], _N[11]): 0.10,   # 先進国債券 - トレンドフォロー
    (_N[5], _N[12]): 0.65,   # 先進国債券 - 453A
    (_N[5], _N[13]): 0.30,   # 先進国債券 - 563A
    (_N[5], _N[14]): 0.60,   # 先進国債券 - 米ドル現預金

    (_N[6], _N[7]): 0.50,    # 国内リート - 先進国リート
    (_N[6], _N[8]): 0.05,    # 国内リート - ゴールド
    (_N[6], _N[9]): 0.15,    # 国内リート - コモディティ
    (_N[6], _N[10]): 0.30,   # 国内リート - インド株式
    (_N[6], _N[11]): -0.05,  # 国内リート - トレンドフォロー
    (_N[6], _N[12]): 0.15,   # 国内リート - 453A
    (_N[6], _N[13]): 0.35,   # 国内リート - 563A
    (_N[6], _N[14]): 0.15,   # 国内リート - 米ドル現預金

    (_N[7], _N[8]): 0.10,    # 先進国リート - ゴールド
    (_N[7], _N[9]): 0.20,    # 先進国リート - コモディティ
    (_N[7], _N[10]): 0.35,   # 先進国リート - インド株式
    (_N[7], _N[11]): -0.05,  # 先進国リート - トレンドフォロー
    (_N[7], _N[12]): 0.25,   # 先進国リート - 453A
    (_N[7], _N[13]): 0.60,   # 先進国リート - 563A
    (_N[7], _N[14]): 0.45,   # 先進国リート - 米ドル現預金

    (_N[8], _N[9]): 0.50,    # ゴールド - コモディティ
    (_N[8], _N[10]): 0.15,   # ゴールド - インド株式
    (_N[8], _N[11]): 0.20,   # ゴールド - トレンドフォロー
    (_N[8], _N[12]): 0.20,   # ゴールド - 453A
    (_N[8], _N[13]): 0.10,   # ゴールド - 563A
    (_N[8], _N[14]): 0.30,   # ゴールド - 米ドル現預金

    (_N[9], _N[10]): 0.20,   # コモディティ - インド株式
    (_N[9], _N[11]): 0.30,   # コモディティ - トレンドフォロー
    (_N[9], _N[12]): 0.05,   # コモディティ - 453A
    (_N[9], _N[13]): 0.20,   # コモディティ - 563A
    (_N[9], _N[14]): 0.25,   # コモディティ - 米ドル現預金

    (_N[10], _N[11]): 0.00,  # インド株式 - トレンドフォロー
    (_N[10], _N[12]): 0.15,  # インド株式 - 453A
    (_N[10], _N[13]): 0.45,  # インド株式 - 563A
    (_N[10], _N[14]): 0.35,  # インド株式 - 米ドル現預金

    (_N[11], _N[12]): 0.05,  # トレンドフォロー - 453A
    (_N[11], _N[13]): 0.00,  # トレンドフォロー - 563A
    (_N[11], _N[14]): 0.10,  # トレンドフォロー - 米ドル現預金

    (_N[12], _N[13]): 0.20,  # 453A - 563A
    (_N[12], _N[14]): 0.70,  # 453A - 米ドル現預金

    (_N[13], _N[14]): 0.50,  # 563A - 米ドル現預金
    # 円現預金（_N[15]）はどの銘柄とも 0（ボラティリティ0のため設定不要）
}


def build_default_corr(names_list) -> pd.DataFrame:
    """既知の標準銘柄名同士は参考相関値を、それ以外は無相関(0)をセットした相関行列を返す。"""
    df = pd.DataFrame(np.eye(len(names_list)), index=names_list, columns=names_list, dtype=float)
    for (a, b), v in DEFAULT_CORR_PAIRS.items():
        if a in df.index and b in df.columns:
            df.loc[a, b] = v
            df.loc[b, a] = v
    return df


def nearest_psd_correlation(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """相関行列を最近接の正半定値行列へ射影して返す。

    入力値をできるだけ保ったまま、Cholesky分解可能な相関行列に近づける。
    対角要素は1に正規化し、対称性も保証する。
    """
    a = np.array(matrix, dtype=float, copy=True)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 1.0)

    eigval, eigvec = np.linalg.eigh(a)
    eigval = np.clip(eigval, eps, None)
    psd = eigvec @ np.diag(eigval) @ eigvec.T
    psd = (psd + psd.T) / 2.0

    # 対角を1に戻して相関行列として正規化
    diag = np.sqrt(np.clip(np.diag(psd), eps, None))
    psd = psd / np.outer(diag, diag)
    psd = np.clip(psd, -1.0, 1.0)
    np.fill_diagonal(psd, 1.0)
    return psd


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


def metric_row(*items, width: int = 190) -> None:
    """st.columnsの代わりに使う横並び指標表示ヘルパー。

    st.columnsは画面が狭い（スマホ幅）と各列が1個ずつ縦に積み重なってしまい、
    指標を確認するだけで長いスクロールが必要になる。st.container(horizontal=True)は
    入り切らない分だけ自動的に次の行へ折り返す（flex-wrap）ため、スマホでも
    2〜3個ずつ横に並び、PCでは従来通り1行に並ぶ。

    items: (label, value) または (label, value, {st.metricへの追加kwargs}) のタプルを可変長で渡す。
    """
    with st.container(horizontal=True, gap="small"):
        for item in items:
            if len(item) == 3:
                label, value, extra = item
            else:
                label, value = item
                extra = {}
            st.metric(label, value, width=width, **extra)


# ============================================================
# ⑤ ポートフォリオ最適化（平均分散最適化・効率的フロンティア）
# ============================================================
def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def solve_min_variance_portfolio(mu: np.ndarray, cov: np.ndarray, target_return=None, w0=None) -> np.ndarray:
    """空売り・レバレッジなし（各銘柄0〜100%、合計100%）の制約下で、
    target_returnが指定されればそのリターンを満たす最小分散配分を、
    指定がなければ純粋な最小分散配分（グローバル最小分散ポートフォリオ）を返す。"""
    n = len(mu)
    if w0 is None or len(w0) != n:
        w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: float(np.dot(w, mu)) - target_return})

    result = _scipy_minimize(
        lambda w: _portfolio_variance(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-14},
    )
    w = np.clip(result.x, 0.0, 1.0)
    total = w.sum()
    return w / total if total > 1e-9 else w0


def build_efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 25):
    """最小分散配分〜最大期待リターン配分（単一銘柄集中）までの効率的フロンティアを、
    n_points個の(リターン, リスク, 配分)の点列として返す。"""
    w_minvar = solve_min_variance_portfolio(mu, cov, target_return=None)
    r_minvar = float(np.dot(w_minvar, mu))
    r_max = float(np.max(mu))

    if r_max <= r_minvar + 1e-9:
        targets = np.array([r_minvar])
    else:
        targets = np.linspace(r_minvar, r_max, n_points)

    points = []
    w_prev = w_minvar
    for r in targets:
        w = solve_min_variance_portfolio(mu, cov, target_return=float(r), w0=w_prev)
        risk = float(np.sqrt(max(_portfolio_variance(w, cov), 0.0)))
        points.append({"return": float(r), "risk": risk, "weights": w})
        w_prev = w
    return points, r_minvar, r_max


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
        "初期投資額（万円）", min_value=0.0, value=1000.0, step=10.0, format="%.0f", key="initial_investment_input"
    )
    years = st.number_input(
        "投資年数（年）", min_value=1, max_value=100, value=15, step=1, key="years_input"
    )
    target_amount = st.number_input(
        "目標金額（万円）",
        min_value=0.0,
        value=2000.0,
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
    "（スマホでは表を左右にスワイプできます。銘柄名の列は固定表示されます。）"
)

# このwidgetに渡す value（baseline）は「入力モードが変わった時」だけ作り直し、
# それ以外のあらゆる再実行では書き換えない。編集結果はdata_editorの戻り値からのみ読み取り、
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
    width="stretch",
    key=editor_key,
    column_order=view_cols,
    column_config={
        "銘柄名": st.column_config.TextColumn(required=True, pinned=True, width="medium"),
        active_col: st.column_config.NumberColumn(
            format="%.1f" if amt_editable else "%.2f", width=100
        ),
        "期待リターン(%)": st.column_config.NumberColumn(format="%.2f", width="small"),
        "ボラティリティ(%)": st.column_config.NumberColumn(min_value=0.0, format="%.2f", width="small"),
        "コスト(%)": st.column_config.NumberColumn(min_value=0.0, format="%.2f", width="small"),
    },
)
st.caption(
    "💡 各列の境界をドラッグすると幅を調整できます。"
    "スマホで横スクロールする際は、見出し（列名）ではなく数値セル側を指でスワイプしてください"
    "（見出しをつかむと『列の並べ替え』操作として扱われ、銘柄名と数値の列が入れ替わることがあります。"
    "万一入れ替わった場合も内部データは列名で管理しているため壊れませんが、気になる場合は同じ操作で"
    "ドラッグして元の並びに戻すか、ページを再読み込みしてください）。"
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

with st.container(horizontal=True, gap="medium"):
    cash_return = st.number_input(
        f"{CASH_NAME}の期待リターン（%）",
        value=float(st.session_state.get("cash_return_input", CASH_DEFAULT_RETURN)),
        step=0.05,
        format="%.2f",
        key="cash_return_input",
        width=260,
    )
    st.metric(f"{CASH_NAME}（自動計算・待機資金）", fmt_man(cash_amount), width=200)
    st.metric(f"{CASH_NAME}の投資比率", f"{cash_ratio:.1f}%", width=160)

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
st.caption("※ シミュレーション時には、入力された相関行列が正半定値でない場合でも、できるだけ入力値を保つ最近接PSD相関行列へ自動補正して計算します。")
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
    width="stretch",
    key=f"corr_editor_{_scenario_gen}",
    column_order=names,
    column_config={
        nm: st.column_config.NumberColumn(
            min_value=-1.0, max_value=1.0, step=0.05, format="%.2f", width="small"
        )
        for nm in names
    },
)
st.caption(
    "💡 銘柄数が多いと表が横に長くなります。スマホでは表を左右にスワイプしてください"
    "（先頭列の銘柄名は固定表示されるので、スクロールしても行が分かります）。"
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
    width="stretch",
    key=f"cashflow_editor_{_scenario_gen}",
    column_order=CASHFLOW_COLS,
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

    corr_raw = corr_df.to_numpy(dtype=float, copy=True)
    corr = nearest_psd_correlation(corr_raw)
    cov = np.outer(sigma, sigma) * corr

    rng = np.random.default_rng(None if seed == 0 else int(seed))

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # ボラティリティ0の資産（円現預金など）を含むと、
        # 共分散行列は数学的に半正定値でも特異になり得る。
        # その場合はごく小さな数値安定化を加えてCholesky分解する。
        jitter = 1e-12
        L = np.linalg.cholesky(cov + np.eye(len(sigma)) * jitter)

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

    # ⑤ポートフォリオ最適化用に効率的フロンティアを計算（scipyが無い場合はスキップ）
    if SCIPY_AVAILABLE:
        try:
            frontier_pts, r_minvar, r_max_asset = build_efficient_frontier(mu, cov, n_points=25)
        except Exception:
            frontier_pts, r_minvar, r_max_asset = [], None, None
    else:
        frontier_pts, r_minvar, r_max_asset = [], None, None

    # 計算結果一式をsession_stateへ保存する。
    # st.button()の戻り値は押下直後の1回のrerunだけTrueになる仕様のため、下の結果表示部分を
    # 「if run:」の中に置いたままだと、⑤の新しいスライダーなど別のウィジェットを操作しただけで
    # runがFalseに戻り、結果表示ごと消えてしまう。session_stateにキャッシュし、結果表示は
    # 「計算済みの結果があるかどうか」で出し分けることで、この問題を避ける。
    st.session_state["_sim_result"] = dict(
        names=names, mu=mu, cov=cov, sigma=sigma, corr=corr, weights=weights,
        amounts_raw=amounts_raw, actual_initial=actual_initial,
        assets_noncash_valid=assets_noncash_valid.copy(),
        initial_investment=initial_investment, target_amount=target_amount,
        years=years, T=T, rebalance=rebalance, cash_return=float(cash_return),
        paths_full=paths_full, final_values=final_values, max_drawdown=max_drawdown,
        flow_arr=flow_arr, has_cashflow=has_cashflow, principal_base=principal_base,
        expected_return_simple=expected_return_simple, portfolio_risk_pct=portfolio_risk_pct,
        sharpe_ratio=sharpe_ratio, risk_free_rate_pct=risk_free_rate_pct,
        mean_cagr=mean_cagr, mean_final=mean_final, median_final=median_final,
        low_final=low_final, high_final=high_final, mode_final=mode_final,
        prob_loss=prob_loss, prob_target=prob_target,
        mean_mdd=mean_mdd, median_mdd=median_mdd, worst_mdd=worst_mdd,
        frontier_pts=frontier_pts, r_minvar=r_minvar, r_max_asset=r_max_asset,
    )

sim_result = st.session_state.get("_sim_result")
if sim_result:
    globals().update(sim_result)

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------
    st.subheader("④ 結果")

    st.markdown("##### ポートフォリオ全体の前提（年率・入力値ベース）")
    metric_row(
        ("期待リターン（加重平均）", f"{expected_return_simple:.2f}%"),
        ("リスク（標準偏差）", f"{portfolio_risk_pct:.2f}%"),
        (
            "シャープレシオ",
            f"{sharpe_ratio:.2f}" if np.isfinite(sharpe_ratio) else "—",
            {"help": f"無リスク金利として{CASH_NAME}の期待リターン（{risk_free_rate_pct:.2f}%）を使用して算出しています。"},
        ),
    )

    st.markdown("##### シミュレーション結果（確率）")
    if mean_cagr is not None:
        cagr_value = f"{mean_cagr * 100:.2f}%"
    else:
        cagr_value = "—（積立/取崩ありのため非表示）"
    metric_row(
        ("シミュレーション平均CAGR", cagr_value),
        ("元本割れ確率", f"{prob_loss * 100:.1f}%"),
        ("目標額到達確率", f"{prob_target * 100:.1f}%"),
    )
    st.caption(f"※ 元本割れ確率は累計投入元本（初期投資額 + 積立 − 取崩 = {fmt_man(principal_base)}）との比較です。")

    st.markdown("##### 最終資産額の分布（万円）")
    metric_row(
        ("平均値", fmt_man(mean_final)),
        ("中央値", fmt_man(median_final)),
        ("最頻値（推定）", fmt_man(mode_final)),
        ("最低額（99%範囲）", fmt_man(low_final)),
        ("最高額（99%範囲）", fmt_man(high_final)),
    )
    st.caption(
        "※ 最頻値はシミュレーション結果をヒストグラム化し、最も度数の多い区間の中央値を推定値としたものです。"
        "最低額・最高額は全試行のうち極端な外れ値1%ずつを除いた、確率98%が収まる範囲"
        "（下位1%タイル〜上位99%タイル）です。"
    )

    st.markdown("##### 最大ドローダウン")
    metric_row(
        ("平均", f"{mean_mdd * 100:.1f}%"),
        ("中央値", f"{median_mdd * 100:.1f}%"),
        ("悪化5%タイル", f"{worst_mdd * 100:.1f}%"),
    )

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig2, width="stretch")

    # --------------------------------------------------------
    # 最大ドローダウンのヒストグラム
    # --------------------------------------------------------
    st.markdown("#### 最大ドローダウンの分布")
    st.caption("キャッシュフロー（積立/取崩）による資産評価額の増減も含みます。")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=max_drawdown * 100, nbinsx=500, marker_color="indianred"))
    fig3.update_layout(xaxis_title="最大ドローダウン（%）", yaxis_title="頻度", height=350)
    st.plotly_chart(fig3, width="stretch")

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

    # --------------------------------------------------------
    # ⑤ ポートフォリオ最適化（参考）
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("⑤ ポートフォリオ最適化（参考）")

    if not SCIPY_AVAILABLE:
        st.warning(
            "この機能を使うには scipy が必要です。ターミナルで `pip install scipy` を実行し、"
            "アプリを再起動してください。"
        )
    elif r_minvar is None or not frontier_pts:
        st.warning("最適化の計算に失敗しました（銘柄数が少ない、または分散共分散行列が不正な可能性があります）。")
    else:
        st.caption(
            "①の期待リターン・ボラティリティと②の相関から、平均分散最適化（効率的フロンティア）により、"
            "同じ銘柄群のままリスク・リターンのバランスだけを変えた配分を試算します"
            "（空売り・レバレッジなし、各銘柄0〜100%、合計100%が前提の理論値です）。"
            "スライダーを左に動かすほど「期待リターンを維持しつつリスクを抑える」配分に、"
            "右に動かすほど「リスクを取ってリターンを狙う」配分になります。"
            "実際の最頻値への影響を確かめるには、気に入った配分を①へ反映し、"
            "改めて「▶ シミュレーション実行」してください。"
        )

        current_return = expected_return_simple / 100.0
        current_risk = portfolio_risk_pct / 100.0

        slider_min = min(r_minvar, current_return) * 100.0
        slider_max = max(r_max_asset, current_return) * 100.0
        if slider_max - slider_min < 0.05:
            slider_max = slider_min + 0.05

        default_target_pct = float(np.clip(current_return * 100.0, slider_min, slider_max))
        target_return_pct = st.slider(
            "目標の期待リターン（年率・コスト控除後、%）",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=default_target_pct,
            step=0.05,
            key="opt_target_return_pct",
        )
        target_return = target_return_pct / 100.0

        # 直近のフロンティア点をウォームスタートに、目標リターンぴったりの最小分散配分を再計算
        nearest = min(frontier_pts, key=lambda p: abs(p["return"] - target_return))
        w_opt = solve_min_variance_portfolio(mu, cov, target_return=target_return, w0=nearest["weights"])
        risk_opt = float(np.sqrt(max(_portfolio_variance(w_opt, cov), 0.0)))

        metric_row(
            ("提案配分の期待リターン", f"{target_return * 100:.2f}%"),
            (
                "提案配分のリスク（標準偏差）",
                f"{risk_opt * 100:.2f}%",
                {
                    "delta": f"{(risk_opt - current_risk) * 100:+.2f}pt（現状比）",
                    "delta_color": "inverse",
                },
            ),
            ("現状の期待リターン／リスク（参考）", f"{current_return * 100:.2f}% / {current_risk * 100:.2f}%"),
        )

        frontier_risk_pct = [p["risk"] * 100.0 for p in frontier_pts]
        frontier_return_pct = [p["return"] * 100.0 for p in frontier_pts]
        fig_ef = go.Figure()
        fig_ef.add_trace(
            go.Scatter(
                x=frontier_risk_pct, y=frontier_return_pct, mode="lines",
                name="効率的フロンティア", line=dict(color="royalblue", width=2),
            )
        )
        fig_ef.add_trace(
            go.Scatter(
                x=[current_risk * 100.0], y=[current_return * 100.0], mode="markers",
                name="現在の配分", marker=dict(color="gray", size=13, symbol="diamond"),
            )
        )
        fig_ef.add_trace(
            go.Scatter(
                x=[risk_opt * 100.0], y=[target_return * 100.0], mode="markers",
                name="提案配分（スライダー）", marker=dict(color="orange", size=15, symbol="star"),
            )
        )
        fig_ef.update_layout(
            xaxis_title="リスク（標準偏差, %・年率）",
            yaxis_title="期待リターン（%・年率）",
            height=420,
        )
        st.plotly_chart(fig_ef, width="stretch")

        weight_table = pd.DataFrame(
            {
                "銘柄名": names,
                "現在の比率(%)": weights * 100.0,
                "提案の比率(%)": w_opt * 100.0,
            }
        )
        weight_table["差分(pt)"] = weight_table["提案の比率(%)"] - weight_table["現在の比率(%)"]
        st.dataframe(
            weight_table.style.format(
                {"現在の比率(%)": "{:.1f}", "提案の比率(%)": "{:.1f}", "差分(pt)": "{:+.1f}"}
            ),
            width="stretch",
        )

        if st.button("💡 この提案配分を①銘柄設定に反映する", key="apply_optimized_weights"):
            n_noncash = len(assets_noncash_valid)
            updated = assets_noncash_valid.copy().reset_index(drop=True)
            w_noncash = w_opt[:n_noncash]
            updated["投資金額(万円)"] = w_noncash * actual_initial
            updated["投資比率(%)"] = w_noncash * 100.0
            st.session_state.assets_df = updated[ASSET_COLS]
            # ①のdata_editorに確実に反映させるため、baselineを作り直させたうえで
            # ウィジェットのkey自体もローテーションし、古い編集差分を引き継がせない
            # （JSON読込時に③積立取崩が反映されない不具合の修正と同じ仕組み）。
            st.session_state.pop("assets_editor_baseline", None)
            st.session_state.pop("_assets_baseline_key", None)
            st.session_state["_scenario_gen"] = st.session_state.get("_scenario_gen", 0) + 1
            st.success(
                "①銘柄設定に反映しました。内容を確認のうえ、"
                "「▶ シミュレーション実行」で最頻値などへの影響を確認してください。"
            )
            st.rerun()
else:
    st.info("① 銘柄設定・② 相関・③ 積立取崩を確認し、「▶ シミュレーション実行」ボタンを押してください。")
