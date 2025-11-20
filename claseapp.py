import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =================== CONFIG DE PÁGINA ===================
st.set_page_config(
    page_title="Análisis de empresas — MQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== ESTILOS ===================
st.markdown("""
<style>
:root{
  --bg:#f9f4ef; --bg2:#fffdf9; --panel:#f3e9dd;
  --ink:#2b2b2b; --muted:#6b5b4d; --line:#d8c9b6;
  --brand:#d78e6c; --brand2:#b5835a;
}
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at top left, #fff5ea 0, #f9f4ef 45%, #ffffff 100%) !important;
  color: var(--ink) !important;
}
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--panel) 0%, #f7efe4 100%) !important;
  border-right:1px solid var(--line);
}
.card{
  background: var(--panel);
  border:1px solid var(--line);
  border-radius:18px;
  padding:1rem 1.2rem;
  box-shadow:0 8px 22px rgba(183,131,90,.16);
}
.kpi{ font-weight:700; font-size:1.05rem; color:var(--brand2);}
.small{ color:var(--muted); font-size:.92rem;}
hr{ border:none; border-top:1px solid var(--line); margin:.75rem 0;}
h1, h2, h3{ color:var(--brand2); letter-spacing:.3px; }
.badge{
  display:inline-flex; align-items:center; gap:.4rem; font-size:.85rem;
  background:#fff3e8; color:var(--brand2);
  border:1px solid #eed8c3; padding:.25rem .6rem; border-radius:12px;
  font-weight:500;
}
.metric-card{
  background: #fffdf9;
  border-radius:16px;
  border:1px solid #e2d5c4;
  padding:.8rem 1rem;
  box-shadow:0 6px 18px rgba(0,0,0,.04);
}
.metric-label{
  font-size:.80rem;
  color:var(--muted);
}
.metric-value{
  font-size:1.25rem;
  font-weight:700;
  color:var(--brand2);
}
.stDataFrame, .stTable{
  background:var(--panel)!important;
  border-radius:14px;
  border:1px solid var(--line);
  overflow:hidden;
}
</style>
""", unsafe_allow_html=True)

# =================== HEADER ===================
st.markdown("### MQ Finance · App educativa de análisis bursátil")
st.title("📊 Análisis de empresas — Mariana Quezada (MQ)")
st.markdown("<span class='small'>Explora empresas, compara contra un índice y entiende su perfil de riesgo 🪶</span>", unsafe_allow_html=True)

# =================== SIDEBAR ===================
with st.sidebar:
    st.header("⚙️ Controles")
    stonk = st.text_input("Ticker (ej. AAPL, MSFT, TSLA):", "AAPL")
    benchmark_str = st.text_input("Índice ref. (ej. SPY, ^GSPC, QQQ):", "SPY")
    years_candles = st.slider("Años en velas", 1, 5, 1)
    show_beta = st.checkbox("Calcular Beta/Correlación (opcional)", value=True)
    show_var  = st.checkbox("Incluir VaR histórico 95% (opcional)", value=True)

    st.markdown("---")
    risk_free_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0, max_value=15.0, value=4.0, step=0.25,
        help="Aproximación para calcular Sharpe ratio (se convierte a tasa diaria)."
    )

    st.markdown("---")
    st.caption("App educativa · No es recomendación de inversión.")

# =================== UTILIDADES ===================
@st.cache_data(show_spinner=False, ttl=300)
def get_info_safe(tkr: str) -> dict:
    try:
        tk = yf.Ticker(tkr)
        info = tk.get_info()
        return info or {}
    except Exception:
        return {}

@st.cache_data(show_spinner=True, ttl=300)
def get_history(tkr: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(tkr, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != ""]).strip() for tup in df.columns]

    def norm(col: str) -> str:
        c = str(col).lower().replace("_", " ").strip()
        if "adj" in c and "close" in c: return "Adj Close"
        if c == "open":  return "Open"
        if c == "high":  return "High"
        if c == "low":   return "Low"
        if c == "close": return "Close"
        if "volume" in c: return "Volume"
        return str(col)

    df = df.rename(columns={c: norm(c) for c in df.columns})

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if cols:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    ohlc_present = [c for c in ["Open","High","Low","Close"] if c in df.columns]
    if len(ohlc_present) == 4:
        df = df.dropna(how="any", subset=ohlc_present)
    else:
        df = df.dropna(how="any")
    return df

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    have_all = all(c in df.columns for c in ["Open","High","Low","Close"])
    if have_all:
        return df
    p = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
    if p is None or p.empty:
        return df
    p = pd.to_numeric(p, errors="coerce").dropna()
    if p.empty:
        return df
    o = p.shift(1).fillna(p)
    c = p
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    df2 = df.copy()
    df2.loc[p.index, "Open"] = o
    df2.loc[p.index, "High"] = h
    df2.loc[p.index, "Low"]  = l
    df2.loc[p.index, "Close"]= c
    return df2.dropna(subset=["Open","High","Low","Close"])

def last_n_years(df: pd.DataFrame, years: int = 1):
    if df.empty: return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start]

def slice_from_offset(df: pd.DataFrame, *, years=0, months=0, ytd=False):
    if df.empty: return df
    end = df.index.max()
    start = pd.Timestamp(year=end.year, month=1, day=1) if ytd else end - pd.DateOffset(years=years, months=months)
    return df.loc[df.index >= start]

def period_specs():
    return {
        "3M": dict(months=3), "6M": dict(months=6), "9M": dict(months=9),
        "YTD": dict(ytd=True), "1Y": dict(years=1), "3Y": dict(years=3), "5Y": dict(years=5),
    }

def arithmetic_return(series): 
    return np.nan if series is None or series.size < 2 else series.iloc[-1]/series.iloc[0]-1

def daily_returns(price):
    return price.pct_change().dropna()

def annualized_vol(returns):
    if returns.dropna().empty:
        return np.nan
    return returns.std(ddof=1)*np.sqrt(252)

def beta_and_corr(asset_rets, bench_rets):
    joined = pd.concat([asset_rets, bench_rets], axis=1, join="inner").dropna()
    if joined.shape[0] < 20:
        return np.nan, np.nan
    x, y = joined.iloc[:,1], joined.iloc[:,0]
    var_m = np.var(x, ddof=1)
    cov = np.cov(x, y, ddof=1)[0,1]
    beta = cov/var_m if var_m != 0 else np.nan
    corr = np.corrcoef(x,y)[0,1]
    return beta, corr

def hist_var(returns, level=0.95):
    if returns.dropna().empty:
        return np.nan
    return float(np.quantile(returns.dropna(), 1-level))

def to_price_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:,0]
    return pd.to_numeric(s, errors="coerce").dropna()

def base_zero(series):
    if series is None or series.empty:
        return pd.Series(dtype=float)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:,0]
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    base = float(series.iloc[0]) if not np.isnan(series.iloc[0]) else 0
    if base == 0:
        return pd.Series(index=series.index, dtype=float)
    out = series/base - 1.0
    out.name = getattr(series, "name", None)
    return out

def max_drawdown(price_series):
    if	price_series is None or price_series.empty:
        return np.nan
    price_series = price_series.dropna()
    if price_series.empty:
        return np.nan
    cum_max = price_series.cummax()
    drawdown = price_series / cum_max - 1
    return drawdown.min()

def sharpe_ratio(returns, rf_daily=0.0):
    if returns is None or returns.dropna().empty:
        return np.nan
    r_excess = returns - rf_daily
    if r_excess.dropna().empty:
        return np.nan
    sigma = r_excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return (r_excess.mean() * 252) / (sigma * np.sqrt(252))

# =================== 1) INFO DE LA EMPRESA + GEMINI ===================
try:
    info = get_info_safe(stonk)
    nombre = info.get("longName", info.get("shortName", stonk.upper()))
    industria = info.get("industry", "No disponible")
    descripcion = info.get("longBusinessSummary", "Descripción no encontrada")
    logo_url = info.get("logo_url", None)

    head_col1, head_col2 = st.columns([1.2, 1])

    with head_col1:
        st.markdown("#### Empresa seleccionada")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if logo_url:
            st.image(logo_url, width=90)
        st.markdown(
            f"""
            <div class='kpi'>{nombre}</div>
            <div class='small'>
                Ticker: <b>{stonk.upper()}</b> · Industria: <b>{industria}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with head_col2:
        st.markdown("#### Snapshot de mercado")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cols_kpi = st.columns(2)
        metrics = [
            ("Precio actual", info.get("currentPrice"), "${:,.2f}"),
            ("Market Cap", info.get("marketCap"), "${:,.0f}"),
            ("P/E (TTM)", info.get("trailingPE"), "{:,.2f}"),
            ("Dividend Yield", info.get("dividendYield") * 100 if info.get("dividendYield") else None, "{:.2f}%"),
        ]
        for idx, (label, val, fmt) in enumerate(metrics):
            with cols_kpi[idx % 2]:
                if val is not None:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{fmt.format(val)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧾 Descripción de la empresa (Yahoo Finance)"):
        st.write(descripcion)

    # =========================================================
    #     🔥 AQUÍ INCLUIMOS DIRECTAMENTE TU API KEY
    # =========================================================
st.markdown(
    "<span class='badge'>📌 Nota: la descripción anterior viene directamente de Yahoo Finance (sin traducción automática).</span>",
    unsafe_allow_html=True
)

except Exception as e:
    st.error("No se pudo obtener información de la empresa.")
    st.exception(e)

st.markdown("---")

# =================== 2) DESCARGA DE DATOS BASE ===================
asset_df = get_history(stonk, "max", "1d")
bench_df = get_history(benchmark_str, "max", "1d")

if asset_df.empty:
    st.error("No se pudieron descargar datos históricos del ticker. Verifica el símbolo.")
    st.stop()

rf_daily = risk_free_pct / 100 / 252.0
one_year_asset = slice_from_offset(asset_df, years=1)
px_1y = to_price_series(one_year_asset)

# =================== 3) TABS PRINCIPALES ===================
tab_overview, tab_charts, tab_risk, tab_data = st.tabs([
    "🏠 Overview",
    "📈 Gráficos",
    "📊 Riesgo & Rendimientos",
    "📂 Datos & Descarga"
])

# ---------- TAB OVERVIEW ----------
with tab_overview:
    st.subheader("📌 Resumen rápido del desempeño (último año)")

    col1, col2, col3, col4 = st.columns(4)
    col5, col6 = st.columns(2)

    if px_1y.size >= 2:
        ret_1y = arithmetic_return(px_1y)
        rets_1y_daily = daily_returns(px_1y)
        vol_1y = annualized_vol(rets_1y_daily)
        sharpe_1y = sharpe_ratio(rets_1y_daily, rf_daily=rf_daily)
        dd_1y = max_drawdown(px_1y)

        if show_beta and not bench_df.empty:
            bench_1y = slice_from_offset(bench_df, years=1)
            bench_px_1y = to_price_series(bench_1y)
            if bench_px_1y.size >= 2:
                bench_rets_1y = daily_returns(bench_px_1y)
                beta_1y, corr_1y = beta_and_corr(rets_1y_daily, bench_rets_1y)
            else:
                beta_1y, corr_1y = (np.nan, np.nan)
        else:
            beta_1y, corr_1y = (np.nan, np.nan)

        col1.metric("Retorno 1Y", f"{ret_1y:,.2%}")
        col2.metric("Volatilidad ann. 1Y", f"{vol_1y:,.2%}" if not np.isnan(vol_1y) else "N/D")
        col3.metric("Sharpe 1Y", f"{sharpe_1y:,.2f}" if not np.isnan(sharpe_1y) else "N/D")
        col4.metric("Máx. drawdown 1Y", f"{dd_1y:,.2%}" if not np.isnan(dd_1y) else "N/D")
        col5.metric("Beta 1Y", f"{beta_1y:,.2f}" if not np.isnan(beta_1y) else "N/D")
        col6.metric("Correlación 1Y", f"{corr_1y:,.2f}" if not np.isnan(corr_1y) else "N/D")

        st.markdown("#### ⚖️ Comparación base-cero contra el índice (último año)")
        one_year_bench = slice_from_offset(bench_df, years=1) if not bench_df.empty else pd.DataFrame()
        asset_ref = to_price_series(one_year_asset)
        bench_ref = to_price_series(one_year_bench) if not one_year_bench.empty else pd.Series(dtype=float)

        if bench_ref.empty:
            joined = base_zero(asset_ref).to_frame(name=stonk.upper())
        else:
            asset_bz = base_zero(asset_ref); asset_bz.name = stonk.upper()
            bench_bz = base_zero(bench_ref); bench_bz.name = benchmark_str.upper()
            joined = pd.concat([asset_bz, bench_bz], axis=1, join="inner").dropna()

        fig_base = px.line(joined, labels={"value":"Rendimiento acumulado","index":"Fecha","variable":"Serie"})
        fig_base.update_traces(mode="lines", hovertemplate="%{y:.2%}")
        fig_base.update_layout(
            yaxis_tickformat=".0%", template="plotly_white", height=420,
            legend_title_text="", margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_base, use_container_width=True)

# ---------- TAB GRÁFICOS ----------
with tab_charts:
    st.subheader("📉 Gráfico de velas (candlesticks)")
    candles_raw = last_n_years(asset_df, years=years_candles)
    candles = ensure_ohlc(candles_raw)

    if candles.empty or not set(["Open","High","Low","Close"]).issubset(candles.columns):
        st.warning("No fue posible construir OHLC para este rango/ticker.")
    else:
        fig_candles = go.Figure(
            data=[go.Candlestick(
                x=candles.index,
                open=candles["Open"],
                high=candles["High"],
                low=candles["Low"],
                close=candles["Close"],
                name=stonk.upper()
            )]
        )
        fig_candles.update_layout(
            height=480, template="plotly_white",
            xaxis_title="Fecha", yaxis_title="Precio",
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_candles, use_container_width=True)

# ---------- TAB RIESGO ----------
with tab_risk:
    st.subheader("📐 Rendimientos y riesgos por periodo")

    periods = period_specs()
    records = []

    for label, spec in periods.items():
        window = slice_from_offset(asset_df, **spec)
        px_series = to_price_series(window)

        if px_series.size < 2:
            ret = vol = beta = corr = var95 = np.nan
        else:
            ret = arithmetic_return(px_series)
            rets_d = daily_returns(px_series)
            vol = annualized_vol(rets_d)

            if show_beta and not bench_df.empty:
                bench_win = slice_from_offset(bench_df, **spec)
                bench_series = to_price_series(bench_win)
                if bench_series.size > 1:
                    beta, corr = beta_and_corr(daily_returns(px_series), daily_returns(bench_series))
                else:
                    beta, corr = (np.nan, np.nan)
            else:
                beta, corr = (np.nan, np.nan)

            var95 = hist_var(rets_d) if show_var else np.nan

        records.append({
            "Periodo": label,
            "Retorno Acum.": ret,
            "Volatilidad (ann)": vol,
            "Beta (opc)": beta,
            "Correlación (opc)": corr,
            "VaR 95% diario (opc)": var95
        })

    table = pd.DataFrame(records).set_index("Periodo")
    fmt = {
        "Retorno Acum.": "{:.2%}",
        "Volatilidad (ann)": "{:.2%}",
        "Beta (opc)": "{:.2f}",
        "Correlación (opc)": "{:.2f}",
        "VaR 95% diario (opc)": "{:.2%}"
    }

    st.dataframe(table.style.format(fmt), use_container_width=True)

    st.markdown("#### Distribución de rendimientos diarios (1Y)")
    if px_1y.size >= 2:
        rets_1y_daily = daily_returns(px_1y)
        fig_hist = px.histogram(rets_1y_daily, nbins=40)
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------- TAB DATOS ----------
with tab_data:
    st.subheader("📂 Datos históricos de la acción")
    st.dataframe(asset_df.tail(200), use_container_width=True)

    csv_data = asset_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV completo",
        data=csv_data,
        file_name=f"{stonk.upper()}_historico.csv",
        mime="text/csv"
    )

# =================== FOOTER ===================
st.markdown(
    "<div class='small' style='margin-top:15px; text-align:center; color:#b5835a;'>"
    "© 2025 Mariana Quezada — MQ Finance · App educativa de análisis financiero 🌾"
    "</div>",
    unsafe_allow_html=True
)
