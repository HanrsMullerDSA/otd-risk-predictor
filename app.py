# app.py — OTD Risk Predictor
# Supply Chain Analytics Lab · Projeto 1
# ──────────────────────────────────────────────────────────────────────────────

import json
import requests
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="OTD Risk Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ────────────────────────────────────────────────────────────────
THRESHOLD = 0.372
AUC_ROC   = 0.674
RECALL    = 0.383

FEATURE_ORDER = [
    "freight_value", "product_weight_g",
    "customer_state", "seller_state", "seller_id", "route",
    "weekday", "month",
    "hist_atraso_route_t", "hist_atraso_seller_t", "hist_atraso_route_smooth",
    "prazo_prometido_dias", "distancia_km", "volume_seller_7d_t",
]

STATE_COORDS = {
    "AC": (-9.97, -67.81), "AL": (-9.71, -35.74), "AM": (-3.10, -60.02),
    "AP": ( 0.03, -51.07), "BA": (-12.97,-38.48),  "CE": (-3.72, -38.54),
    "DF": (-15.78,-47.93), "ES": (-20.32,-40.34),  "GO": (-16.68,-49.25),
    "MA": (-2.53, -44.30), "MG": (-19.92,-43.94),  "MS": (-20.44,-54.65),
    "MT": (-15.60,-56.10), "PA": (-1.46, -48.50),  "PB": (-7.12, -34.86),
    "PE": (-8.05, -34.88), "PI": (-5.09, -42.80),  "PR": (-25.43,-49.27),
    "RJ": (-22.91,-43.17), "RN": (-5.79, -35.21),  "RO": (-8.76, -63.90),
    "RR": ( 2.82, -60.67), "RS": (-30.03,-51.23),  "SC": (-27.59,-48.55),
    "SE": (-10.91,-37.07), "SP": (-23.55,-46.64),  "TO": (-10.25,-48.32),
}
ESTADOS     = sorted(STATE_COORDS.keys())
DIAS_SEMANA = ["Segunda","Terça","Quarta","Quinta","Sexta","Sábado","Domingo"]

GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood"
    "/master/public/data/brazil-states.geojson"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.80rem; color: #666; }
.block-container              { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE CACHE
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Carregando modelo…")
def load_artifacts():
    model         = joblib.load("artifacts/modelo_balanceado.joblib")
    lookup_route  = pd.read_parquet("artifacts/lookup_route.parquet")
    lookup_seller = pd.read_parquet("artifacts/lookup_seller.parquet")
    with open("artifacts/encoders.json", "r", encoding="utf-8") as f:
        encoders = json.load(f)
    return model, lookup_route, lookup_seller, encoders


@st.cache_data(show_spinner=False)
def load_brazil_geojson() -> dict | None:
    """Baixa GeoJSON dos estados brasileiros (com fallback para None)."""
    try:
        r = requests.get(GEOJSON_URL, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def prepare_map_data(_lookup_route: pd.DataFrame):
    """
    Processa lookup_route → agrega por estado de destino e lista top rotas.
    Prefixo _ no parâmetro para o Streamlit não tentar hashear o DataFrame.
    """
    df = _lookup_route.copy()

    # Se houver série temporal, pegar a linha mais recente por rota
    sort_col = next(
        (c for c in ["known_at_reference_route", "label_known_at"] if c in df.columns),
        None,
    )
    if sort_col:
        df = df.sort_values(sort_col).groupby("route", as_index=False).last()

    # Extrair estados da rota (formato fixo: "XX_XX")
    df["seller_state"]   = df["route"].str[:2]
    df["customer_state"] = df["route"].str[3:]

    # Agregação por estado de destino
    agg_cols = {"hist_atraso_route_t": "mean", "route": "count"}
    if "hist_atraso_route_smooth" in df.columns:
        agg_cols["hist_atraso_route_smooth"] = "mean"

    state_risk = (
        df.groupby("customer_state")
        .agg(**{
            "taxa_bruta":  ("hist_atraso_route_t", "mean"),
            "n_rotas":     ("route", "count"),
            **({"taxa_smooth": ("hist_atraso_route_smooth", "mean")}
               if "hist_atraso_route_smooth" in df.columns else {}),
        })
        .reset_index()
        .rename(columns={"customer_state": "estado"})
        .sort_values("taxa_bruta", ascending=False)
        .reset_index(drop=True)
    )
    # Converter para percentual somente na coluna de display
    state_risk["taxa_pct"] = (state_risk["taxa_bruta"] * 100).round(2)

    # Top rotas
    sort_by = "hist_atraso_route_smooth" if "hist_atraso_route_smooth" in df.columns else "hist_atraso_route_t"
    top_routes = (
        df[["route", "hist_atraso_route_t"]
           + (["hist_atraso_route_smooth"] if "hist_atraso_route_smooth" in df.columns else [])]
        .sort_values(sort_by, ascending=False)
        .head(15)
        .copy()
        .reset_index(drop=True)
    )
    top_routes["hist_atraso_route_t"] = (top_routes["hist_atraso_route_t"] * 100).round(2)
    if "hist_atraso_route_smooth" in top_routes.columns:
        top_routes["hist_atraso_route_smooth"] = (top_routes["hist_atraso_route_smooth"] * 100).round(2)

    return state_risk, top_routes


# ═════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return float(R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def encode(value: str, mapping: dict) -> int:
    return mapping.get(str(value), -1)


def fetch_route_hist(df: pd.DataFrame, route: str):
    rows = df[df["route"] == route]
    if rows.empty:
        return None, None
    sort_col = next(
        (c for c in ["known_at_reference_route", "label_known_at"] if c in rows.columns),
        None,
    )
    last   = rows.sort_values(sort_col).iloc[-1] if sort_col else rows.iloc[-1]
    t      = float(last["hist_atraso_route_t"])       if "hist_atraso_route_t"      in last.index else None
    smooth = float(last["hist_atraso_route_smooth"])  if "hist_atraso_route_smooth" in last.index else t
    return t, smooth


def fetch_seller_globals(df: pd.DataFrame):
    h = float(df["hist_atraso_seller_t"].mean()) if "hist_atraso_seller_t" in df.columns else 0.067
    v = float(df["volume_seller_7d_t"].mean())   if "volume_seller_7d_t"   in df.columns else 8.0
    return h, v


def risk_meta(p: float):
    if p < 0.05:      return "Baixo",    "#27ae60", "✅"
    if p < THRESHOLD: return "Moderado", "#f39c12", "⚠️"
    if p < 0.50:      return "Alto",     "#e67e22", "🔴"
    return                   "Crítico",  "#c0392b", "🚨"


def bar_color(pct: float) -> str:
    if pct >= 15: return "#c0392b"
    if pct >= 10: return "#e67e22"
    if pct >=  7: return "#f39c12"
    return               "#27ae60"


# ── Gauges e gráficos ─────────────────────────────────────────────────────────
def build_gauge(proba: float) -> go.Figure:
    pct   = proba * 100
    label, color, _ = risk_meta(proba)
    thr   = THRESHOLD * 100
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct,
        number = {"suffix": "%", "font": {"size": 56, "color": color}},
        title  = {"text": f"Probabilidade de Atraso &nbsp;·&nbsp; Nível: <b>{label}</b>",
                  "font": {"size": 15}},
        gauge  = {
            "axis":  {"range": [0, 100], "ticksuffix": "%", "nticks": 6},
            "bar":   {"color": color, "thickness": 0.26},
            "steps": [
                {"range": [0, 5],    "color": "#d5f5e3"},
                {"range": [5, thr],  "color": "#fdebd0"},
                {"range": [thr, 50], "color": "#fad7a0"},
                {"range": [50, 100], "color": "#f5b7b1"},
            ],
            "threshold": {
                "line":      {"color": "#2c3e50", "width": 3},
                "thickness": 0.85,
                "value":     thr,
            },
        },
    ))
    fig.update_layout(
        height=310, margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_choropleth(state_risk: pd.DataFrame, geojson: dict | None) -> go.Figure:
    """Choropleth colorido por taxa de atraso. Fallback: bubble map."""
    if geojson is not None:
        fig = px.choropleth(
            state_risk,
            geojson        = geojson,
            locations      = "estado",
            featureidkey   = "properties.abbreviation",
            color          = "taxa_pct",
            color_continuous_scale = "RdYlGn_r",
            range_color    = [0, min(state_risk["taxa_pct"].max(), 25)],
            hover_name     = "estado",
            hover_data     = {"taxa_pct": ":.1f", "n_rotas": True, "estado": False},
            labels         = {"taxa_pct": "Taxa de Atraso (%)", "n_rotas": "Nº de Rotas"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
    else:
        # Fallback: bolhas nos centroides dos estados
        sr = state_risk.copy()
        sr["lat"] = sr["estado"].map(lambda s: STATE_COORDS.get(s, (0, 0))[0])
        sr["lon"] = sr["estado"].map(lambda s: STATE_COORDS.get(s, (0, 0))[1])
        fig = go.Figure(go.Scattergeo(
            lat        = sr["lat"],
            lon        = sr["lon"],
            text       = sr["estado"],
            mode       = "markers+text",
            textposition = "top center",
            marker     = dict(
                size       = sr["taxa_pct"] * 3.5,
                color      = sr["taxa_pct"],
                colorscale = "RdYlGn_r",
                showscale  = True,
                colorbar   = dict(title="Taxa (%)"),
                sizemode   = "area",
            ),
            customdata = sr[["taxa_pct", "n_rotas"]].values,
            hovertemplate = "<b>%{text}</b><br>Taxa: %{customdata[0]:.1f}%<br>Rotas: %{customdata[1]}<extra></extra>",
        ))
        fig.update_geos(
            scope         = "south america",
            showcountries = True,
            countrycolor  = "#ccc",
            showland      = True,
            landcolor     = "#f8f8f8",
            showcoastlines= True,
            coastlinecolor= "#aaa",
            center        = dict(lat=-14, lon=-52),
            projection_scale = 3.2,
        )
    fig.update_layout(
        height          = 520,
        margin          = dict(t=10, b=10, l=0, r=0),
        paper_bgcolor   = "rgba(0,0,0,0)",
        coloraxis_colorbar = dict(title="Taxa de<br>Atraso (%)", ticksuffix="%"),
    )
    return fig


def build_bar_chart(state_risk: pd.DataFrame) -> go.Figure:
    top10  = state_risk.head(10).sort_values("taxa_pct")
    colors = [bar_color(v) for v in top10["taxa_pct"]]
    fig = go.Figure(go.Bar(
        x            = top10["taxa_pct"],
        y            = top10["estado"],
        orientation  = "h",
        marker_color = colors,
        text         = [f"{v:.1f}%" for v in top10["taxa_pct"]],
        textposition = "outside",
        hovertemplate= "<b>%{y}</b><br>Taxa de Atraso: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title   = "Taxa Histórica de Atraso (%)",
        yaxis_title   = None,
        height        = 380,
        margin        = dict(t=10, b=30, l=10, r=60),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        xaxis         = dict(showgrid=True, gridcolor="#eee"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📦 OTD Risk Predictor")
    st.caption("Supply Chain Analytics Lab · Projeto 1")
    st.divider()

    st.markdown("### 📍 Origem e Destino")
    seller_state   = st.selectbox("Estado do vendedor (origem)", ESTADOS,
                                  index=ESTADOS.index("SP"))
    customer_state = st.selectbox("Estado do cliente (destino)", ESTADOS,
                                  index=ESTADOS.index("RJ"))

    st.markdown("### 📦 Produto")
    product_weight = st.number_input("Peso do produto (g)",
                                     min_value=1, max_value=50_000,
                                     value=1_000, step=100)
    freight_value  = st.number_input("Valor do frete (R$)",
                                     min_value=0.0, max_value=500.0,
                                     value=20.00, step=0.50, format="%.2f")

    st.markdown("### 📅 Prazo")
    purchase_date = st.date_input("Data da compra", value=date.today())
    delivery_days = st.slider("Prazo prometido (dias corridos)",
                               min_value=2, max_value=60, value=14)

    st.divider()
    run = st.button("🔍  Calcular risco de atraso",
                    use_container_width=True, type="primary")

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("📦 Predição de Atraso Logístico — OTD Risk Predictor")
st.caption(
    "Modelo XGBoost · 96 mil pedidos Olist 2016–2018 · "
    "Engenharia temporal causal (anti-leakage) · Split temporal 80/20"
)
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# ABAS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🔍 Previsão de Risco", "🗺️ Mapa de Risco por Estado"])


# ─────────────────────────────────────────────────────────────────────────────
# ABA 1 — PREVISÃO
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if run:
        with st.spinner("Calculando probabilidade de atraso…"):
            try:
                model, lookup_route, lookup_seller, encoders = load_artifacts()
            except FileNotFoundError as exc:
                st.error(
                    f"**Artefato não encontrado:** `{exc}`\n\n"
                    "Certifique-se de que a pasta `artifacts/` contém:\n"
                    "- `modelo_balanceado.joblib`\n"
                    "- `lookup_route.parquet`\n"
                    "- `lookup_seller.parquet`\n"
                    "- `encoders.pkl`"
                )
                st.stop()

            # Features derivadas
            route     = f"{seller_state}_{customer_state}"
            weekday   = purchase_date.weekday()
            month     = purchase_date.month

            lat1, lon1 = STATE_COORDS[seller_state]
            lat2, lon2 = STATE_COORDS[customer_state]
            distancia  = haversine_km(lat1, lon1, lat2, lon2)

            hist_rt, hist_rs = fetch_route_hist(lookup_route, route)
            cold_start = (hist_rt is None)
            if cold_start:
                prior   = (lookup_route["hist_atraso_route_t"].mean()
                           if "hist_atraso_route_t" in lookup_route.columns else 0.067)
                hist_rt = prior
                hist_rs = prior

            hist_seller, vol_seller = fetch_seller_globals(lookup_seller)

            X = pd.DataFrame([{
                "freight_value":             freight_value,
                "product_weight_g":          float(product_weight),
                "customer_state":            encode(customer_state, encoders["customer_state"]),
                "seller_state":              encode(seller_state,   encoders["seller_state"]),
                "seller_id":                 -1,
                "route":                     encode(route,          encoders["route"]),
                "weekday":                   weekday,
                "month":                     month,
                "hist_atraso_route_t":       hist_rt,
                "hist_atraso_seller_t":      hist_seller,
                "hist_atraso_route_smooth":  hist_rs,
                "prazo_prometido_dias":      delivery_days,
                "distancia_km":              distancia,
                "volume_seller_7d_t":        vol_seller,
            }])[FEATURE_ORDER]

            idx_1  = list(model.classes_).index(1)
            proba  = float(model.predict_proba(X)[0, idx_1])
            nivel, cor, icone = risk_meta(proba)

        # Resultados
        col_g, col_d = st.columns([1, 1], gap="large")

        with col_g:
            st.plotly_chart(build_gauge(proba), use_container_width=True)
            if proba >= THRESHOLD:
                st.error(
                    f"**{icone} INTERVENÇÃO RECOMENDADA**  \n"
                    f"Prob. de atraso: **{proba*100:.1f}%** "
                    f"— acima do threshold operacional ({THRESHOLD*100:.1f}%)"
                )
                st.markdown("**💡 Ações sugeridas:**")
                st.markdown("""
- 📞 Acionar equipe de follow-up logístico
- 🚛 Avaliar transportadora ou rota alternativa
- 📩 Comunicar cliente preventivamente
- 📋 Monitorar expedição junto ao vendedor
                """)
            else:
                st.success(
                    f"**{icone} Dentro do padrão operacional**  \n"
                    f"Prob. de atraso: **{proba*100:.1f}%** "
                    f"— abaixo do threshold ({THRESHOLD*100:.1f}%)"
                )

        with col_d:
            st.subheader("📊 Parâmetros da Previsão")
            c1, c2 = st.columns(2)
            c1.metric("Rota",                  route)
            c2.metric("Distância estimada",    f"{distancia:,.0f} km")
            c1.metric("Prazo prometido",       f"{delivery_days} dias")
            c2.metric("Dia da compra",         DIAS_SEMANA[weekday])
            c1.metric("Mês",                   purchase_date.strftime("%B/%Y").capitalize())
            c2.metric("Frete",                 f"R$ {freight_value:.2f}")

            st.markdown("**Sinais históricos da rota:**")
            c3, c4 = st.columns(2)
            c3.metric("Taxa bruta (rota)",     f"{hist_rt*100:.1f}%",
                      help="hist_atraso_route_t — acumulado sem suavização")
            c4.metric("Taxa suavizada (rota)", f"{hist_rs*100:.1f}%",
                      help="hist_atraso_route_smooth — Bayesian smoothing α=30")

            if cold_start:
                st.info("ℹ️ **Rota sem histórico.** Usando prior global como referência.")
            st.caption(
                "_Seller desconhecido → cold start (prior global)._  \n"
                "_Distância via Haversine usando centroides dos estados._"
            )

        with st.expander("🔬 Dados brutos enviados ao modelo"):
            st.dataframe(X.T.rename(columns={0: "Valor"}), use_container_width=True)

    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Algoritmo",        "XGBoost")
        k2.metric("AUC-ROC",          f"{AUC_ROC:.3f}")
        k3.metric("Recall (atrasos)", f"{RECALL*100:.1f}%")
        k4.metric("Threshold ótimo",  f"{THRESHOLD*100:.1f}%")

        st.info("👈 **Configure os parâmetros na barra lateral** e clique em *Calcular risco de atraso*.")

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("""
### 🎯 O que este modelo faz?
Estima a **probabilidade de atraso no momento da compra** — antes de qualquer
evento logístico ocorrer — para permitir intervenção preventiva na operação.

---
### 🔑 Principais drivers de atraso
| Feature | Importância (gain) |
|---|---|
| Sazonalidade — mês | 18.4% |
| Histórico da rota (suavizado) | 11.9% |
| Estado de destino | 10.1% |
| Prazo prometido | 7.6% |
| Histórico bruto da rota | 6.7% |
            """)
        with col_r:
            st.markdown(f"""
### ⚙️ Como funciona?
1. **Entradas:** origem, destino, peso, frete, data, prazo
2. **Features derivadas:** rota, distância Haversine, dia da semana, mês
3. **Histórico causal:** taxa de atraso da rota (anti-leakage, apenas dados até t-1)
4. **Modelo:** XGBoost com `scale_pos_weight ≈ 12`
5. **Decisão:** threshold calibrado em **{THRESHOLD*100:.1f}%** (max F1-score)

---
### 📐 Garantia anti-leakage
| ✅ Usado no modelo | ❌ Excluído (pós-evento) |
|---|---|
| Histórico até t-1 | `transport_lead_time` |
| Prazo prometido | `time_variance` |
| CEP, estado, rota | Data real de entrega |
            """)


# ─────────────────────────────────────────────────────────────────────────────
# ABA 2 — MAPA DE RISCO
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🗺️ Taxa Histórica de Atraso por Estado de Destino")
    st.caption(
        "Calculado a partir de `lookup_route.parquet` — taxa média de atraso histórica "
        "agregada por estado de destino, usando engenharia temporal causal (anti-leakage)."
    )

    # Carrega artefatos (já em cache se Tab 1 foi usado)
    try:
        _, lookup_route, _, _ = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Artefato não encontrado: `{exc}`")
        st.stop()

    state_risk, top_routes = prepare_map_data(lookup_route)
    geojson                = load_brazil_geojson()

    if geojson is None:
        st.warning("Não foi possível baixar o GeoJSON do Brasil. Exibindo mapa de bolhas como fallback.")

    # ── Mapa principal ────────────────────────────────────────────────────────
    st.plotly_chart(build_choropleth(state_risk, geojson), use_container_width=True)

    # ── KPIs abaixo do mapa ───────────────────────────────────────────────────
    st.divider()
    top_s  = state_risk.iloc[0]
    bot_s  = state_risk.iloc[-1]
    avg    = state_risk["taxa_pct"].mean()
    above  = (state_risk["taxa_pct"] > avg).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Estado mais crítico",
        top_s["estado"],
        f"{top_s['taxa_pct']:.1f}% de atraso",
        delta_color="inverse",
    )
    k2.metric(
        "Estado mais seguro",
        bot_s["estado"],
        f"{bot_s['taxa_pct']:.1f}% de atraso",
    )
    k3.metric("Taxa média geral",   f"{avg:.1f}%")
    k4.metric("Estados acima da média", f"{above} de {len(state_risk)}")

    # ── Gráfico de barras + Tabela de rotas ───────────────────────────────────
    st.divider()
    col_bar, col_tbl = st.columns([1, 1], gap="large")

    with col_bar:
        st.markdown("**Top 10 Estados de Destino por Risco Histórico**")
        st.plotly_chart(build_bar_chart(state_risk), use_container_width=True)

        # Legenda de cores
        leg1, leg2, leg3, leg4 = st.columns(4)
        leg1.markdown("🟢 **< 7%** Baixo")
        leg2.markdown("🟡 **7–10%** Moderado")
        leg3.markdown("🟠 **10–15%** Alto")
        leg4.markdown("🔴 **> 15%** Crítico")

    with col_tbl:
        st.markdown("**Top 15 Rotas com Maior Risco Histórico**")

        rename_map = {"route": "Rota", "hist_atraso_route_t": "Taxa Bruta (%)"}
        if "hist_atraso_route_smooth" in top_routes.columns:
            rename_map["hist_atraso_route_smooth"] = "Taxa Suavizada (%)"

        display_tbl = top_routes.rename(columns=rename_map)

        fmt = {"Taxa Bruta (%)": "{:.1f}"}
        if "Taxa Suavizada (%)" in display_tbl.columns:
            fmt["Taxa Suavizada (%)"] = "{:.1f}"

        styled = (
            display_tbl.style
            .background_gradient(subset=["Taxa Bruta (%)"], cmap="RdYlGn_r", vmin=0, vmax=30)
            .format(fmt)
        )
        st.dataframe(styled, use_container_width=True, height=430)

    # ── Insights automáticos ──────────────────────────────────────────────────
    st.divider()
    with st.expander("💡 Insights automáticos da base histórica", expanded=True):
        rota_critica = top_routes.iloc[0]["Rota"] if "Rota" in top_routes.columns else top_routes.iloc[0]["route"]
        taxa_critica = top_routes.iloc[0]["Taxa Bruta (%)"]

        estados_criticos = state_risk[state_risk["taxa_pct"] >= 15]["estado"].tolist()
        estados_str      = ", ".join(estados_criticos) if estados_criticos else "nenhum"

        st.markdown(f"""
| 🔍 Insight | Detalhe |
|---|---|
| **Rota mais crítica** | `{rota_critica}` com **{taxa_critica:.1f}%** de taxa histórica de atraso |
| **Estados com risco crítico (≥ 15%)** | {estados_str} |
| **Concentração de risco** | Os 5 estados mais críticos respondem por **{state_risk.head(5)['taxa_pct'].mean():.1f}%** de taxa média |
| **Estados abaixo de 5%** | {(state_risk['taxa_pct'] < 5).sum()} estados com risco considerado baixo |
| **Amplitude** | {state_risk['taxa_pct'].max():.1f}% (máximo) × {state_risk['taxa_pct'].min():.1f}% (mínimo) — amplitude de {state_risk['taxa_pct'].max() - state_risk['taxa_pct'].min():.1f} p.p. |
        """)

    with st.expander("📚 Metodologia — como a taxa histórica é calculada?"):
        st.markdown(f"""
**Fonte dos dados:** `lookup_route.parquet` — tabela gerada durante a engenharia temporal do notebook.

Para cada rota (`SELLER_STATE_CUSTOMER_STATE`), a taxa histórica foi calculada com:

```
hist_atraso_route_t = soma de atrasos da rota / pedidos observados
```

usando **apenas pedidos cujo resultado já era conhecido antes do momento da compra**
(`label_known_at < order_purchase_timestamp`). Nenhum dado futuro contamina o cálculo.

A **taxa suavizada** (Bayesian smoothing com α = 30) ancora rotas com poucos pedidos
ao prior global, evitando estimativas extremas (0% ou 100%) em rotas raras.

O **mapa** agrega a taxa bruta por estado de destino (média simples entre as rotas que chegam àquele estado).
        """)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Supply Chain Analytics Lab · Projeto 1 — OTD Prediction · "
    "Dataset Olist 2016–2018 · XGBoost + Engenharia Temporal Causal"
)
