import os
import datetime
import base64

import streamlit as st
import ee
import pandas as pd
import altair as alt
import folium
import geemap
from streamlit_folium import st_folium
from folium.features import GeoJsonPopup

# =====================
# CONFIG
# =====================
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:left; font-size:40px;'>🌾 Monitoramento NDVI MODIS + Precipitação (ERA5-Land)</h1>", unsafe_allow_html=True)
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

# =====================
# EE INIT (secrets)
# =====================
sa = st.secrets["earthengine"]
key = sa["private_key"].replace("\\n", "\n")
creds = ee.ServiceAccountCredentials(sa["client_email"], key)
ee.Initialize(creds)

# =====================
# DADOS EE
# =====================
PIVOS_PT = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_PT")
PIVOS_AREA = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_AREA")

modis = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
    .select("NDVI")
    .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
)
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("total_precipitation")

# =====================
# HELPERS
# =====================
def monthly_dates(start_date_str="2021-01-01"):
    start = ee.Date(start_date_str)
    today = ee.Date(datetime.datetime.now())
    last_full_month = ee.Date.fromYMD(today.get('year'), today.get('month'), 1).advance(-1, 'month')
    n = last_full_month.difference(start, 'month')
    return ee.List.sequence(0, n.subtract(1)).map(lambda m: start.advance(m, 'month'))

@st.cache_data(show_spinner=False, ttl=24*3600)
def compute_ndvi_series(pivo_id: int) -> pd.DataFrame:
    area = PIVOS_AREA.filter(ee.Filter.eq("id_ref", int(pivo_id))).first().geometry()
    dates = monthly_dates()

    def monthly_ndvi(date):
        start = ee.Date(date); end = start.advance(1, "month")
        filtered = modis.filterDate(start, end)
        def empty_case():
            return ee.Image.constant(0).rename("NDVI").clip(area).set({"system:time_start": start.millis()})
        def non_empty_case():
            return filtered.mean().focal_mean(3, "square", "pixels").clip(area).set({"system:time_start": start.millis()})
        return ee.Image(ee.Algorithms.If(filtered.size().eq(0), empty_case(), non_empty_case()))
    coll = ee.ImageCollection(dates.map(monthly_ndvi))

    def extract(img):
        mean = img.reduceRegion(ee.Reducer.mean(), area, 250, bestEffort=True, maxPixels=1e13).get("NDVI")
        return ee.Feature(None, {
            "date": ee.Date(img.get("system:time_start")).format("YYYY-MM"),
            "ndvi": ee.Algorithms.If(ee.Algorithms.IsEqual(mean, None), 0, mean)
        })
    feats = coll.map(extract).toList(coll.size()).getInfo()
    df = pd.DataFrame([f["properties"] for f in feats])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
        df = df.dropna(subset=["ndvi"]).sort_values("date")
    return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def compute_precip_series(pivo_id: int) -> pd.DataFrame:
    area = PIVOS_AREA.filter(ee.Filter.eq("id_ref", int(pivo_id))).first().geometry()
    dates = monthly_dates()

    def month_prec(date):
        start = ee.Date(date); end = start.advance(1, "month")
        monthly_sum_m = era5.filterDate(start, end).sum().clip(area)
        monthly_sum_mm = monthly_sum_m.multiply(1000).rename("precip_mm")
        mean_mm = monthly_sum_mm.reduceRegion(ee.Reducer.mean(), area, 9000, bestEffort=True, maxPixels=1e13).get("precip_mm")
        return ee.Feature(None, {"date": start.format("YYYY-MM"),
                                 "precip_mm": ee.Algorithms.If(ee.Algorithms.IsEqual(mean_mm, None), 0, mean_mm)})
    feats = ee.FeatureCollection(dates.map(month_prec)).toList(dates.size()).getInfo()
    df = pd.DataFrame([f["properties"] for f in feats])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0)
        df = df.sort_values("date")
    return df

@st.cache_data(ttl=24*3600, show_spinner=False)
def pivos_ids_sorted():
    return PIVOS_PT.sort('id_ref').aggregate_array('id_ref').getInfo()

@st.cache_data(ttl=24*3600, show_spinner=False)
def all_pivots_geojson_simplified(tol_m: float = 3.0):
    """GeoJSON de TODOS os pivôs **simplificados** para performance, com id_ref e popup de clique."""
    fc_simple = PIVOS_AREA.map(lambda f: ee.Feature(f.geometry().simplify(tol_m), {'id_ref': f.get('id_ref')}))
    gj = geemap.ee_to_geojson(fc_simple)
    # remove features sem geometria
    feats = []
    for f in gj.get("features", []):
        g = f.get("geometry")
        if g and isinstance(g, dict) and g.get("coordinates"):
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}

@st.cache_data(ttl=3600, show_spinner=False)
def ndvi_tile_for_pivot(pivo_id: int, last_date_str: str):
    area = PIVOS_AREA.filter(ee.Filter.eq("id_ref", int(pivo_id))).first().geometry()
    ndvi_img = modis.filterDate(last_date_str, ee.Date(last_date_str).advance(1, "month")).mean().clip(area)
    return ndvi_img.getMapId({'min': 0, 'max': 1, 'palette': ['#ff0000', '#ffff00', '#00a000']})

# =====================
# SIDEBAR + ESTADO
# =====================
pivo_ids = pivos_ids_sorted()
if "selected_pivo" not in st.session_state:
    st.session_state["selected_pivo"] = pivo_ids[0] if pivo_ids else None

st.sidebar.markdown("### Parâmetros")
selected_pivo = st.sidebar.selectbox("🧩 Selecione o Pivô", options=pivo_ids, index=pivo_ids.index(st.session_state["selected_pivo"]) if st.session_state["selected_pivo"] in pivo_ids else 0, key="selected_pivo")
threshold = st.sidebar.slider("Limiar (NDVI)", 0.0, 1.0, 0.2, 0.01)
st.sidebar.caption("Clique no polígono para ver o número do pivô. Eixo NDVI fixo começa em 0,2.")

# =====================
# SÉRIES (independem do mapa)
# =====================
with st.spinner("🔄 Carregando séries (NDVI + Precipitação)..."):
    df_ndvi = compute_ndvi_series(int(selected_pivo))
    df_prec = compute_precip_series(int(selected_pivo))
    df = pd.merge(df_ndvi, df_prec, on="date", how="left") if not df_ndvi.empty else pd.DataFrame(columns=["date","ndvi","precip_mm"])

# =====================
# RESUMO
# =====================
st.subheader(f"Resumo do pivô {selected_pivo}")
col1, col2, col3 = st.columns(3)
if df.empty:
    col1.metric("NDVI último mês", "—")
    col2.metric("Média 3 meses", "—")
    col3.metric("Variação vs. mês anterior", "—")
else:
    try:
        ult = float(df["ndvi"].iloc[-1])
        med3 = float(df["ndvi"].tail(3).mean())
        var = ult - float(df["ndvi"].iloc[-2]) if len(df) >= 2 else 0.0
    except Exception:
        ult, med3, var = 0.0, 0.0, 0.0
    col1.metric("NDVI último mês", f"{ult:.3f}")
    col2.metric("Média 3 meses", f"{med3:.3f}")
    col3.metric("Variação vs. mês anterior", f"{var:+.3f}")

st.divider()

# =====================
# TABS
# =====================
tab1, tab2 = st.tabs(["🗺️ Mapa", "📈 Séries NDVI + Precip"])

# =====================
# MAPA (ÚNICO)
# =====================
with tab1:
    # Centro no pivô selecionado
    try:
        center_coords = PIVOS_PT.filter(ee.Filter.eq("id_ref", int(selected_pivo))).first().geometry().centroid().coordinates().getInfo()
        lat_center, lon_center = center_coords[1], center_coords[0]
    except Exception:
        lat_center, lon_center = -15.0, -55.0

    m = folium.Map(location=[lat_center, lon_center], zoom_start=14, prefer_canvas=True)

    # Tile NDVI (apenas sobre o pivô selecionado — leve)
    last_date = (df["date"].max() if not df.empty else pd.Timestamp(datetime.datetime.now())).strftime("%Y-%m-%d")
    try:
        ndvi_mapid = ndvi_tile_for_pivot(int(selected_pivo), last_date)
        folium.TileLayer(
            tiles=ndvi_mapid['tile_fetcher'].url_format,
            attr='GEE NDVI',
            name='NDVI',
            overlay=True,
            control=False
        ).add_to(m)
    except Exception:
        pass

    # Todos os pivôs (geojson simplificado) — popup no clique com id_ref
    try:
        gj_all = all_pivots_geojson_simplified(tol_m=3.0)  # ajuste se quiser mais/menos detalhe
        if gj_all["features"]:
            folium.GeoJson(
                gj_all,
                name="Pivôs",
                style_function=lambda x: {'color': '#1f2937', 'weight': 1, 'fillOpacity': 0.0},
                highlight_function=lambda x: {'color': '#2563eb', 'weight': 3, 'fillOpacity': 0.0},
                popup=GeoJsonPopup(
                    fields=["id_ref"],
                    aliases=["Pivô: "],
                    labels=True,
                    localize=True
                )
            ).add_to(m)
    except Exception as e:
        st.warning("Falha ao carregar a camada de pivôs.")
        st.exception(e)

    st_folium(m, use_container_width=True, height=520)

# =====================
# GRÁFICO (NDVI fixo 0.2–1)
# =====================
with tab2:
    if df.empty:
        st.warning("Sem dados para o período/área selecionados (NDVI/precipitação).")
    else:
        ndvi_scale = alt.Scale(domain=[0.2, 1])
        bars_precip = alt.Chart(df).mark_bar(color="#3b82f6", opacity=0.5).encode(
            x=alt.X('date:T', title='Data'),
            y=alt.Y('precip_mm:Q', title='Precipitação (mm/mês)', axis=alt.Axis(titleColor='#3b82f6')),
            tooltip=[alt.Tooltip('date:T', title='Data'),
                     alt.Tooltip('precip_mm:Q', title='Precipitação (mm)', format=".2f")]
        )
        line_ndvi = alt.Chart(df).mark_line(color='green', strokeWidth=2).encode(
            x=alt.X('date:T', title='Data'),
            y=alt.Y('ndvi:Q', title='NDVI', scale=ndvi_scale,
                    axis=alt.Axis(format=".3f", orient='right', titleColor='green'))
        )
        st.altair_chart(
            alt.layer(bars_precip, line_ndvi).resolve_scale(y='independent').properties(
                title=f'NDVI (linha) x Precipitação (barras) - Pivô {selected_pivo}',
                width='container', height=360
            ).configure_axis(grid=True, gridOpacity=0.15, labelFontSize=11, titleFontSize=12).configure_view(strokeWidth=0),
            use_container_width=True
        )

# =====================
# EXPORT CSV
# =====================
if not df.empty:
    csv = df[['date','ndvi','precip_mm']].copy()
    csv['date'] = csv['date'].dt.strftime('%Y-%m')
    b64 = base64.b64encode(csv.to_csv(index=False).encode()).decode()
    st.markdown(f'<a href="data:text/csv;base64,{b64}" download="ndvi_precip_{selected_pivo}.csv">📥 Baixar dados (NDVI + Precip) CSV</a>', unsafe_allow_html=True)
