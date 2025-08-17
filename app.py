import os
import datetime
import base64
import json
import streamlit as st
import tempfile
from google.oauth2 import service_account
import ee
import pandas as pd
import altair as alt
import folium
import geemap
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# ---------------- CONFIGURAÇÃO ----------------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:left; font-size:40px;'>🌾 Monitoramento NDVI MODIS + Precipitação (ERA5-Land)</h1>", unsafe_allow_html=True)
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

service_account_info = st.secrets["earthengine"]
credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"], key_data=service_account_info["private_key"]
)
ee.Initialize(credentials)

PIVOS_PT = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_PT")
PIVOS_AREA = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_AREA")

modis = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterDate("2021-01-01", ee.Date(datetime.datetime.now()))
    .select("NDVI")
    .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
)
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select("total_precipitation")

# ---------------- UTILITÁRIOS ----------------
def monthly_dates(start_date_str="2021-01-01"):
    start_date = ee.Date(start_date_str)
    today = ee.Date(datetime.datetime.now())
    months = ee.List.sequence(0, today.difference(start_date, "month").round().subtract(1))
    dates = months.map(lambda m: start_date.advance(m, "month"))
    return dates

@st.cache_data(show_spinner=False)
def compute_ndvi_series(pivo_id):
    area = PIVOS_AREA.filter(ee.Filter.eq("id_ref", int(pivo_id))).first().geometry()
    dates = monthly_dates()

    def monthly_composite(date):
        start = ee.Date(date); end = start.advance(1, "month")
        filtered = modis.filterDate(start, end)

        def empty_case():
            return ee.Image.constant(0).rename("NDVI").clip(area).set({
                "month": start.format("YYYY-MM"),
                "system:time_start": start.millis()
            })
        def non_empty_case():
            ndvi = filtered.mean().focal_mean(3, "square", "pixels")
            return ndvi.clip(area).set({
                "month": start.format("YYYY-MM"),
                "system:time_start": start.millis()
            })
        return ee.Image(ee.Algorithms.If(filtered.size().eq(0), empty_case(), non_empty_case()))

    monthly_collection = ee.ImageCollection(dates.map(monthly_composite))
    def extract(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=area, scale=250,
            bestEffort=True, maxPixels=1e13
        ).get("NDVI")
        return ee.Feature(None, {
            "date": ee.Date(image.get("system:time_start")).format("YYYY-MM"),
            "ndvi": ee.Algorithms.If(ee.Algorithms.IsEqual(mean, None), 0, mean)
        })

    series = monthly_collection.map(extract).toList(monthly_collection.size())
    features = series.getInfo()
    df = pd.DataFrame([f["properties"] for f in features])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
        df = df.dropna(subset=["ndvi"]).sort_values("date")
    return df

@st.cache_data(show_spinner=False)
def compute_precip_series(pivo_id):
    area = PIVOS_AREA.filter(ee.Filter.eq("id_ref", int(pivo_id))).first().geometry()
    dates = monthly_dates()

    def month_feature(date):
        start = ee.Date(date); end = start.advance(1, "month")
        monthly_sum_m = era5.filterDate(start, end).sum().clip(area)
        monthly_sum_mm = monthly_sum_m.multiply(1000).rename("precip_mm")
        mean_mm = monthly_sum_mm.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=area, scale=9000,
            bestEffort=True, maxPixels=1e13
        ).get("precip_mm")
        return ee.Feature(None, {
            "date": start.format("YYYY-MM"),
            "precip_mm": ee.Algorithms.If(ee.Algorithms.IsEqual(mean_mm, None), 0, mean_mm)
        })

    fc = ee.FeatureCollection(dates.map(month_feature))
    feats = fc.toList(fc.size()).getInfo()
    df = pd.DataFrame([f["properties"] for f in feats])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0)
        df = df.sort_values("date")
    return df

def get_all_centroids_df():
    feats = PIVOS_PT.getInfo()["features"]
    rows = []
    for f in feats:
        coords = f["geometry"]["coordinates"]
        pid = f["properties"].get("id_ref")
        rows.append({"lat": coords[1], "lon": coords[0], "pid": pid})
    return pd.DataFrame(rows)

# ---------------- SIDEBAR ----------------
pivo_ids = PIVOS_PT.aggregate_array("id_ref").sort().getInfo()
st.sidebar.markdown("### Parâmetros")
selected_pivo = st.sidebar.selectbox("🧩 Selecione o Pivô", options=pivo_ids)
threshold = st.sidebar.slider("Limiar (NDVI)", 0.0, 1.0, 0.2, 0.01)

# ---------------- FLUXO ----------------
if selected_pivo:
    with st.spinner("🔄 Carregando séries (NDVI + Precipitação)..."):
        df_ndvi = compute_ndvi_series(selected_pivo)
        df_prec = compute_precip_series(selected_pivo)
        df = pd.merge(df_ndvi, df_prec, on="date", how="left") if not df_ndvi.empty else pd.DataFrame()

    st.subheader(f"Resumo do pivô {selected_pivo}")
    col1, col2, col3 = st.columns(3)
    if df.empty:
        col1.metric("NDVI último mês", "—")
        col2.metric("Média 3 meses", "—")
        col3.metric("Variação vs. mês anterior", "—")
    else:
        ult = float(df["ndvi"].iloc[-1])
        med3 = float(df["ndvi"].tail(3).mean())
        var = ult - float(df["ndvi"].iloc[-2]) if len(df) >= 2 else 0.0
        col1.metric("NDVI último mês", f"{ult:.3f}")
        col2.metric("Média 3 meses", f"{med3:.3f}")
        col3.metric("Variação vs. mês anterior", f"{var:+.3f}")

    st.divider()

    # -------------- MAPA + GRÁFICO EM TABS ----------------
    tab1, tab2 = st.tabs(["🗺️ Mapa", "📈 Séries NDVI + Precip"])

    with tab1:
        # centro do pivô selecionado
        pivo_geom = PIVOS_PT.filter(ee.Filter.eq("id_ref", int(selected_pivo))).first().geometry().centroid().coordinates().getInfo()
        lat_center, lon_center = pivo_geom[1], pivo_geom[0]
        m = folium.Map(location=[lat_center, lon_center], zoom_start=15, prefer_canvas=True)

        # NDVI do mês mais recente
        last_date = (df["date"].max() if not df.empty else pd.Timestamp(datetime.datetime.now())).strftime("%Y-%m-%d")
        ndvi_image = modis.filterDate(last_date, ee.Date(last_date).advance(1, "month")).mean().clip(PIVOS_AREA)
        mapid = ndvi_image.getMapId({"min": 0, "max": 1, "palette": ["red", "yellow", "green"]})
        folium.TileLayer(tiles=mapid["tile_fetcher"].url_format, attr="GEE NDVI", name="NDVI", overlay=True).add_to(m)

        # Áreas
        geo_area = geemap.ee_to_geojson(PIVOS_AREA)
        if geo_area and geo_area.get("features"):
            folium.GeoJson(geo_area, name="Área dos Pivôs",
                           style_function=lambda x: {"color": "blue", "weight": 2, "fillOpacity": 0}).add_to(m)

        # Marcadores numerados
        df_pts = get_all_centroids_df()
        cluster = MarkerCluster(
            name="Pivôs", disableClusteringAtZoom=16,
            showCoverageOnHover=False, spiderfyOnMaxZoom=True,
            zoomToBoundsOnClick=True, chunkedLoading=True
        ).add_to(m)

        try:
            from folium.plugins import BeautifyIcon
            def add_marker(lat, lon, pid):
                icon = BeautifyIcon(number=str(pid),
                                    border_color="#1f2937", text_color="#111827",
                                    background_color="#ffffff", icon_shape="circle", border_width=2)
                folium.Marker(location=[lat, lon], icon=icon).add_to(cluster)
        except Exception:
            def add_marker(lat, lon, pid):
                folium.Marker(location=[lat, lon], tooltip=str(pid)).add_to(cluster)

        for _, row in df_pts.iterrows():
            add_marker(row["lat"], row["lon"], row["pid"])

        folium.LayerControl().add_to(m)
        st_folium(m, use_container_width=True, height=520)

    with tab2:
        if df.empty:
            st.warning("Sem dados.")
        else:
            ndvi_scale = alt.Scale(domain=[0.2, 1])
            bars_precip = alt.Chart(df).mark_bar(color="#3b82f6", opacity=0.5).encode(
                x="date:T", y=alt.Y("precip_mm:Q", title="Precipitação (mm/mês)", axis=alt.Axis(titleColor="#3b82f6"))
            )
            line_green = alt.Chart(df).mark_line(color="green", strokeWidth=2).encode(
                x="date:T", y=alt.Y("ndvi:Q", title="NDVI", scale=ndvi_scale,
                                    axis=alt.Axis(format=".3f", orient="right", titleColor="green"))
            )
            chart = alt.layer(bars_precip, line_green).resolve_scale(y="independent").properties(
                title=f"NDVI x Precipitação - Pivô {selected_pivo}", width="container", height=360
            )
            st.altair_chart(chart, use_container_width=True)

    # CSV
    csv = df[["date", "ndvi", "precip_mm"]].copy()
    csv["date"] = csv["date"].dt.strftime("%Y-%m")
    b64 = base64.b64encode(csv.to_csv(index=False).encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="ndvi_precip_{selected_pivo}.csv">📥 Baixar CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
