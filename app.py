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
import geemap  # para ee_to_geojson
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# =====================
# CONFIG DA PÁGINA
# =====================
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align:left; font-size:40px;'>🌾 Monitoramento NDVI MODIS + Precipitação (ERA5-Land)</h1>",
    unsafe_allow_html=True
)

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

# =====================
# AUTENTICAÇÃO / EE INIT (CACHE)
# =====================
@st.cache_resource(show_spinner=False)
def init_ee_from_secrets():
    service_account_info = st.secrets["earthengine"]
    key_data = service_account_info["private_key"].replace('\\n', '\n')
    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"], key_data=key_data
    )
    ee.Initialize(credentials)
    return True

with st.spinner("Inicializando Earth Engine..."):
    init_ee_from_secrets()

# =====================
# COLEÇÕES DO USUÁRIO (CACHE)
# =====================
@st.cache_resource(show_spinner=False)
def get_user_collections():
    PIVOS_PT = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_PT")
    PIVOS_AREA = ee.FeatureCollection("users/lucaseducarvalho/PIVOS_AREA")
    return PIVOS_PT, PIVOS_AREA

PIVOS_PT, PIVOS_AREA = get_user_collections()

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_pivo_ids_sorted():
    return PIVOS_PT.sort('id_ref').aggregate_array('id_ref').getInfo()

pivo_ids = get_pivo_ids_sorted()

# =====================
# DADOS BASE (CACHE DE COLEÇÃO)
# =====================
@st.cache_resource(show_spinner=False)
def get_base_collections():
    # NDVI (MODIS) em escala 0.0001
    modis = (
        ee.ImageCollection('MODIS/061/MOD13Q1')
        .select('NDVI')
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ['system:time_start']))
    )
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select('total_precipitation')
    return modis, era5

modis, era5 = get_base_collections()

# =====================
# UTILIDADES (CACHEADAS)
# =====================
@st.cache_data(ttl=24*3600, show_spinner=False)
def ee_to_valid_geojson_cached(fc_path: str):
    """Carrega FC por caminho e converte para GeoJSON válido (removendo features sem geometria)."""
    fc = ee.FeatureCollection(fc_path)
    raw = geemap.ee_to_geojson(fc)
    valid_features = []
    for f in raw.get('features', []):
        geom = f.get('geometry')
        if geom and isinstance(geom, dict) and geom.get('coordinates'):
            valid_features.append(f)
    return {'type': 'FeatureCollection', 'features': valid_features}

@st.cache_data(ttl=24*3600, show_spinner=False)
def monthly_dates_range(start_date_str: str, end_date_str: str, include_current: bool):
    """Lista de datas mensais [start, end). Se include_current=True, inclui mês corrente até hoje; senão, até último mês completo."""
    start = ee.Date(start_date_str)
    today = ee.Date(datetime.datetime.now())
    if include_current:
        last_month_anchor = ee.Date.fromYMD(today.get('year'), today.get('month'), 1)
    else:
        last_month_anchor = ee.Date.fromYMD(today.get('year'), today.get('month'), 1).advance(-1, 'month')

    end = ee.Date(end_date_str) if end_date_str else last_month_anchor
    n = end.difference(start, 'month')
    return ee.List.sequence(0, n.subtract(1)).map(lambda m: start.advance(m, 'month'))

# =====================
# CÁLCULO DE SÉRIES (CACHE)
# =====================
@st.cache_data(ttl=24*3600, show_spinner=True)
def compute_ndvi_series_cached(pivo_id: int, start_date: str, end_date: str, scale_ndvi: int, include_current: bool):
    area = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(pivo_id))).first().geometry()
    dates = monthly_dates_range(start_date, end_date, include_current)

    def monthly_composite(date):
        start = ee.Date(date)
        end = start.advance(1, "month")
        filtered = modis.filterDate(start, end)

        def empty_case():
            return ee.Image.constant(0).rename('NDVI').clip(area).set({
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
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=scale_ndvi,
            bestEffort=True,
            maxPixels=1e13
        ).get('NDVI')
        return ee.Feature(None, {
            'date': ee.Date(image.get('system:time_start')).format('YYYY-MM'),
            'ndvi': ee.Algorithms.If(ee.Algorithms.IsEqual(mean, None), 0, mean)
        })

    series = monthly_collection.map(extract)
    feats = series.toList(series.size()).getInfo()
    df = pd.DataFrame([f['properties'] for f in feats])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['ndvi'] = pd.to_numeric(df['ndvi'], errors='coerce')
        df = df.dropna(subset=['ndvi']).sort_values('date')
    return df

@st.cache_data(ttl=24*3600, show_spinner=True)
def compute_precip_series_cached(pivo_id: int, start_date: str, end_date: str, scale_era5: int, include_current: bool):
    area = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(pivo_id))).first().geometry()
    dates = monthly_dates_range(start_date, end_date, include_current)

    def month_feature(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        monthly_sum_m = era5.filterDate(start, end).sum().clip(area)
        monthly_sum_mm = monthly_sum_m.multiply(1000).rename('precip_mm')
        mean_mm = monthly_sum_mm.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=scale_era5,  # resolução efetiva
            bestEffort=True,
            maxPixels=1e13
        ).get('precip_mm')
        return ee.Feature(None, {
            'date': start.format('YYYY-MM'),
            'precip_mm': ee.Algorithms.If(ee.Algorithms.IsEqual(mean_mm, None), 0, mean_mm)
        })

    fc = ee.FeatureCollection(dates.map(month_feature))
    feats = fc.toList(fc.size()).getInfo()
    df = pd.DataFrame([f['properties'] for f in feats])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['precip_mm'] = pd.to_numeric(df['precip_mm'], errors='coerce').fillna(0.0)
        df = df.sort_values('date')
    return df

# =====================
# AUXILIAR: SEGMENTOS <= LIMIAR
# =====================
import numpy as np

def segments_below_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['date', 'ndvi', 'seg'])
    df = df.sort_values('date').reset_index(drop=True)
    seg_rows, seg_id, in_seg, prev = [], 0, False, None

    for _, curr in df.iterrows():
        if prev is None:
            if curr['ndvi'] <= threshold:
                seg_id += 1
                in_seg = True
                seg_rows.append({'date': curr['date'], 'ndvi': float(curr['ndvi']), 'seg': seg_id})
            prev = curr
            continue

        p_ndvi = float(prev['ndvi'])
        c_ndvi = float(curr['ndvi'])
        p_below = p_ndvi <= threshold
        c_below = c_ndvi <= threshold

        if p_below and c_below:
            if not in_seg:
                seg_id += 1
                in_seg = True
            seg_rows.append({'date': prev['date'], 'ndvi': p_ndvi, 'seg': seg_id})
            seg_rows.append({'date': curr['date'], 'ndvi': c_ndvi, 'seg': seg_id})

        elif (not p_below) and c_below:
            t1_ns, t2_ns = prev['date'].value, curr['date'].value
            alpha = 0.0 if c_ndvi == p_ndvi else (threshold - p_ndvi) / (c_ndvi - p_ndvi)
            alpha = max(0.0, min(1.0, alpha))
            tc = pd.to_datetime(int(round(t1_ns + alpha * (t2_ns - t1_ns))))
            seg_id += 1
            in_seg = True
            seg_rows.append({'date': tc, 'ndvi': threshold, 'seg': seg_id})
            seg_rows.append({'date': curr['date'], 'ndvi': c_ndvi, 'seg': seg_id})

        elif p_below and (not c_below):
            t1_ns, t2_ns = prev['date'].value, curr['date'].value
            alpha = 0.0 if c_ndvi == p_ndvi else (threshold - p_ndvi) / (c_ndvi - p_ndvi)
            alpha = max(0.0, min(1.0, alpha))
            tc = pd.to_datetime(int(round(t1_ns + alpha * (t2_ns - t1_ns))))
            if not in_seg:
                seg_id += 1
                in_seg = True
            seg_rows.append({'date': prev['date'], 'ndvi': p_ndvi, 'seg': seg_id})
            seg_rows.append({'date': tc, 'ndvi': threshold, 'seg': seg_id})
            in_seg = False
        else:
            in_seg = False

        prev = curr

    if not seg_rows:
        return pd.DataFrame(columns=['date', 'ndvi', 'seg'])
    return pd.DataFrame(seg_rows, columns=['date', 'ndvi', 'seg']).drop_duplicates().sort_values('date')

# =====================
# SIDEBAR — PARÂMETROS DE PERFORMANCE
# =====================
st.sidebar.markdown("### Parâmetros")
selected_pivo = st.sidebar.selectbox("🧩 Selecione o Pivô", options=pivo_ids)

# Período — por padrão, últimos 24 meses
today = datetime.date.today()
default_start = (today.replace(day=1) - datetime.timedelta(days=730)).replace(day=1)
start_date = st.sidebar.date_input("Início (YYYY-MM-DD)", value=default_start)
end_date = st.sidebar.date_input("Fim (YYYY-MM-DD)", value=today)
include_current = st.sidebar.checkbox("Incluir mês corrente", value=False)

# Modo rápido: usa escalas mais grossas nas reduções (↑velocidade)
fast_mode = st.sidebar.toggle("⚡️ Modo rápido (mais rápido, menos preciso)", value=True, help="Usa escala 500–1000 m nas reduções espaciais.")

# Limiar NDVI
threshold = st.sidebar.slider("Limiar (NDVI)", 0.0, 1.0, 0.2, 0.01)
st.sidebar.caption("Linha verde contínua. Trechos NDVI ≤ limiar: linha e pontos vermelhos. Barras: precipitação mensal (mm).")

# Escalas das reduções
scale_ndvi = 500 if fast_mode else 250
scale_era5 = 9000  # ERA5-Land ~9km (não muda no fast_mode)

# =====================
# FLUXO PRINCIPAL
# =====================
if selected_pivo:
    with st.spinner("🔄 Carregando séries (NDVI + Precipitação) com cache..."):
        df_ndvi = compute_ndvi_series_cached(
            int(selected_pivo), start_date.isoformat(), end_date.isoformat(), scale_ndvi, include_current
        )
        df_prec = compute_precip_series_cached(
            int(selected_pivo), start_date.isoformat(), end_date.isoformat(), scale_era5, include_current
        )
        if df_ndvi is None or df_ndvi.empty:
            df = pd.DataFrame(columns=['date', 'ndvi', 'precip_mm'])
        else:
            df = pd.merge(df_ndvi, df_prec, on='date', how='left')
            df['ndvi'] = pd.to_numeric(df['ndvi'], errors='coerce')
            df['precip_mm'] = pd.to_numeric(df.get('precip_mm', 0.0), errors='coerce').fillna(0.0)
            df = df.dropna(subset=['ndvi']).sort_values('date')

    # 2) Cards de resumo
    st.subheader(f"Resumo do pivô {selected_pivo}")
    col1, col2, col3 = st.columns(3)
    if df.empty:
        col1.metric("NDVI último mês", "—")
        col2.metric("Média 3 meses", "—")
        col3.metric("Variação vs. mês anterior", "—")
    else:
        try:
            ult = float(df['ndvi'].iloc[-1])
            med3 = float(df['ndvi'].tail(3).mean())
            var = ult - float(df['ndvi'].iloc[-2]) if len(df) >= 2 else 0.0
        except Exception:
            ult, med3, var = 0.0, 0.0, 0.0
        col1.metric("NDVI último mês", f"{ult:.3f}")
        col2.metric("Média 3 meses", f"{med3:.3f}")
        col3.metric("Variação vs. mês anterior", f"{var:+.3f}")

    st.divider()

    # 3) MAPA (ajuste: clip no pivô)
    area_fc = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(selected_pivo))).first()
    area_geom = area_fc.geometry()

    # bounds para fit
    bounds = ee.Geometry(area_geom).bounds().coordinates().getInfo()[0]
    minx = min(c[0] for c in bounds); maxx = max(c[0] for c in bounds)
    miny = min(c[1] for c in bounds); maxy = max(c[1] for c in bounds)

    m = folium.Map()
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # NDVI do mês mais recente disponível
    last_date_pd = (df['date'].max() if not df.empty else pd.Timestamp(datetime.datetime.now()))
    last_date = last_date_pd.strftime('%Y-%m-%d')

    ndvi_image = (
        modis.filterDate(last_date, ee.Date(last_date).advance(1, 'month')).mean().clip(area_geom)
    )
    mapid = ndvi_image.getMapId({'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']})
    folium.TileLayer(
        tiles=mapid['tile_fetcher'].url_format,
        attr='GEE NDVI',
        name='NDVI',
        overlay=True,
        control=True
    ).add_to(m)

    # GeoJSONs (cacheados)
    pivos_area_geojson = ee_to_valid_geojson_cached("users/lucaseducarvalho/PIVOS_AREA")
    pivos_pt_geojson = ee_to_valid_geojson_cached("users/lucaseducarvalho/PIVOS_PT")

    if pivos_area_geojson['features']:
        folium.GeoJson(
            pivos_area_geojson,
            name='Área dos Pivôs',
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillOpacity': 0}
        ).add_to(m)

    # Rótulos com cluster (toggle via fast_mode para aliviar em coleções muito grandes)
    if pivos_pt_geojson['features'] and (not fast_mode or len(pivos_pt_geojson['features']) <= 2000):
        label_cluster = MarkerCluster(
            name="Rótulos",
            disableClusteringAtZoom=16,
            showCoverageOnHover=False,
            spiderfyOnMaxZoom=True,
            zoomToBoundsOnClick=True,
            chunkedLoading=True,
            maxClusterRadius=60
        ).add_to(m)

        for f in pivos_pt_geojson['features']:
            coords = f['geometry']['coordinates']
            label = str(f['properties'].get('id_ref', ''))
            folium.Marker(
                location=[coords[1], coords[0]],
                icon=folium.DivIcon(
                    html=f"""
                        <div style="
                            font-size:16px;
                            font-weight:bold;
                            color:black;
                            text-shadow:
                                -2px -2px 0 white,
                                2px -2px 0 white,
                                -2px 2px 0 white,
                                2px 2px 0 white,
                                0px -2px 0 white,
                                0px 2px 0 white,
                                -2px 0px 0 white,
                                2px 0px 0 white;">
                            {label}
                        </div>
                    """,
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                    class_name="pivot-label"
                )
            ).add_to(label_cluster)

    folium.LayerControl().add_to(m)

    tab1, tab2 = st.tabs(["🗺️ Mapa", "📈 Séries NDVI + Precip"])
    with tab1:
        st_folium(m, use_container_width=True, height=520)
        st.caption("Legenda NDVI: vermelho (↓) → amarelo → verde (↑). Barras: precipitação mensal acumulada (ERA5-Land).")

    # 4) GRÁFICO COMBINADO
    with tab2:
        if df.empty:
            st.warning("Sem dados para o período/área selecionados (NDVI/precipitação).")
        else:
            ndvi_min = max(threshold, float(df['ndvi'].min())) if not df.empty else 0.2
            ndvi_max = float(df['ndvi'].max()) if not df.empty else 1.0
            ndvi_scale = alt.Scale(domain=[ndvi_min, ndvi_max])

            bars_precip = alt.Chart(df).mark_bar(color='#3b82f6', opacity=0.5).encode(
                x=alt.X('date:T', title='Data'),
                y=alt.Y('precip_mm:Q', title='Precipitação (mm/mês)', axis=alt.Axis(titleColor='#3b82f6')),
                tooltip=[
                    alt.Tooltip('date:T', title='Data'),
                    alt.Tooltip('precip_mm:Q', title='Precipitação (mm)', format=".2f")
                ]
            )

            line_green_full = alt.Chart(df).mark_line(color='green', strokeWidth=2).encode(
                x=alt.X('date:T', title='Data'),
                y=alt.Y('ndvi:Q', title='NDVI', scale=ndvi_scale,
                        axis=alt.Axis(format=".3f", orient='right', titleColor='green'))
            )

            df_below = segments_below_threshold(df[['date', 'ndvi']].copy(), threshold)
            line_red_overlay = alt.Chart(df_below).mark_line(color='red', strokeWidth=3).encode(
                x='date:T',
                y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale),
                detail='seg:N'
            )

            points_green = alt.Chart(df[df['ndvi'] > threshold]).mark_point(
                color='green', filled=True, opacity=1
            ).encode(
                x='date:T',
                y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale),
                tooltip=[alt.Tooltip('date:T', title='Data'),
                         alt.Tooltip('ndvi:Q', title='NDVI', format=".3f")]
            )

            points_red = alt.Chart(df[df['ndvi'] <= threshold]).mark_point(
                color='red', filled=True, opacity=1
            ).encode(
                x='date:T',
                y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale),
                tooltip=[alt.Tooltip('date:T', title='Data'),
                         alt.Tooltip('ndvi:Q', title='NDVI', format=".3f")]
            )

            chart = alt.layer(
                bars_precip, line_green_full, line_red_overlay, points_green, points_red
            ).resolve_scale(
                y='independent'
            ).properties(
                title=f'NDVI (linha) x Precipitação (barras) - Pivô {selected_pivo}',
                width='container',
                height=360
            ).configure_axis(
                grid=True,
                gridOpacity=0.15,
                labelFontSize=11,
                titleFontSize=12
            ).configure_view(
                strokeWidth=0
            ).configure_title(
                fontSize=14
            )

            st.altair_chart(chart, use_container_width=True)

    # CSV download (PT-BR friendly)
    csv = df[['date', 'ndvi', 'precip_mm']].copy()
    csv['date'] = csv['date'].dt.strftime('%Y-%m')
    csv_str = csv.to_csv(index=False, sep=';', decimal=',')
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="ndvi_precip_{selected_pivo}.csv">📥 Baixar dados (NDVI + Precip) CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Aviso de zeros consecutivos
    if not df.empty and (df['ndvi'] == 0).rolling(3).sum().max() >= 3:
        st.info("ℹ️ Muitos meses com NDVI=0. Pode indicar ausência de cenas válidas no período/área.")
