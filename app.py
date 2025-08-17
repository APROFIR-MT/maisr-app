import os
import datetime
import base64
import urllib.parse
from math import cos, radians

import streamlit as st
import ee
import pandas as pd
import altair as alt
import folium
import geemap
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.features import CustomIcon

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
# CACHE EM DISCO
# =====================
CACHE_DIR = "data_cache"
NDVI_CACHE = os.path.join(CACHE_DIR, "ndvi")
PREC_CACHE = os.path.join(CACHE_DIR, "prec")
MERGED_CACHE = os.path.join(CACHE_DIR, "merged")
for p in (CACHE_DIR, NDVI_CACHE, PREC_CACHE, MERGED_CACHE):
    os.makedirs(p, exist_ok=True)

# =====================
# EARTH ENGINE
# =====================
@st.cache_resource(show_spinner=False)
def init_ee():
    sa = st.secrets["earthengine"]
    key = sa["private_key"].replace("\\n", "\n")
    creds = ee.ServiceAccountCredentials(sa["client_email"], key_data=key)
    ee.Initialize(creds)
    return True

with st.spinner("Inicializando Earth Engine..."):
    init_ee()

@st.cache_resource(show_spinner=False)
def get_user_collections():
    return (
        ee.FeatureCollection("users/lucaseducarvalho/PIVOS_PT"),
        ee.FeatureCollection("users/lucaseducarvalho/PIVOS_AREA"),
    )
PIVOS_PT, PIVOS_AREA = get_user_collections()

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_pivo_ids_sorted():
    return PIVOS_PT.sort('id_ref').aggregate_array('id_ref').getInfo()
pivo_ids = get_pivo_ids_sorted()

# Preserva seleção do pivô e estado do mapa
if "selected_pivo" not in st.session_state:
    st.session_state["selected_pivo"] = pivo_ids[0] if pivo_ids else None
if "map_bounds" not in st.session_state:
    st.session_state["map_bounds"] = None  # dict: {'north':..,'south':..,'east':..,'west':..}
if "map_center" not in st.session_state:
    st.session_state["map_center"] = None  # (lat, lon)
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = None

@st.cache_resource(show_spinner=False)
def get_base_collections():
    modis = (
        ee.ImageCollection('MODIS/061/MOD13Q1')
        .select('NDVI')
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ['system:time_start']))
    )
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select('total_precipitation')
    return modis, era5
modis, era5 = get_base_collections()

# =====================
# DATAS MENSAL
# =====================
@st.cache_data(ttl=24*3600, show_spinner=False)
def monthly_dates(start_date_str='2021-01-01'):
    start = ee.Date(start_date_str)
    today = ee.Date(datetime.datetime.now())
    last_full_month = ee.Date.fromYMD(today.get('year'), today.get('month'), 1).advance(-1, 'month')
    n = last_full_month.difference(start, 'month')
    return ee.List.sequence(0, n.subtract(1)).map(lambda m: start.advance(m, 'month'))

# =====================
# CACHE DE SÉRIES
# =====================
def _paths_for(pivo_id: int):
    pid = int(pivo_id)
    return (
        os.path.join(NDVI_CACHE, f"ndvi_{pid}.parquet"),
        os.path.join(PREC_CACHE, f"prec_{pid}.parquet"),
        os.path.join(MERGED_CACHE, f"ndvi_prec_{pid}.parquet"),
    )

@st.cache_data(show_spinner=True)
def build_and_store_series_for_pivot(pivo_id: int, start_date: str = '2021-01-01') -> str:
    area = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(pivo_id))).first().geometry()
    dates = monthly_dates(start_date)

    # NDVI
    def monthly_ndvi(date):
        start = ee.Date(date)
        end = start.advance(1, "month")
        filtered = modis.filterDate(start, end)
        def empty_case():
            return ee.Image.constant(0).rename('NDVI').clip(area).set({"system:time_start": start.millis()})
        def non_empty_case():
            ndvi = filtered.mean().focal_mean(3, "square", "pixels")
            return ndvi.clip(area).set({"system:time_start": start.millis()})
        return ee.Image(ee.Algorithms.If(filtered.size().eq(0), empty_case(), non_empty_case()))
    ndvi_coll = ee.ImageCollection(dates.map(monthly_ndvi))

    def ndvi_extract(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=area, scale=250, bestEffort=True, maxPixels=1e13
        ).get('NDVI')
        return ee.Feature(None, {
            'date': ee.Date(image.get('system:time_start')).format('YYYY-MM'),
            'ndvi': ee.Algorithms.If(ee.Algorithms.IsEqual(mean, None), 0, mean)
        })
    ndvi_fc = ndvi_coll.map(ndvi_extract)
    df_ndvi = pd.DataFrame([f['properties'] for f in ndvi_fc.toList(ndvi_fc.size()).getInfo()])
    if not df_ndvi.empty:
        df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
        df_ndvi['ndvi'] = pd.to_numeric(df_ndvi['ndvi'], errors='coerce')
        df_ndvi = df_ndvi.dropna(subset=['ndvi']).sort_values('date')

    # Precip
    def month_prec(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        monthly_sum_m = era5.filterDate(start, end).sum().clip(area)
        monthly_sum_mm = monthly_sum_m.multiply(1000).rename('precip_mm')
        mean_mm = monthly_sum_mm.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=area, scale=9000, bestEffort=True, maxPixels=1e13
        ).get('precip_mm')
        return ee.Feature(None, {
            'date': start.format('YYYY-MM'),
            'precip_mm': ee.Algorithms.If(ee.Algorithms.IsEqual(mean_mm, None), 0, mean_mm)
        })
    prec_fc = ee.FeatureCollection(dates.map(month_prec))
    df_prec = pd.DataFrame([f['properties'] for f in prec_fc.toList(prec_fc.size()).getInfo()])
    if not df_prec.empty:
        df_prec['date'] = pd.to_datetime(df_prec['date'])
        df_prec['precip_mm'] = pd.to_numeric(df_prec['precip_mm'], errors='coerce').fillna(0.0)
        df_prec = df_prec.sort_values('date')

    df = pd.merge(df_ndvi, df_prec, on='date', how='left') if not df_ndvi.empty else pd.DataFrame(columns=['date','ndvi','precip_mm'])
    ndvi_path, prec_path, merged_path = _paths_for(pivo_id)
    df_ndvi.to_parquet(ndvi_path, index=False)
    df_prec.to_parquet(prec_path, index=False)
    df.to_parquet(merged_path, index=False)
    return merged_path

@st.cache_data(show_spinner=False)
def merged_from_cache(pivo_id: int) -> pd.DataFrame:
    _, _, merged_path = _paths_for(pivo_id)
    if not os.path.exists(merged_path):
        build_and_store_series_for_pivot(int(pivo_id))
    return pd.read_parquet(merged_path)

# =====================
# MAP HELPERS
# =====================
def svg_number_icon_data_uri(number: int, size: int = 26) -> str:
    r = size // 2
    font_size = int(size * 0.55)
    svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <circle cx="{r}" cy="{r}" r="{r-1}" fill="#ffffff" stroke="#1f2937" stroke-width="2"/>
  <text x="50%" y="52%" font-family="Arial, Helvetica, sans-serif" font-size="{font_size}" font-weight="700"
        text-anchor="middle" dominant-baseline="middle" fill="#111827">{number}</text>
</svg>'''.strip()
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

@st.cache_data(ttl=3600, show_spinner=False)
def get_ndvi_mapid_for(pivo_id: int, last_date: str):
    area = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(pivo_id))).first().geometry()
    ndvi_image = modis.filterDate(last_date, ee.Date(last_date).advance(1, 'month')).mean().clip(area)
    vis = {'min': 0, 'max': 1, 'palette': ['#ff0000', '#ffff00', '#00a000']}
    return ndvi_image.getMapId(vis)

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_pivots_outline_mapid():
    edges = ee.Image().paint(PIVOS_AREA, 1, 2)  # 2 px
    vis = {'min': 0, 'max': 1, 'palette': ['#000000']}
    return edges.getMapId(vis)

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_all_centroids_list():
    fc = PIVOS_PT.select(['id_ref']).map(
        lambda f: ee.Feature(ee.Geometry(f.geometry()).centroid(), {'id_ref': f.get('id_ref')})
    )
    feats = fc.getInfo()['features']
    # retorna DataFrame para filtro vetorizado
    dfc = pd.DataFrame({
        "lat": [f['geometry']['coordinates'][1] for f in feats],
        "lon": [f['geometry']['coordinates'][0] for f in feats],
        "pid": [f['properties'].get('id_ref') for f in feats],
    })
    return dfc

@st.cache_data(ttl=24*3600, show_spinner=False)
def get_selected_area_geojson(pivo_id: int, simplify_m: float = 2.0):
    feat = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(pivo_id))).first()
    if feat is None:
        return None
    geom = ee.Geometry(feat.geometry()).simplify(simplify_m)
    fc_sel = ee.FeatureCollection([ee.Feature(geom, {'id_ref': int(pivo_id)})])
    gj = geemap.ee_to_geojson(fc_sel)
    feats = []
    for f in gj.get('features', []):
        g = f.get('geometry')
        if g and isinstance(g, dict) and g.get('coordinates'):
            feats.append(f)
    return {'type': 'FeatureCollection', 'features': feats}

# =====================
# SIDEBAR (estado estável)
# =====================
st.sidebar.markdown("### Parâmetros")
if st.session_state["selected_pivo"] in pivo_ids:
    default_index = pivo_ids.index(st.session_state["selected_pivo"])
else:
    default_index = 0

selected_pivo = st.sidebar.selectbox(
    "🧩 Selecione o Pivô",
    options=pivo_ids,
    index=default_index,
    key="selected_pivo",
)
threshold = st.sidebar.slider("Limiar (NDVI)", 0.0, 1.0, 0.2, 0.01)
st.sidebar.caption("Linha verde contínua. Trechos NDVI ≤ limiar: linha e pontos vermelhos. Barras: precipitação mensal (mm).")

# =====================
# SÉRIES
# =====================
with st.spinner("🔄 Lendo séries (NDVI + Precipitação) do cache local..."):
    df = merged_from_cache(int(selected_pivo)) if selected_pivo is not None else pd.DataFrame(columns=['date','ndvi','precip_mm'])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['ndvi'] = pd.to_numeric(df['ndvi'], errors='coerce')
        df['precip_mm'] = pd.to_numeric(df.get('precip_mm', 0.0), errors='coerce').fillna(0.0)
        df = df.dropna(subset=['ndvi']).sort_values('date')

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
    ult = float(df['ndvi'].iloc[-1])
    med3 = float(df['ndvi'].tail(3).mean())
    var = ult - float(df['ndvi'].iloc[-2]) if len(df) >= 2 else 0.0
    col1.metric("NDVI último mês", f"{ult:.3f}")
    col2.metric("Média 3 meses", f"{med3:.3f}")
    col3.metric("Variação vs. mês anterior", f"{var:+.3f}")

st.divider()

# =====================
# MAPA (2-pass: captura bounds primeiro, depois desenha pinos)
# =====================
last_date_pd = (df['date'].max() if not df.empty else pd.Timestamp(datetime.datetime.now()))
last_date = last_date_pd.strftime('%Y-%m-%d')

# Passo A: mapa básico (tiles + enquadrar) — SEM pinos
m = folium.Map(prefer_canvas=True)
try:
    area_fc = PIVOS_AREA.filter(ee.Filter.eq('id_ref', int(selected_pivo))).first()
    area_geom = area_fc.geometry()
    bounds = ee.Geometry(area_geom).bounds().coordinates().getInfo()[0]
    minx = min(c[0] for c in bounds); maxx = max(c[0] for c in bounds)
    miny = min(c[1] for c in bounds); maxy = max(c[1] for c in bounds)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
except Exception:
    pass

# Tiles leves
all_outlines_mapid = get_all_pivots_outline_mapid()
folium.TileLayer(
    tiles=all_outlines_mapid['tile_fetcher'].url_format,
    attr='Pivôs (outline EE tile)',
    name='Pivôs (outline)',
    overlay=True,
    control=False
).add_to(m)

ndvi_mapid = get_ndvi_mapid_for(int(selected_pivo), last_date)
folium.TileLayer(
    tiles=ndvi_mapid['tile_fetcher'].url_format,
    attr='GEE NDVI',
    name='NDVI',
    overlay=True,
    control=False
).add_to(m)

# Polígono do selecionado
sel_geojson = get_selected_area_geojson(int(selected_pivo), simplify_m=2.0)
if sel_geojson and sel_geojson.get('features'):
    folium.GeoJson(
        sel_geojson,
        name='Área do Pivô (selecionado)',
        style_function=lambda x: {'color': '#2563eb', 'weight': 3, 'fillOpacity': 0}
    ).add_to(m)

tab1, tab2 = st.tabs(["🗺️ Mapa", "📈 Séries NDVI + Precip"])

# Render 1: sem pinos — captura estado do mapa do usuário
with tab1:
    map_state = st_folium(m, use_container_width=True, height=520, returned_objects=["last_active_drawing"])
    # guarda bounds/zoom/center para o próximo rerun
    if map_state and "bounds" in map_state and map_state["bounds"]:
        st.session_state["map_bounds"] = map_state["bounds"]
    if map_state and "zoom" in map_state and map_state["zoom"] is not None:
        st.session_state["map_zoom"] = map_state["zoom"]
    if map_state and "center" in map_state and map_state["center"]:
        st.session_state["map_center"] = (map_state["center"]["lat"], map_state["center"]["lng"])

# Passo B: se tiver bounds+zoom, refaz mapa incluindo SOMENTE pinos visíveis
bounds = st.session_state.get("map_bounds")
zoom = st.session_state.get("map_zoom") or 0
center = st.session_state.get("map_center")

if bounds and zoom >= 12:
    # filtra pontos por bounds
    df_pts = get_all_centroids_list()
    south, west = bounds["south"], bounds["west"]
    north, east = bounds["north"], bounds["east"]

    # trata dateline (simples): assume bbox normal
    vis = df_pts[(df_pts["lat"] >= south) & (df_pts["lat"] <= north) &
                 (df_pts["lon"] >= west)  & (df_pts["lon"] <= east)]

    # CAP por render (limita DOM) — prioriza pontos próximos ao centro
    MAX_PINS = 1500
    if not vis.empty and len(vis) > MAX_PINS and center:
        clat, clon = center
        # distância aproximada em graus (ajuste longitude pelo cos(lat))
        lon_scale = cos(radians(clat))
        d = (vis["lat"] - clat).abs() + (vis["lon"] - clon).abs() * lon_scale
        vis = vis.assign(_d=d).sort_values("_d").head(MAX_PINS).drop(columns=["_d"])

    # novo mapa com pinos visíveis
    m2 = folium.Map(prefer_canvas=True)
    try:
        m2.fit_bounds([[south, west], [north, east]])
    except Exception:
        pass

    folium.TileLayer(
        tiles=all_outlines_mapid['tile_fetcher'].url_format,
        attr='Pivôs (outline EE tile)',
        name='Pivôs (outline)',
        overlay=True,
        control=False
    ).add_to(m2)
    folium.TileLayer(
        tiles=ndvi_mapid['tile_fetcher'].url_format,
        attr='GEE NDVI',
        name='NDVI',
        overlay=True,
        control=False
    ).add_to(m2)
    if sel_geojson and sel_geojson.get('features'):
        folium.GeoJson(
            sel_geojson,
            name='Área do Pivô (selecionado)',
            style_function=lambda x: {'color': '#2563eb', 'weight': 3, 'fillOpacity': 0}
        ).add_to(m2)

    # cluster com chunk agressivo
    cluster = MarkerCluster(
        name="Pivôs",
        disableClusteringAtZoom=16,
        showCoverageOnHover=False,
        spiderfyOnMaxZoom=True,
        zoomToBoundsOnClick=True,
        chunkedLoading=True,
        chunkInterval=100,  # ms entre lotes
        chunkDelay=25,      # ms de descanso
        maxClusterRadius=60
    ).add_to(m2)

    icon_cache = {}
    for lat, lon, pid in vis[["lat", "lon", "pid"]].itertuples(index=False):
        if pid not in icon_cache:
            data_uri = svg_number_icon_data_uri(int(pid), size=26)
            icon_cache[pid] = CustomIcon(icon_image=data_uri, icon_size=(26, 26), icon_anchor=(13, 13))
        folium.Marker(location=[lat, lon], icon=icon_cache[pid]).add_to(cluster)

    # Render 2: com pinos do viewport
    with tab1:
        st_folium(m2, use_container_width=True, height=520)
        st.caption(f"Pinos numerados no viewport (zoom ≥ 12). Mostrando até {MAX_PINS} pinos mais próximos do centro.")

# =====================
# GRÁFICO (NDVI fixo 0.2–1)
# =====================
with tab2:
    if df.empty:
        st.warning("Sem dados para o período/área selecionados (NDVI/precipitação).")
    else:
        ndvi_scale = alt.Scale(domain=[0.2, 1])
        bars_precip = alt.Chart(df).mark_bar(color='#3b82f6', opacity=0.5).encode(
            x=alt.X('date:T', title='Data'),
            y=alt.Y('precip_mm:Q', title='Precipitação (mm/mês)', axis=alt.Axis(titleColor='#3b82f6')),
            tooltip=[alt.Tooltip('date:T', title='Data'),
                     alt.Tooltip('precip_mm:Q', title='Precipitação (mm)', format=".2f")]
        )
        line_green_full = alt.Chart(df).mark_line(color='green', strokeWidth=2).encode(
            x=alt.X('date:T', title='Data'),
            y=alt.Y('ndvi:Q', title='NDVI', scale=ndvi_scale,
                    axis=alt.Axis(format=".3f", orient='right', titleColor='green'))
        )

        def segments_below_threshold(local_df: pd.DataFrame, thr: float) -> pd.DataFrame:
            if local_df.empty:
                return pd.DataFrame(columns=['date', 'ndvi', 'seg'])
            local_df = local_df.sort_values('date').reset_index(drop=True)
            seg_rows, seg_id, in_seg, prev = [], 0, False, None
            for _, curr in local_df.iterrows():
                if prev is None:
                    if curr['ndvi'] <= thr:
                        seg_id += 1; in_seg = True
                        seg_rows.append({'date': curr['date'], 'ndvi': float(curr['ndvi']), 'seg': seg_id})
                    prev = curr; continue
                p_ndvi = float(prev['ndvi']); c_ndvi = float(curr['ndvi'])
                p_below = p_ndvi <= thr; c_below = c_ndvi <= thr
                if p_below and c_below:
                    if not in_seg: seg_id += 1; in_seg = True
                    seg_rows.append({'date': prev['date'], 'ndvi': p_ndvi, 'seg': seg_id})
                    seg_rows.append({'date': curr['date'], 'ndvi': c_ndvi, 'seg': seg_id})
                elif (not p_below) and c_below:
                    t1, t2 = prev['date'].value, curr['date'].value
                    alpha = 0.0 if c_ndvi == p_ndvi else (thr - p_ndvi) / (c_ndvi - p_ndvi)
                    alpha = max(0.0, min(1.0, alpha))
                    tc = pd.to_datetime(int(round(t1 + alpha * (t2 - t1))))
                    seg_id += 1; in_seg = True
                    seg_rows.append({'date': tc, 'ndvi': thr, 'seg': seg_id})
                    seg_rows.append({'date': curr['date'], 'ndvi': c_ndvi, 'seg': seg_id})
                elif p_below and (not c_below):
                    t1, t2 = prev['date'].value, curr['date'].value
                    alpha = 0.0 if c_ndvi == p_ndvi else (thr - p_ndvi) / (c_ndvi - p_ndvi)
                    alpha = max(0.0, min(1.0, alpha))
                    tc = pd.to_datetime(int(round(t1 + alpha * (t2 - t1))))
                    if not in_seg: seg_id += 1; in_seg = True
                    seg_rows.append({'date': prev['date'], 'ndvi': p_ndvi, 'seg': seg_id})
                    seg_rows.append({'date': tc, 'ndvi': thr, 'seg': seg_id})
                    in_seg = False
                else:
                    in_seg = False
                prev = curr
            if not seg_rows:
                return pd.DataFrame(columns=['date','ndvi','seg'])
            return pd.DataFrame(seg_rows, columns=['date','ndvi','seg']).drop_duplicates().sort_values('date')

        df_below = segments_below_threshold(df[['date','ndvi']].copy(), threshold)
        line_red_overlay = alt.Chart(df_below).mark_line(color='red', strokeWidth=3).encode(
            x='date:T', y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale), detail='seg:N'
        )
        points_green = alt.Chart(df[df['ndvi'] > threshold]).mark_point(color='green', filled=True, opacity=1).encode(
            x='date:T', y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale),
            tooltip=[alt.Tooltip('date:T', title='Data'), alt.Tooltip('ndvi:Q', title='NDVI', format=".3f")]
        )
        points_red = alt.Chart(df[df['ndvi'] <= threshold]).mark_point(color='red', filled=True, opacity=1).encode(
            x='date:T', y=alt.Y('ndvi:Q', axis=None, scale=ndvi_scale),
            tooltip=[alt.Tooltip('date:T', title='Data'), alt.Tooltip('ndvi:Q', title='NDVI', format=".3f")]
        )

        chart = alt.layer(bars_precip, line_green_full, line_red_overlay, points_green, points_red
        ).resolve_scale(y='independent'
        ).properties(title=f'NDVI (linha) x Precipitação (barras) - Pivô {selected_pivo}', width='container', height=360
        ).configure_axis(grid=True, gridOpacity=0.15, labelFontSize=11, titleFontSize=12
        ).configure_view(strokeWidth=0).configure_title(fontSize=14)
        st.altair_chart(chart, use_container_width=True)

# =====================
# EXPORT
# =====================
if not df.empty:
    csv = df[['date','ndvi','precip_mm']].copy()
    csv['date'] = csv['date'].dt.strftime('%Y-%m')
    csv_str = csv.to_csv(index=False)
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="ndvi_precip_{selected_pivo}.csv">📥 Baixar dados (NDVI + Precip) CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
