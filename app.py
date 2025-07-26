import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from itertools import product
from haversine import haversine, Unit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import io
import base64

# --- アプリケーションの基本設定 ---
st.set_page_config(layout="wide", page_title="R5年度ロードキルマップ")

# --- 1. 設定項目 ---
CSV_PATH = '上下区別なし_R5.4～R6.3路上障害物（ロードキル）.csv'
SECTIONS_SHP_PATH = 'N06-23_HighwaySection.shp'
IC_SHP_PATH = 'N06-23_Joint.shp'

# CSVの列名
CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL, CSV_DIRECTION_COL = '道路名', '区間', '上下'
CSV_WEATHER_COL, CSV_ANIMAL_COL = '排除時天候', '小分類'
CSV_MONTH_COL, CSV_HOUR_COL, CSV_DAY_OF_WEEK_COL = '月', '時', '曜'
CSV_LENGTH_COL = '区間長_km' 

# シェープファイルの列名
IC_NAME_COL, ROUTE_NAME_COL = 'N06_018', 'N06_007'

# 地図設定
MAP_CENTER, MAP_ZOOM = [38.5, 140.0], 5.5
DISTANCE_THRESHOLD_M = 3000

# --- 2. 補助関数 ---
def normalize_name(name, is_route=False):
    if not isinstance(name, str): return ""
    name = name.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
    name = name.replace('　', ' ')
    if is_route:
        if name.endswith('道'): name = name.replace('道', '')
        return name.strip()
    else:
        suffixes = ['ＩＣ', 'IC', 'ＪＣＴ', 'JCT', 'ＳＩＣ', 'SIC', 'ＳＡ', 'SA', 'ＰＡ', 'PA', 'ＴＢ', 'TB']
        for suffix in suffixes: name = name.replace(suffix, '')
        return name.strip()

def get_color_from_gradient(value, max_value):
    norm = mcolors.Normalize(vmin=0, vmax=max_value if max_value > 0 else 1)
    cmap = plt.get_cmap('coolwarm')
    rgba = cmap(norm(value))
    return [int(c * 255) for c in rgba[:3]] + [200]

def create_legend_image(max_value):
    try:
        plt.rcParams['font.family'] = 'MS Gothic'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(3, 0.35)) 
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=0, vmax=max_value if max_value > 0 else 1)
    
    cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('1kmあたりの件数', fontsize=8) 
    cb.ax.tick_params(labelsize=7)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    # ★★★ 凡例の位置を調整 (bottomの値を大きくする) ★★★
    legend_html = f'''
    <div style="
        position: absolute; 
        bottom: 50px;
        right: 20px;
        z-index: 1000;
        ">
        <div style="
            background-color: white;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #ccc;
            font-size: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
            <img src="data:image/png;base64,{img_str}" style="width: 100%;">
        </div>
    </div>
    '''
    return legend_html


# --- 3. データ読み込みと前処理（キャッシュ） ---
@st.cache_data
def load_data():
    try:
        roadkill_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', header=2)
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}"); st.stop()
    roadkill_df.columns = roadkill_df.columns.str.strip()
    
    required_cols = [CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL, CSV_DIRECTION_COL, CSV_WEATHER_COL, CSV_ANIMAL_COL, CSV_MONTH_COL, CSV_HOUR_COL, CSV_DAY_OF_WEEK_COL, CSV_LENGTH_COL]
    roadkill_df.dropna(subset=required_cols, inplace=True)
    
    numeric_cols = [CSV_MONTH_COL, CSV_HOUR_COL, CSV_LENGTH_COL]
    for col in numeric_cols:
        roadkill_df[col] = pd.to_numeric(roadkill_df[col], errors='coerce')
    roadkill_df.dropna(subset=numeric_cols, inplace=True)
    roadkill_df[CSV_MONTH_COL] = roadkill_df[CSV_MONTH_COL].astype(int)
    roadkill_df[CSV_HOUR_COL] = roadkill_df[CSV_HOUR_COL].astype(int)

    ic_gdf = gpd.read_file(IC_SHP_PATH, encoding='utf-8').to_crs(epsg=4326)
    sections_gdf = gpd.read_file(SECTIONS_SHP_PATH, encoding='utf-8').to_crs(epsg=4326)
    ic_gdf['ic_name_norm'] = ic_gdf[IC_NAME_COL].apply(normalize_name)
    ic_locations = ic_gdf.groupby('ic_name_norm')['geometry'].apply(list).to_dict()
    sections_gdf['route_name_norm'] = sections_gdf[ROUTE_NAME_COL].apply(lambda x: normalize_name(x, is_route=True))
    route_geometries = sections_gdf.groupby('route_name_norm')['geometry'].apply(lambda geoms: gpd.GeoSeries(geoms).union_all()).to_dict()
    return roadkill_df, ic_locations, route_geometries

# 地図データ作成関数
def get_map_data(_section_counts, _ic_locations, _route_geometries):
    csv_routes_raw = _section_counts[CSV_ROUTE_NAME_COL].unique()
    shp_routes_raw = list(_route_geometries.keys())
    route_name_map = {}
    for csv_name in csv_routes_raw:
        norm_csv_name = normalize_name(csv_name, is_route=True)
        for shp_name in shp_routes_raw:
            if norm_csv_name in shp_name: route_name_map[csv_name] = shp_name; break
    
    map_data = []
    for _, row in _section_counts.iterrows():
        csv_route_name, start_name, end_name = row[CSV_ROUTE_NAME_COL], row['始点_norm'], row['終点_norm']
        official_route_name = route_name_map.get(csv_route_name)
        if not official_route_name: continue
        target_route_geom = _route_geometries.get(official_route_name)
        if not target_route_geom: continue
        start_candidates, end_candidates = _ic_locations.get(start_name, []), _ic_locations.get(end_name, [])
        if not start_candidates or not end_candidates: continue
        valid_starts = [p for p in start_candidates if p.distance(target_route_geom) * 111000 < DISTANCE_THRESHOLD_M]
        valid_ends = [p for p in end_candidates if p.distance(target_route_geom) * 111000 < DISTANCE_THRESHOLD_M]
        if not valid_starts or not valid_ends: continue
        
        min_dist, best_pair = float('inf'), None
        for start_point, end_point in product(valid_starts, valid_ends):
            dist = haversine((start_point.y, start_point.x), (end_point.y, end_point.x))
            if dist > 0 and dist < min_dist: min_dist, best_pair = dist, (start_point, end_point)
        
        if best_pair:
            map_data.append({"路線名": csv_route_name, "区間": row[CSV_SECTION_NAME_COL], 
                             "件数": row['件数'], "区間長_km": row[CSV_LENGTH_COL],
                             "start_lon": best_pair[0].x, "start_lat": best_pair[0].y,
                             "end_lon": best_pair[1].x, "end_lat": best_pair[1].y})
    return pd.DataFrame(map_data)

# --- 4. メイン処理 ---
st.title("R5年度ロードキルマップ（合計版）")

try:
    roadkill_df, ic_locations, route_geometries = load_data()

    st.sidebar.header("表示フィルタ")
    filter_mode = st.sidebar.radio("フィルタの選択方法", ('単一選択', '複数選択'), horizontal=True)
    st.sidebar.markdown("---") 

    month_options = sorted(roadkill_df[CSV_MONTH_COL].unique())
    hour_options = sorted(roadkill_df[CSV_HOUR_COL].unique())
    day_of_week_options = ['月', '火', '水', '木', '金', '土', '日']
    weather_options = sorted([w for w in roadkill_df[CSV_WEATHER_COL].unique() if pd.notna(w)])
    animal_options = sorted([a for a in roadkill_df[CSV_ANIMAL_COL].unique() if pd.notna(a)])

    filtered_df = roadkill_df.copy()
    if filter_mode == '単一選択':
        selected_month = st.sidebar.selectbox("月を選択", options=['すべて'] + month_options)
        selected_hour = st.sidebar.selectbox("時間帯を選択", options=['すべて'] + hour_options)
        selected_day_of_week = st.sidebar.selectbox("曜日を選択", options=['すべて'] + day_of_week_options)
        selected_weather = st.sidebar.selectbox("天候を選択", options=['すべて'] + weather_options)
        selected_animal = st.sidebar.selectbox("動物の種類を選択", options=['すべて'] + animal_options)
        if selected_month != 'すべて': filtered_df = filtered_df[filtered_df[CSV_MONTH_COL] == selected_month]
        if selected_hour != 'すべて': filtered_df = filtered_df[filtered_df[CSV_HOUR_COL] == selected_hour]
        if selected_day_of_week != 'すべて': filtered_df = filtered_df[filtered_df[CSV_DAY_OF_WEEK_COL] == selected_day_of_week]
        if selected_weather != 'すべて': filtered_df = filtered_df[filtered_df[CSV_WEATHER_COL] == selected_weather]
        if selected_animal != 'すべて': filtered_df = filtered_df[filtered_df[CSV_ANIMAL_COL] == selected_animal]
    else:
        selected_months = st.sidebar.multiselect("月を選択", options=month_options, default=month_options)
        selected_hours = st.sidebar.multiselect("時間帯を選択", options=hour_options, default=hour_options)
        selected_days_of_week = st.sidebar.multiselect("曜日を選択", options=day_of_week_options, default=day_of_week_options)
        selected_weathers = st.sidebar.multiselect("天候を選択", options=weather_options, default=weather_options)
        selected_animals = st.sidebar.multiselect("動物の種類を選択", options=animal_options, default=animal_options)
        if selected_months: filtered_df = filtered_df[filtered_df[CSV_MONTH_COL].isin(selected_months)]
        if selected_hours: filtered_df = filtered_df[filtered_df[CSV_HOUR_COL].isin(selected_hours)]
        if selected_days_of_week: filtered_df = filtered_df[filtered_df[CSV_DAY_OF_WEEK_COL].isin(selected_days_of_week)]
        if selected_weathers: filtered_df = filtered_df[filtered_df[CSV_WEATHER_COL].isin(selected_weathers)]
        if selected_animals: filtered_df = filtered_df[filtered_df[CSV_ANIMAL_COL].isin(selected_animals)]
        
    map_container = st.container()

    if not filtered_df.empty:
        key_cols = [CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL]
        agg_funcs = {'件数': ('区間', 'size'), CSV_LENGTH_COL: (CSV_LENGTH_COL, 'first')}
        section_counts = filtered_df.groupby(key_cols).agg(**agg_funcs).reset_index()
        
        split_sections = section_counts[CSV_SECTION_NAME_COL].str.split('〜', expand=True)
        section_counts['始点_norm'] = split_sections[0].apply(normalize_name)
        section_counts['終点_norm'] = split_sections[1].apply(normalize_name) if 1 in split_sections.columns else ""
        
        map_df = get_map_data(section_counts, ic_locations, route_geometries)

        with map_container:
            if not map_df.empty:
                map_df['件数_per_km'] = map_df.apply(lambda row: row['件数'] / row['区間長_km'] if row['区間長_km'] > 0 else 0, axis=1)
                max_density = map_df['件数_per_km'].max()
                map_df['color'] = map_df['件数_per_km'].apply(lambda density: get_color_from_gradient(density, max_density))
                
                map_df['tooltip'] = map_df.apply(
                    lambda row: f"<b>路線名:</b> {row['路線名']}<br/>"
                                f"<b>区間:</b> {row['区間']}<br/>"
                                f"<b>1kmあたり件数:</b> {row['件数_per_km']:.2f} 件/km<br/>"
                                f"<b>合計件数:</b> {row['件数']} 件<br/>"
                                f"<b>区間長:</b> {row['区間長_km']:.1f} km",
                    axis=1
                )
                
                st.pydeck_chart(pdk.Deck(
                    map_style="road",
                    initial_view_state=pdk.ViewState(latitude=MAP_CENTER[0], longitude=MAP_CENTER[1], zoom=MAP_ZOOM, pitch=0),
                    layers=[
                        pdk.Layer("LineLayer", data=map_df, get_source_position=["start_lon", "start_lat"],
                                  get_target_position=["end_lon", "end_lat"], get_width=15, 
                                  width_min_pixels=1.5, get_color='color', pickable=True, auto_highlight=True),
                    ],
                    tooltip={"html": "{tooltip}"}
                ))
                
                legend_html = create_legend_image(max_density)
                st.markdown(legend_html, unsafe_allow_html=True)
                
            else:
                st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=MAP_CENTER[0], longitude=MAP_CENTER[1], zoom=MAP_ZOOM, pitch=0)))
                st.warning("フィルタ条件に一致し、地図上に表示できる区間がありませんでした。")
    
        st.subheader("フィルタ適用後のデータ")
        st.dataframe(filtered_df)
        
    else:
        # フィルタに一致するデータがない場合でも、凡例は表示したい
        with map_container:
            st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=MAP_CENTER[0], longitude=MAP_CENTER[1], zoom=MAP_ZOOM, pitch=0)))
            # 全データの最大値で凡例を作成
            all_counts = roadkill_df.groupby([CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL]).agg(件数=('区間', 'size'), 区間長_km=(CSV_LENGTH_COL, 'first')).reset_index()
            all_counts['件数_per_km'] = all_counts.apply(lambda row: row['件数'] / row['区間長_km'] if row['区間長_km'] > 0 else 0, axis=1)
            overall_max_density = all_counts['件数_per_km'].max()
            legend_html = create_legend_image(overall_max_density)
            st.markdown(legend_html, unsafe_allow_html=True)

        st.warning("フィルタ条件に一致するロードキルデータがありません。")

except Exception as e:
    st.error(f"アプリケーションの実行中に予期せぬエラーが発生しました: {e}")