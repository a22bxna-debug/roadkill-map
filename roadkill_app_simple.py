import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from itertools import product
from haversine import haversine, Unit
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- アプリケーションの基本設定 ---
st.set_page_config(layout="wide", page_title="ロードキル可視化マップ")

# --- 1. 設定項目 ---
CSV_PATH = '分析用_R5.4～R6.3路上障害物（ロードキル）.csv'
SECTIONS_SHP_PATH = 'N06-23_HighwaySection.shp'
IC_SHP_PATH = 'N06-23_Joint.shp'

# CSVの列名
CSV_ROUTE_NAME_COL = '道路名' 
CSV_SECTION_NAME_COL = '区間'
CSV_DIRECTION_COL = '上下'
CSV_WEATHER_COL = '排除時天候'
CSV_ANIMAL_COL = '小分類'

# シェープファイルの列名
IC_NAME_COL = 'N06_018'
ROUTE_NAME_COL = 'N06_007'

# 地図設定
MAP_CENTER = [38.5, 140.0]
MAP_ZOOM = 5.5
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

def get_offset_coords(start_lon, start_lat, end_lon, end_lat, direction):
    offset_dist = 0.0015
    angle = np.arctan2(end_lat - start_lat, end_lon - start_lon) + np.pi / 2
    offset_lon, offset_lat = offset_dist * np.cos(angle), offset_dist * np.sin(angle)
    direction_factor = 1
    if direction in ['下り', '西行き', '外回り', '外']: direction_factor = -1
    return {
        "start_lon": start_lon + offset_lon * direction_factor,
        "start_lat": start_lat + offset_lat * direction_factor,
        "end_lon": end_lon + offset_lon * direction_factor,
        "end_lat": end_lat + offset_lat * direction_factor,
    }

# ★★★ 新しいグラデーション色を生成する関数 ★★★
def get_color_from_gradient(value, max_value):
    # viridisカラースキームを使用 (0.0 -> 1.0 の値を想定)
    # 値が小さいほど青紫、大きいほど黄色になる
    norm = mcolors.Normalize(vmin=0, vmax=max_value)
    cmap = cm.get_cmap('viridis')
    # RGBAを0-255の整数リストに変換
    rgba = cmap(norm(value))
    return [int(c * 255) for c in rgba[:3]] + [200] # 透明度を200に設定

# --- 3. データ読み込みと前処理（キャッシュ） ---
@st.cache_data
def load_data():
    try:
        roadkill_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', header=2, dtype=str)
    except Exception as e:
        st.error(f"CSVファイルの読み込みに失敗: {e}"); st.stop()
    roadkill_df.columns = roadkill_df.columns.str.strip()
    required_cols = [CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL, CSV_DIRECTION_COL, CSV_WEATHER_COL, CSV_ANIMAL_COL]
    roadkill_df.dropna(subset=required_cols, inplace=True)
    
    ic_gdf = gpd.read_file(IC_SHP_PATH, encoding='utf-8').to_crs(epsg=4326)
    sections_gdf = gpd.read_file(SECTIONS_SHP_PATH, encoding='utf-8').to_crs(epsg=4326)
    ic_gdf['ic_name_norm'] = ic_gdf[IC_NAME_COL].apply(normalize_name)
    ic_locations = ic_gdf.groupby('ic_name_norm')['geometry'].apply(list).to_dict()
    sections_gdf['route_name_norm'] = sections_gdf[ROUTE_NAME_COL].apply(lambda x: normalize_name(x, is_route=True))
    route_geometries = sections_gdf.groupby('route_name_norm')['geometry'].apply(lambda geoms: geoms.unary_union).to_dict()
    return roadkill_df, ic_locations, route_geometries

# --- 4. メイン処理 ---
st.title("高速道路ロードキルデータ可視化マップ（IC区間別・方向別）")

try:
    roadkill_df, ic_locations, route_geometries = load_data()

    st.sidebar.header("表示フィルタ")
    weather_options = ['すべて'] + sorted(list(roadkill_df[CSV_WEATHER_COL].unique()))
    selected_weather = st.sidebar.selectbox("天候を選択", options=weather_options)
    animal_options = ['すべて'] + sorted(list(roadkill_df[CSV_ANIMAL_COL].unique()))
    selected_animal = st.sidebar.selectbox("動物の種類を選択", options=animal_options)

    filtered_df = roadkill_df.copy()
    if selected_weather != 'すべて':
        filtered_df = filtered_df[filtered_df[CSV_WEATHER_COL] == selected_weather]
    if selected_animal != 'すべて':
        filtered_df = filtered_df[filtered_df[CSV_ANIMAL_COL] == selected_animal]

    if not filtered_df.empty:
        section_counts = filtered_df.groupby([CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL, CSV_DIRECTION_COL]).size().reset_index(name='件数')
        split_sections = section_counts[CSV_SECTION_NAME_COL].str.split('〜', expand=True)
        section_counts['始点_norm'] = split_sections[0].apply(normalize_name)
        section_counts['終点_norm'] = split_sections[1].apply(normalize_name) if 1 in split_sections.columns else ""
        
        @st.cache_data
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
                    offset_coords = get_offset_coords(best_pair[0].x, best_pair[0].y, best_pair[1].x, best_pair[1].y, row[CSV_DIRECTION_COL])
                    map_data.append({"路線名": csv_route_name, "区間": row[CSV_SECTION_NAME_COL], "方向": row[CSV_DIRECTION_COL], "件数": row['件数'], **offset_coords})
            return pd.DataFrame(map_data)

        map_df = get_map_data(section_counts, ic_locations, route_geometries)

        if not map_df.empty:
            max_count = map_df['件数'].max()
            # ★★★ 件数に応じた色情報をデータフレームに列として追加 ★★★
            map_df['color'] = map_df['件数'].apply(lambda count: get_color_from_gradient(count, max_count))
            
            st.pydeck_chart(pdk.Deck(
                map_style="road",
                initial_view_state=pdk.ViewState(latitude=MAP_CENTER[0], longitude=MAP_CENTER[1], zoom=MAP_ZOOM, pitch=0, bearing=0),
                layers=[
                    pdk.Layer(
                        "LineLayer",
                        data=map_df,
                        get_source_position=["start_lon", "start_lat"],
                        get_target_position=["end_lon", "end_lat"],
                        # ★★★ 線の太さを固定値に変更 ★★★
                        get_width=15, # メートル単位の幅
                        width_min_pixels=1.5,
                        # ★★★ 色をデータフレームの'color'列から取得 ★★★
                        get_color='color',
                        pickable=True, auto_highlight=True,
                    ),
                ],
                tooltip={"html": "<b>路線名:</b> {路線名}<br/><b>区間:</b> {区間} ({方向})<br/><b>件数:</b> {件数}件",
                         "style": {"backgroundColor": "steelblue", "color": "white"}}
            ))
            st.write("線の色は、区間・方向ごとの発生件数が多いほど青→緑→黄色に変化します。")
            st.subheader("フィルタ適用後のデータ")
            st.dataframe(filtered_df)
        else:
            st.warning("フィルタ条件に一致し、地図上に表示できる区間がありませんでした。")
    else:
        st.warning("フィルタ条件に一致するロードキルデータがありません。")

except Exception as e:
    st.error(f"アプリケーションの実行中に予期せぬエラーが発生しました: {e}")