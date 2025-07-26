import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import io
import base64
from geopy.geocoders import Nominatim

# --- アプリケーションの基本設定 ---
st.set_page_config(layout="wide", page_title="R5年度ロードキルマップ")

# --- 1. 設定項目 ---
CSV_PATH = '上下区別なし_R5.4～R6.3路上障害物（ロードキル）.csv'
SECTIONS_SHP_PATH = 'final_highway_sections_with_ic.shp' 

# CSVの列名
# ★★★ 指示通りに変更 ★★★
CSV_OFFICIAL_NAME_COL = '正式名称' 
CSV_ROUTE_NAME_COL = '道路名' # ツールチップ表示用の通称を追加
CSV_SECTION_NAME_COL = '区間'
CSV_DIRECTION_COL = '上下'
CSV_WEATHER_COL = '排除時天候'
CSV_ANIMAL_COL = '小分類'
CSV_MONTH_COL = '月'
CSV_HOUR_COL = '時'
CSV_DAY_OF_WEEK_COL = '曜'
CSV_LENGTH_COL = '区間長_km' 

# シェープファイルの列名
SHP_START_IC_COL = 'start_IC' 
SHP_END_IC_COL = 'end_IC'

# 地図設定
INITIAL_VIEW_STATE = pdk.ViewState(
    latitude=38.5, 
    longitude=140.0, 
    zoom=5.5, 
    pitch=0
)

# --- 2. 補助関数 ---
def normalize_name(name):
    if not isinstance(name, str): return ""
    name = name.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
    name = name.replace('　', ' ')
    return name.strip()

def get_color(value, max_value):
    if value <= 0: return [200, 200, 200, 40] 
    norm = mcolors.Normalize(vmin=0, vmax=max_value if max_value > 0 else 1)
    cmap = cm.get_cmap('coolwarm')
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
    legend_html = f'''
    <div style="position: absolute; bottom: 50px; right: 20px; z-index: 1000;">
        <div style="background-color: white; border-radius: 8px; padding: 8px; border: 1px solid #ccc; font-size: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{img_str}" style="width: 100%;">
        </div>
    </div>'''
    return legend_html

# --- 3. データ読み込みと前処理（キャッシュ） ---
@st.cache_data
def load_data():
    try:
        roadkill_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', header=2)
        sections_gdf = gpd.read_file(SECTIONS_SHP_PATH, encoding='utf-8').to_crs(epsg=4326)
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}"); st.stop()
    roadkill_df.columns = roadkill_df.columns.str.strip()
    # ★★★ 指示通りに変更 ★★★
    required_cols = [CSV_OFFICIAL_NAME_COL, CSV_ROUTE_NAME_COL, CSV_SECTION_NAME_COL, CSV_DIRECTION_COL, CSV_LENGTH_COL]
    roadkill_df.dropna(subset=required_cols, inplace=True)
    numeric_cols = [CSV_MONTH_COL, CSV_HOUR_COL, CSV_LENGTH_COL]
    for col in numeric_cols:
        roadkill_df[col] = pd.to_numeric(roadkill_df[col], errors='coerce')
    roadkill_df.dropna(subset=numeric_cols, inplace=True)
    roadkill_df[CSV_MONTH_COL] = roadkill_df[CSV_MONTH_COL].astype(int)
    roadkill_df[CSV_HOUR_COL] = roadkill_df[CSV_HOUR_COL].astype(int)
    roadkill_df['section_norm'] = roadkill_df[CSV_SECTION_NAME_COL].apply(normalize_name)
    sections_gdf['start_norm'] = sections_gdf[SHP_START_IC_COL].astype(str).apply(normalize_name)
    sections_gdf['end_norm'] = sections_gdf[SHP_END_IC_COL].astype(str).apply(normalize_name)
    sections_gdf['section_norm_key_1'] = sections_gdf['start_norm'] + '〜' + sections_gdf['end_norm']
    sections_gdf['section_norm_key_2'] = sections_gdf['end_norm'] + '〜' + sections_gdf['start_norm']
    return roadkill_df, sections_gdf

# --- 4. メイン処理 ---
st.title("R5年度ロードキルマップ（合計版）")

try:
    roadkill_df, sections_gdf = load_data()
    
    def reset_all_states():
        st.session_state.view_state = INITIAL_VIEW_STATE
        if "location_query_input" in st.session_state:
            st.session_state.location_query_input = "" 
        if "data_selector" in st.session_state:
            st.session_state.data_selector = {"selection": {"rows": []}}

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

    # --- 集計とデータ結合 ---
    all_sections_map = sections_gdf.copy()
    if not filtered_df.empty:
        # ★★★ 指示通りに変更 ★★★
        agg_funcs = {
            '件数': (CSV_SECTION_NAME_COL, 'size'), 
            '区間長_km': (CSV_LENGTH_COL, 'first'),
            CSV_ROUTE_NAME_COL: (CSV_ROUTE_NAME_COL, 'first') # 表示用の通称を保持
        }
        section_counts = filtered_df.groupby('section_norm').agg(**agg_funcs).reset_index()
    else:
        section_counts = pd.DataFrame(columns=['section_norm', '件数', '区間長_km', CSV_ROUTE_NAME_COL])
        
    map_gdf_1 = pd.merge(all_sections_map, section_counts, left_on='section_norm_key_1', right_on='section_norm', how='left')
    map_gdf_2 = pd.merge(all_sections_map, section_counts, left_on='section_norm_key_2', right_on='section_norm', how='left')
    map_gdf = map_gdf_1
    map_gdf['件数'] = map_gdf_1['件数'].fillna(map_gdf_2['件数']).fillna(0)
    map_gdf['区間長_km'] = map_gdf_1['区間長_km'].fillna(map_gdf_2['区間長_km'])
    map_gdf[CSV_ROUTE_NAME_COL] = map_gdf_1[CSV_ROUTE_NAME_COL].fillna(map_gdf_2[CSV_ROUTE_NAME_COL]) # 通称も結合
    map_gdf.dropna(subset=['geometry'], inplace=True)

    # --- 地図連携UI ---
    if "view_state" not in st.session_state:
        st.session_state.view_state = INITIAL_VIEW_STATE
    
    st.sidebar.markdown("---")
    st.sidebar.header("地図移動")
    
    location_query = st.sidebar.text_input("地名を入力して検索", key="location_query_input")
    if st.sidebar.button("検索"):
        if location_query:
            geolocator = Nominatim(user_agent="roadkill_mapper_app")
            try:
                location = geolocator.geocode(location_query, country_codes="jp")
                if location:
                    st.session_state.view_state = pdk.ViewState(
                        latitude=location.latitude, longitude=location.longitude,
                        zoom=10, pitch=0, transition_duration=1000,
                    )
                    st.rerun()
                else:
                    st.sidebar.error("場所が見つかりませんでした。")
            except Exception as e:
                st.sidebar.error("位置情報の取得中にエラーが発生しました。")
        else:
            st.sidebar.warning("地名を入力してください。")

    st.sidebar.button("地図表示をリセット", on_click=reset_all_states)

    map_container = st.container()

    if not map_gdf.empty:
        map_gdf['件数_per_km'] = map_gdf.apply(lambda row: row['件数'] / row['区間長_km'] if pd.notna(row['区間長_km']) and row['区間長_km'] > 0 else 0, axis=1)
        max_density = map_gdf['件数_per_km'].max()
        map_gdf['color'] = map_gdf['件数_per_km'].apply(lambda d: get_color(d, max_density))
        map_gdf['tooltip'] = map_gdf.apply(
            # ★★★ 指示通りに変更 ★★★
            lambda row: f"<b>路線名:</b> {row[CSV_ROUTE_NAME_COL]}<br/>"
                        f"<b>区間:</b> {row[SHP_START_IC_COL]}〜{row[SHP_END_IC_COL]}<br/>"
                        f"<b>1kmあたり件数:</b> {row['件数_per_km']:.2f} 件/km<br/>"
                        f"<b>合計件数:</b> {int(row['件数'])} 件<br/>"
                        f"<b>区間長:</b> {row['区間長_km']:.1f} km" if pd.notna(row['区間長_km']) else f"<b>路線名:</b> {row[CSV_ROUTE_NAME_COL]}<br/><b>区間:</b> {row[SHP_START_IC_COL]}〜{row[SHP_END_IC_COL]}<br/>件数: {int(row['件数'])} 件",
            axis=1
        )
        
        with map_container:
            st.pydeck_chart(pdk.Deck(
                map_style="road",
                initial_view_state=st.session_state.view_state,
                layers=[
                    pdk.Layer("GeoJsonLayer", data=map_gdf, get_line_color='color',
                              get_line_width=45, line_width_min_pixels=6,
                              pickable=True, auto_highlight=True),
                ],
                tooltip={"html": "{tooltip}"}
            ))
            legend_html = create_legend_image(max_density)
            st.markdown(legend_html, unsafe_allow_html=True)
            
        st.subheader("区間別データ（クリックで地図移動）")
        display_df = map_gdf[map_gdf['件数'] > 0].sort_values(by='件数_per_km', ascending=False)
        display_df_view = display_df[['件数_per_km', '件数', '区間長_km', SHP_START_IC_COL, SHP_END_IC_COL]]
        
        st.dataframe(
            display_df_view,
            key="data_selector",
            on_select="rerun",
            selection_mode="single-row"
        )

        if "data_selector" in st.session_state and st.session_state.data_selector['selection']['rows']:
            selected_index = st.session_state.data_selector['selection']['rows'][0]
            selected_row = display_df.iloc[selected_index]
            centroid = selected_row.geometry.centroid
            
            st.session_state.view_state = pdk.ViewState(
                latitude=centroid.y, longitude=centroid.x,
                zoom=10, pitch=0, transition_duration=1000,
            )
            st.session_state.data_selector['selection']['rows'] = []
            st.rerun()
        
    else:
        st.warning("地図データの読み込み、またはロードキルデータとの結合に失敗しました。")

except Exception as e:
    st.error(f"アプリケーションの実行中に予期せぬエラーが発生しました: {e}")