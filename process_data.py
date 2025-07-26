import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import warnings

# UserWarningを非表示にする
warnings.filterwarnings('ignore', category=UserWarning)

print("処理を開始します...")

# --- 1. 設定項目 ---
HIGHWAY_SHP_PATH = 'N06-23_HighwaySection.shp'
IC_SHP_PATH = '高速道路ICポイントデータ.shp'
OUTPUT_SHP_PATH = 'final_highway_sections_with_ic.shp' # 新しい出力ファイル名

# 列名
IC_NAME_COL = 'N06_018'
ROUTE_NAME_COL = 'N06_007'

# --- 2. データの読み込み ---
try:
    highways_gdf = gpd.read_file(HIGHWAY_SHP_PATH, encoding='shift-jis')
    ic_gdf = gpd.read_file(IC_SHP_PATH, encoding='utf-8')
    print("シェープファイルの読み込みが完了しました。")
except Exception as e:
    print(f"エラー: ファイルの読み込みに失敗しました。{e}")
    exit()

# CRS（座標参照系）を統一
if highways_gdf.crs != ic_gdf.crs:
    print("CRSを統一しています...")
    ic_gdf = ic_gdf.to_crs(highways_gdf.crs)

# ★★★ ここからが修正部分 ★★★
def add_ic_suffix(name):
    """IC名に適切な語尾を付ける関数"""
    if not isinstance(name, str):
        return name
    # すでに語尾がついているかチェック
    suffixes = ['JCT', 'SIC', 'SA', 'PA', 'TB', 'IC']
    if any(name.endswith(s) for s in suffixes):
        return name
    # 語尾がなければ「IC」を付ける
    return name + 'IC'

# ★★★ 修正部分ここまで ★★★

# --- 3. メイン処理 ---
final_segments_data = []
ic_points_unary = ic_gdf.unary_union

for index, highway in highways_gdf.iterrows():
    line = highway.geometry
    if not isinstance(line, LineString):
        continue
        
    original_attrs = highway.drop('geometry').to_dict()

    possible_ic_gdf = ic_gdf[ic_gdf.intersects(line.buffer(0.01))]
    if possible_ic_gdf.empty:
        continue

    cut_distances = {0.0, line.length}
    for ic_point in possible_ic_gdf.geometry:
        distance = line.project(ic_point)
        cut_distances.add(distance)
    
    sorted_distances = sorted(list(cut_distances))

    for i in range(len(sorted_distances) - 1):
        start_dist = sorted_distances[i]
        end_dist = sorted_distances[i+1]

        if abs(start_dist - end_dist) < 1e-6:
            continue

        start_point_on_line = line.interpolate(start_dist)
        end_point_on_line = line.interpolate(end_dist)

        new_segment_coords = []
        for coord in line.coords:
            p = Point(coord)
            dist_on_line = line.project(p)
            if start_dist <= dist_on_line <= end_dist:
                new_segment_coords.append(coord)
        
        if not new_segment_coords or Point(new_segment_coords[0]) != start_point_on_line:
            new_segment_coords.insert(0, start_point_on_line.coords[0])
        if not new_segment_coords or Point(new_segment_coords[-1]) != end_point_on_line:
            new_segment_coords.append(end_point_on_line.coords[0])
            
        if len(new_segment_coords) < 2:
            continue
        
        new_segment = LineString(new_segment_coords)

        start_ic_info = ic_gdf[ic_gdf.geometry == nearest_points(start_point_on_line, ic_points_unary)[1]]
        end_ic_info = ic_gdf[ic_gdf.geometry == nearest_points(end_point_on_line, ic_points_unary)[1]]

        start_ic_name = start_ic_info[IC_NAME_COL].iloc[0] if not start_ic_info.empty else "N/A"
        end_ic_name = end_ic_info[IC_NAME_COL].iloc[0] if not end_ic_info.empty else "N/A"

        attrs = original_attrs.copy()
        # ★★★ ここからが修正部分 ★★★
        attrs['start_IC'] = add_ic_suffix(start_ic_name)
        attrs['end_IC'] = add_ic_suffix(end_ic_name)
        # ★★★ 修正部分ここまで ★★★
        attrs['geometry'] = new_segment
        final_segments_data.append(attrs)

    route_name_display = original_attrs.get(ROUTE_NAME_COL, "路線名なし")
    print(f"路線: {route_name_display} の処理完了")

# --- 4. 結果の保存 ---
if final_segments_data:
    final_gdf = gpd.GeoDataFrame(final_segments_data, crs=highways_gdf.crs)
    final_gdf.to_file(OUTPUT_SHP_PATH, encoding='utf-8')
    print(f"\n処理が完了しました。結果は {OUTPUT_SHP_PATH} に保存されました。")
else:
    print("\n処理対象のセクションが見つかりませんでした。")