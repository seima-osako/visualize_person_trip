import numpy as np
import pandas as pd
import geopandas as gpd

import streamlit as st
from st_aggrid import AgGrid

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide")
st.write("## 第６回（平成30年）パーソントリップ調査")
st.write("[データ取得元](https://www.tokyo-pt.jp/data/01_01)")

gdf_car_possession = gpd.read_file("data/car_possession.geojson")
df_od = pd.read_csv("data/od_kzone.csv")

st.write("### 自動車保有率")
area = st.radio(
    "Please select area", ("全域", "東京都", "神奈川県", "埼玉県", "千葉県", "茨城県"), horizontal=True
)

if area == "全域":
    df_kzone = gdf_car_possession
else:
    df_kzone = gdf_car_possession[gdf_car_possession["都県名"] == area]


fig_1 = px.choropleth_mapbox(
    df_kzone,
    geojson=gpd.GeoSeries(df_kzone["geometry"]).__geo_interface__,
    locations=df_kzone.index,
    color="自動車保有率",
    color_continuous_scale="Jet",
    center={"lon": 139.76602, "lat": 35.67589},
    hover_data=[
        "都県名",
        "市区町村名",
        "kzone",
        "自動車保有数",
        "自動車非保有数",
        "不明数",
        "自動車保有率",
        "自動車非保有率",
        "不明率",
    ],
    mapbox_style="carto-positron",
    opacity=0.5,
    zoom=8,
    height=700,
    range_color=(0, 1),
)

st.plotly_chart(fig_1, use_container_width=True)


st.sidebar.write("## OD可視化")

st.sidebar.write("### 都道府県の選択")

prefecture = st.sidebar.selectbox(
    "Please select prefecture",
    ("東京都", "神奈川県", "埼玉県", "千葉県", "茨城県"),
)

tmp_df = gdf_car_possession[gdf_car_possession["都県名"] == prefecture][
    ["都県名", "市区町村名", "kzone", "geometry"]
].rename(columns={"kzone": "着地"})

st.sidebar.write("### 計画基本ゾーン※")

o_kzone = st.sidebar.selectbox(
    "Please select kzone",
    tmp_df["着地"].tolist(),
)
st.sidebar.write("小ゾーンを数個集めて構成し、広域における計画単位として、また地域としてのまとまりのある交通計画の単位となるゾーンレベル")

target_od = pd.merge(tmp_df, df_od[df_od["発地"] == o_kzone], on="着地")

st.sidebar.write("### 目的種類")
purpose = st.sidebar.selectbox(
    "Please select purpose",
    ("計", "自宅−勤務", "自宅−通学", "自宅−業務", "自宅−私事", "勤務・業務", "私事", "帰宅", "不明"),
)


st.sidebar.write("### 代表交通手段")
move_mode = st.sidebar.selectbox(
    "Please select move_mode",
    ("計", "鉄道", "バス", "自動車", "２輪車", "自転車", "徒歩", "その他", "不明"),
)


vis_od = target_od[target_od["目的種類"] == purpose]
vmax = vis_od[move_mode].quantile(0.95)
vmin = vis_od[move_mode].quantile(0.05)


st.write("### 目的種類別代表交通手段別OD")

bm = st.radio(
    "Please select basemap",
    ("carto-positron", "carto-darkmatter", "open-street-map"),
    horizontal=True,
)

fig_2 = px.choropleth_mapbox(
    vis_od,
    geojson=gpd.GeoSeries(vis_od["geometry"]).__geo_interface__,
    locations=vis_od.index,
    color=move_mode,
    color_continuous_scale="reds",
    center={"lon": 139.76602, "lat": 35.67589},
    hover_data=["発地", "着地", "鉄道", "バス", "自動車", "２輪車", "自転車", "徒歩", "その他", "不明", "計"],
    mapbox_style=bm,
    opacity=0.5,
    zoom=8,
    height=700,
    range_color=(int(vmin), int(vmax)),
)

st.plotly_chart(fig_2, use_container_width=True)
