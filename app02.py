import numpy as np
from dfply import *
import pandas as pd
import geopandas as gpd

from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

st.set_page_config(layout="wide")
#st.write("# 簡易Sim")

df_distribution = pd.read_csv("data/co2_distribution.csv")


prefecture = st.sidebar.selectbox(
    "Please select prefecture",
    ( "神奈川県", "東京都", "埼玉県", "千葉県", "茨城県"),
)

tmp_df = df_distribution[df_distribution["都道府県"] == prefecture]

city = st.sidebar.selectbox(
    "Please select city",
    tmp_df["市区町村"].tolist(),
)

target_distribution = tmp_df[tmp_df["市区町村"] == city]


st.sidebar.write("### 削減目標")
reduction_goal = st.sidebar.slider("Please select reduction rate(%)", 0, 100, 10, 1)

# =======================================================全国按分法=======================================================

st.sidebar.write("### 登録自動車台数の削減")
reduction_rate_car_1 = st.sidebar.slider("Automobile reduction rate(%)", 0, 100, 10, 1)

for r in target_distribution.itertuples():
    co2_2013 = r.CO2_2013
    co2_2018 = r.CO2_2018
    co2_total_2018 = r.CO2_全国_2018
    possession_car_total_2018 = r.保有台数_全国_2018
    possession_car_city_2018 = r.保有台数_2018

possession_car_city_goal = int(
    possession_car_city_2018 * ((100 - reduction_rate_car_1) / 100)
)
co2_goal = int(co2_2013*(100-reduction_goal)/100)
co2_reduction_1 = int(
    co2_total_2018 * possession_car_city_goal / possession_car_total_2018 * 3.67
)
co2_reduction_effect_1 = co2_2013 - co2_reduction_1

df_show_1 = pd.DataFrame(
    data={
        "col": ["2013年(B_d)", "2018年(P_d)", "2030年(B_d–削減目標)", "2018年 施策結果(P'_d)"],
        "val": [co2_2013, co2_2018, co2_goal, co2_reduction_1],
        "color": ["lightblue", "RoyalBlue", "SeaGreen", "orange"],
    }
)
fig_1 = go.Figure()
for r_1 in df_show_1.itertuples():
    fig_1.add_trace(
        go.Bar(
            x=[r_1.col],
            y=[r_1.val],
            marker_color=r_1.color,
            showlegend=False,
        )
    )

fig_1.update_layout(
    template="simple_white",
    xaxis=dict(tickfont=dict(size=15)),
    yaxis=dict(title="1,000 t-CO2eq", tickfont=dict(size=20)),
)

col1, col2 = st.columns(2)

with col1:
    st.info("## 全国按分法")
    possession_car_city_2018 = "{:,}".format(possession_car_city_2018)
    possession_car_total_2018 = "{:,}".format(possession_car_total_2018)
    co2_reduction_effect_1 = "{:,}".format(co2_reduction_effect_1 * 1000)
    st.markdown(
        f"""
        #### 登録自動車台数(2018年)
        ### {possession_car_city_2018} / {possession_car_total_2018} 台
        
        #### 削減効果（B_d – P'_d）
    """
    )
    st.success(f"### {co2_reduction_effect_1} t-CO2eq")
    st.markdown(
        f"""
        ```
        ＜推計式＞
        市区町村のCO2排出量
        ＝全国の自動車・バス炭素排出量 / 全国の自動車・バス保有台数 × 市区町村の自動車・バス保有台数 × 排出係数
        ※保有台数に比例すると仮定し、全国の保有台数あたり炭素排出量に対して、市区町村の保有台数を乗ずる。
        ```
    """
    )
with col2:
    st.plotly_chart(fig_1, use_container_width=True)


# =======================================================燃費法=======================================================
df_od = pd.read_csv("data/od_kzone_aggregation.csv")

st.sidebar.write("### 燃費改善（エコドライブ・渋滞解消）")
car_fuel_economy = st.sidebar.number_input("自動車", value=10.3)
bus_fuel_economy = st.sidebar.number_input("バス", value=4.96)
image = Image.open("data/fuel_table.png")
st.sidebar.image(image, caption="【参考】燃費テーブル")

st.sidebar.write("### モーダルシフト")
modal_shift = st.sidebar.slider("Transition rate from car to bus(%)", -100, 100, 10, 1)
st.sidebar.write("### EV化")
reduction_rate_car_2 = st.sidebar.slider("EV conversion rate(%)", 0, 100, 10, 1)

df_od["バス_new_燃費"] = bus_fuel_economy
df_od["自動車_new_燃費"] = car_fuel_economy
df_1 = df_od[(df_od["発地_都県名"] == prefecture) & (df_od["発地_市区町村名"].str.contains(city))]
df_2 = df_od[(df_od["着地_都県名"] == prefecture) & (df_od["着地_市区町村名"].str.contains(city))]

df = (
    pd.concat(
        [
            df_1[
                [
                    "発地",
                    "着地",
                    "バス",
                    "自動車",
                    "distance",
                    "バス_燃費",
                    "自動車_燃費",
                    "バス_new_燃費",
                    "自動車_new_燃費",
                ]
            ],
            df_2[
                [
                    "発地",
                    "着地",
                    "バス",
                    "自動車",
                    "distance",
                    "バス_燃費",
                    "自動車_燃費",
                    "バス_new_燃費",
                    "自動車_new_燃費",
                ]
            ],
        ]
    )
    >> distinct()
)


def calc_co2(x, mode=None):
    if x["発地"] == x["着地"]:
        frac = 0.5
    else:
        frac = 1

    return (x["distance"] * frac / x[f"{mode}_燃費"]) * (1 / 1000) * 2.322 * x[mode]


def calc_co2_new(x, mode=None):
    if x["発地"] == x["着地"]:
        frac = 0.5
    else:
        frac = 1

    return (x["distance"] * frac / x[f"{mode}_燃費"]) * (1 / 1000) * 2.322 * x[mode]


df["バス_CO2_daily"] = df.apply(lambda x: calc_co2(x, mode="バス"), axis=1)
df["自動車_CO2_daily"] = df.apply(lambda x: calc_co2(x, mode="自動車"), axis=1)

df["modal_shift"] = modal_shift
if modal_shift >=0:
    df['自動車_middle'] = (df['自動車']*((100-df['modal_shift'])/100)).astype(int)
    df['バス_new'] = df['バス']
else:
    df['バス_new'] = (df['バス']*((100-df['modal_shift']*(-1))/100)).astype(int)
    df['自動車_middle'] = (df['自動車'] + (df['バス']-df['バス_new'])).astype(int)

df["rate"] = reduction_rate_car_2
df["自動車_new"] = (df["自動車_middle"] * ((100 - df["rate"]) / 100)).astype(int)

df["バス_new_CO2_daily"] = df.apply(lambda x: calc_co2_new(x, mode="バス_new"), axis=1)
df["自動車_new_CO2_daily"] = df.apply(lambda x: calc_co2_new(x, mode="自動車_new"), axis=1)

df["旅客_CO2_yearly"] = ((df["バス_CO2_daily"] + df["自動車_CO2_daily"]) * 365).astype(int)
df["旅客_new_CO2_yearly"] = (
    (df["バス_new_CO2_daily"] + df["自動車_new_CO2_daily"]) * 365
).astype(int)


bus_trip = df["バス"].sum()
car_trip = df["自動車"].sum()
co2_2018_trip = int(df["旅客_CO2_yearly"].sum() / 1000)
co2_2013_trip = int(co2_2018_trip*co2_2013/co2_2018)
co2_goal_trip = int(co2_2013_trip*(100-reduction_goal)/100)

bus_new_trip = df["バス_new"].sum()
car_new_trip = df["自動車_new"].sum()
co2_reduction_2 = int(df["旅客_new_CO2_yearly"].sum() / 1000)

co2_reduction_effect_2 = co2_2013 - co2_reduction_2

df_show_2 = pd.DataFrame(
    data={
        "col": ["2013年(B_f)", "2018年(P_f)", "2030年(B_f–削減目標)", "2018年 施策結果(P'_f)"],
        "val": [co2_2013_trip,co2_2018_trip,co2_goal_trip,co2_reduction_2],
        "color": ["lightblue", "RoyalBlue", "SeaGreen", "orange"],
    }
)
fig_2 = go.Figure()
for r_2 in df_show_2.itertuples():
    fig_2.add_trace(
        go.Bar(x=[r_2.col], y=[r_2.val], marker_color=r_2.color, showlegend=False)
    )

fig_2.update_layout(
    template="simple_white",
    xaxis=dict(tickfont=dict(size=15)),
    yaxis=dict(title="1,000 t-CO2eq", tickfont=dict(size=20)),
)

col3, col4 = st.columns(2)

with col3:
    st.warning("## 燃費法")
    bus_trip = "{:,}".format(bus_trip)
    car_trip = "{:,}".format(car_trip)
    bus_new_trip = "{:,}".format(bus_new_trip)
    car_new_trip = "{:,}".format(car_new_trip)
    co2_reduction_effect_2 = "{:,}".format(co2_reduction_effect_2 * 1000)
    st.markdown(
        f"""
        #### トリップ数
        #### 自動車：バス＝{car_trip} ： {bus_trip}
        　　　　　　　　　**↓ 自動車⇄バス間のシフト or EV化**
        #### 自動車：バス＝{car_new_trip} ： {bus_new_trip}
        #### 削減効果（B_f – P'_f）
    """
    )
    st.success(f"### {co2_reduction_effect_2} t-CO2eq")
    st.markdown(
        """
        ```
        ＜推計式＞
        市区町村のCO2排出量
        ＝Σ ゾーン間のトリップ数 × 一人当たりトリップ距離(km) / 燃費(km/l) × 1/1000(kl/l) × 単位発熱量(GJ/kl) × 排出係数(t-C/GJ) × 44/12(t-CO2/t-C)
        ```
    """
    )
with col4:
    st.plotly_chart(fig_2, use_container_width=True)


# =======================================================OD可視化=======================================================
gdf_car = gpd.read_file("data/car_possession.geojson")
target_kzone_list = gdf_car[
    (gdf_car["都県名"] == prefecture) & (gdf_car["市区町村名"].str.contains(city))
]["kzone"].tolist()

st.warning("## ゾーン間の年間CO2排出量(1,000t-CO2eq)")
st.write(f"#### {city}の基本計画ゾーン")
target_kzone = o_kzone = st.selectbox("Please select target_kzone", target_kzone_list)

df_o = (
    df[["発地", "着地", "旅客_new_CO2_yearly"]]
    >> inner_join(
        gdf_car[["kzone", "geometry"]].rename(columns={"kzone": "着地"}), by="着地"
    )
    >> mask(X.発地 == target_kzone, X.旅客_new_CO2_yearly != 0)
    >> rename(年間CO排出量=X.旅客_new_CO2_yearly)
)

df_d = (
    df[["発地", "着地", "旅客_new_CO2_yearly"]]
    >> inner_join(
        gdf_car[["kzone", "geometry"]].rename(columns={"kzone": "発地"}), by="発地"
    )
    >> mask(X.着地 == target_kzone, X.旅客_new_CO2_yearly != 0)
    >> rename(年間CO排出量=X.旅客_new_CO2_yearly)
)


tmp_gdf = gdf_car[gdf_car["kzone"] == target_kzone]
tmp_gdf["p"] = tmp_gdf["geometry"].centroid
tmp_gdf["lon"] = tmp_gdf["p"].apply(lambda x: x.coords[0][0])
tmp_gdf["lat"] = tmp_gdf["p"].apply(lambda x: x.coords[0][1])

vmax = pd.concat([df_o, df_d])["年間CO排出量"].quantile(0.95)
vmin = pd.concat([df_o, df_d])["年間CO排出量"].quantile(0.05)

fig_o = px.choropleth_mapbox(
    df_o,
    geojson=gpd.GeoSeries(df_o["geometry"]).__geo_interface__,
    locations=df_o.index,
    color="年間CO排出量",
    color_continuous_scale="reds",
    center={"lon": tmp_gdf["lon"].tolist()[0], "lat": tmp_gdf["lat"].tolist()[0]},
    hover_data=["発地", "着地", "年間CO排出量"],
    mapbox_style="carto-positron",
    opacity=0.5,
    zoom=9,
    height=700,
    range_color=(int(vmin), int(vmax)),
)

fig_d = px.choropleth_mapbox(
    df_d,
    geojson=gpd.GeoSeries(df_d["geometry"]).__geo_interface__,
    locations=df_d.index,
    color="年間CO排出量",
    color_continuous_scale="reds",
    center={"lon": tmp_gdf["lon"].tolist()[0], "lat": tmp_gdf["lat"].tolist()[0]},
    hover_data=["発地", "着地", "年間CO排出量"],
    mapbox_style="carto-positron",
    opacity=0.5,
    zoom=9,
    height=700,
    range_color=(int(vmin), int(vmax)),
)

col5, col6 = st.columns(2)
with col5:
    st.write("### 出発地集計")
    st.plotly_chart(fig_o, use_container_width=True)

with col6:
    st.write("### 目的地集計")
    st.plotly_chart(fig_d, use_container_width=True)
