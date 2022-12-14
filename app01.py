import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
from st_aggrid import AgGrid
from optimize import optimize_loop
from scheduler import create_final_agent_summary, create_wage_distribution
from utils import (
    aggregate_dispatch_demand_summary_0,
    aggregate_dispatch_hourly,
    aggregate_money_summary,
)

input_dir = Path("data/demo01/input")
output_dir = Path("data/demo01/output")
solver_timeout = 60

st.set_page_config(layout="wide")
st.write("## 青柳富士見線・富士見温泉線")

st.sidebar.write("### 関越交通：一人当たり平均運賃")
fare_01 = st.sidebar.slider("fare_01", min_value=100, max_value=800, step=10, value=400)
st.sidebar.write("### 日本中央：一人当たり平均運賃")
fare_23 = st.sidebar.slider("fare_23", min_value=100, max_value=800, step=10, value=400)

st.sidebar.write("### 関越交通：車両ごとの乗車定員数")
capacity_01 = st.sidebar.slider(
    "capacity_01", min_value=10, max_value=50, step=1, value=24
)
st.sidebar.write("### 日本中央：車両ごとの乗車定員数")
capacity_23 = st.sidebar.slider(
    "capacity_23", min_value=10, max_value=50, step=1, value=31
)

st.sidebar.write("### 関越交通：割り当て車両台数")
total_number_bus_01 = st.sidebar.slider(
    "total_number_bus_01", min_value=1, max_value=10, step=1, value=10
)
st.sidebar.write("### 日本中央：割り当て車両台数")
total_number_bus_23 = st.sidebar.slider(
    "total_number_bus_23", min_value=1, max_value=10, step=1, value=10
)

st.sidebar.write("### 関越交通：キロ単価")
cost_01 = st.sidebar.slider("cost_01", min_value=100, max_value=500, step=10, value=240)
st.sidebar.write("### 日本中央：キロ単価")
cost_23 = st.sidebar.slider("cost_23", min_value=100, max_value=500, step=10, value=310)

st.sidebar.write("### 繰り越しして乗客が待つ時間")
max_i = st.sidebar.slider("max_i", min_value=1, max_value=5, step=1, value=3)

st.sidebar.write("### 需要成長係数")
growth_rate = st.sidebar.slider(
    "growth_rate", min_value=0.0, max_value=2.0, step=0.1, value=1.5
)

dfs_company_operation = pd.DataFrame(
    data={
        "company": ["関越交通", "関越交通", "日本中央", "日本中央"],
        "route_id": [0, 1, 2, 3],
        "fare": [fare_01, fare_01, fare_23, fare_23],
        "capacity": [capacity_01, capacity_01, capacity_23, capacity_23],
        "cost": [cost_01, cost_01, cost_23, cost_23],
        "cost_present": [240, 240, 310, 310],
        "total_number_bus": [
            total_number_bus_01,
            total_number_bus_01,
            total_number_bus_23,
            total_number_bus_23,
        ],
        "work_duration_dict": [2, 2, 2, 2],
        "working_distance": [11.0, 11.0, 11.0, 11.0],
        "parking_duration": [2, 2, 2, 2],
    }
)

dfs_common_operation = pd.DataFrame(
    data={
        "route_id": [0, 1, 2, 3],
        "bus_limit_window": [4, 4, 4, 4],
        "max_i": [max_i] * 4,
        "growth_rate": [growth_rate] * 4,
        "route_name": ["富士見温泉線", "富士見温泉線", "青柳富士見線", "青柳富士見線"],
        "route_length": [12.6, 12.6, 18.0, 18.0],
        "fare": [400, 400, 500, 500],
    }
)

dfs_company_operation.to_csv(input_dir.joinpath("input1.csv"), index=None, header=None)
dfs_common_operation.to_csv(input_dir.joinpath("input3.csv"), index=None, header=None)

optimize_loop(input_dir, output_dir, solver_timeout)  # 運行本数の最適化（No.6-1）
aggregate_dispatch_hourly(input_dir, output_dir)  # 運行ダイヤ：簡易運行表（No. 8）
aggregate_dispatch_demand_summary_0(input_dir, output_dir)  # 運行本数・乗客総数サマリ（No. 6-2）
aggregate_money_summary(input_dir, output_dir)  # シミュレーション後収支等の可視化（No. 7）
create_final_agent_summary(input_dir, output_dir)  # 仕業総括表（No. 9）
create_wage_distribution(input_dir, output_dir)  # 運賃分配（No. 10）


st.write("### 年間営業収支")
df_money_summary = pd.read_csv(
    output_dir.joinpath("output3.csv"),
    header=None,
    names=["company", "route_id", "曜日区分", "項目", "種別", "金額", "金額(現行)"],
)

df_money_summary = (
    df_money_summary.groupby(["company", "項目", "種別"])[["金額", "金額(現行)"]]
    .sum()
    .reset_index()
)

fig_1 = px.bar(
    df_money_summary,
    x="company",
    y="金額",
    color="種別",
    barmode="group",
    facet_col="項目",
    text_auto=True,
    title="",
)
st.plotly_chart(fig_1, use_container_width=True)
AgGrid(df_money_summary, theme="streamlit", fit_columns_on_grid_load=True)

st.write("### 運行本数サマリー")
df_dispatch_demand_summary = pd.read_csv(
    output_dir.joinpath("output2.csv"),
    header=None,
    names=["company", "route_id", "曜日区分", "項目", "乗客数", "運行本数"],
)
df_dispatch_summary = (
    df_dispatch_demand_summary.groupby(["company", "項目"])["運行本数"].sum().reset_index()
)
fig_2 = px.bar(
    df_dispatch_summary,
    x="company",
    y="運行本数",
    color="項目",
    barmode="group",
    text_auto=True,
    title="1日の合計バス本数",
)
st.plotly_chart(fig_2, use_container_width=True)

st.write("### 簡易運行表")
df_dispatch_hourly = pd.read_csv(
    output_dir.joinpath("output4.csv"),
    header=None,
    names=["company", "route_id", "曜日区分", "時間帯", "項目", "シミュレーション後本数", "現行本数", "差分"],
)
AgGrid(df_dispatch_hourly, theme="streamlit", fit_columns_on_grid_load=True, height=500)

st.write("### 仕業総括表")
df_shigyo = pd.read_csv(
    output_dir.joinpath("output5.csv"),
    header=None,
    names=[
        "scenario_id",
        "company",
        "曜日区分",
        "バス番号",
        "運転手番号",
        "項目",
        "始業時刻",
        "終業時刻",
        "ハンドル時間",
        "拘束時間",
        "実走距離",
        "トリップ数",
    ],
)
AgGrid(df_shigyo, theme="streamlit", fit_columns_on_grid_load=True, height=500)

st.write("### 運賃分配")
df_unchin_bunpai = pd.read_csv(
    output_dir.joinpath("output6.csv"),
    header=None,
    names=["company", "曜日区分", "category", "収入分配", "収入増減額", "分配後収入"],
)

AgGrid(df_unchin_bunpai, theme="streamlit", fit_columns_on_grid_load=True)
