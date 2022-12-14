from pathlib import Path

import pandas as pd


def aggregate_dispatch_hourly(input_dir: Path, output_dir: Path):
    present_col = ["route_id", "day_of_the_week", "time_slot", "company", "dispatch"]
    optimized_col = [
        "company",
        "route_id",
        "day_of_the_week",
        "time_slot",
        "category",
        "optimized_demand",
        "optimized_dispatch",
    ]

    dfs_present = pd.read_csv(
        input_dir.joinpath("input7.csv"),
        names=present_col,
        dtype={
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "company": str,
            "dispatch": int,
        },
    )
    dfs_optimized = pd.read_csv(
        output_dir.joinpath("output1.csv"),
        names=optimized_col,
        dtype={
            "company": str,
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "category": str,
            "optimized_demand": int,
            "optimized_dispatch": int,
        },
    )

    route_ids = sorted(dfs_optimized["route_id"].unique())
    df_dispatch = pd.merge(
        dfs_present[dfs_present["route_id"].isin(route_ids)],
        dfs_optimized.drop(columns="optimized_demand"),
        on=["company", "route_id", "day_of_the_week", "time_slot"],
    )
    df_dispatch["hour"] = df_dispatch["time_slot"].apply(lambda x: int(x.split(":")[0]))
    df_dispatch_hourly = (
        df_dispatch.groupby(
            ["company", "route_id", "day_of_the_week", "hour", "category"]
        )[["optimized_dispatch", "dispatch"]]
        .sum()
        .reset_index()
    )
    df_dispatch_hourly["diff_dispatch"] = (
        df_dispatch_hourly["optimized_dispatch"] - df_dispatch_hourly["dispatch"]
    )
    df_dispatch_hourly.to_csv(
        output_dir.joinpath("output4.csv"), index=None, header=None
    )


def aggregate_dispatch_demand_summary_0(input_dir: Path, output_dir: Path):
    present_col = ["route_id", "day_of_the_week", "time_slot", "company", "dispatch"]
    demand_col = ["route_id", "day_of_the_week", "time_slot", "demand"]
    optimized_col = [
        "company",
        "route_id",
        "day_of_the_week",
        "time_slot",
        "category",
        "optimized_demand",
        "optimized_dispatch",
    ]

    dfs_demand = pd.read_csv(
        input_dir.joinpath("input6.csv"),
        names=demand_col,
        dtype={
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "demand": int,
        },
    )
    dfs_present = pd.read_csv(
        input_dir.joinpath("input7.csv"),
        names=present_col,
        dtype={
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "company": str,
            "dispatch": int,
        },
    )
    dfs_optimized = pd.read_csv(
        output_dir.joinpath("output1.csv"),
        names=optimized_col,
        dtype={
            "company": str,
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "category": str,
            "optimized_demand": int,
            "optimized_dispatch": int,
        },
    )

    route_ids = sorted(dfs_demand["route_id"].unique())

    df_present = pd.merge(
        dfs_present[dfs_present["route_id"].isin(route_ids)],
        dfs_demand,
        on=["route_id", "day_of_the_week", "time_slot"],
    )
    df_present["demand"] = df_present.apply(
        lambda x: 0 if x["dispatch"] == 0 else x["demand"], axis=1
    )
    df_present["category"] = "現行"

    df_optimized = dfs_optimized.rename(
        columns={"optimized_demand": "demand", "optimized_dispatch": "dispatch"}
    )

    df_dispatch_demand_summary = pd.concat([df_present, df_optimized])
    df_dispatch_demand_summary = (
        df_dispatch_demand_summary.groupby(
            ["company", "route_id", "day_of_the_week", "category"]
        )[["demand", "dispatch"]]
        .sum()
        .reset_index()
    )
    df_dispatch_demand_summary.to_csv(
        output_dir.joinpath("output2.csv"), index=None, header=None
    )


def aggregate_dispatch_demand_summary_1(input_dir: Path, output_dir: Path):
    present_col = ["route_id", "day_of_the_week", "time_slot", "company", "dispatch"]
    demand_col = ["route_id", "day_of_the_week", "time_slot", "demand"]
    optimized_col = [
        "company",
        "route_id",
        "day_of_the_week",
        "time_slot",
        "category",
        "optimized_demand",
        "optimized_dispatch",
    ]

    dfs_demand = pd.read_csv(
        input_dir.joinpath("input6.csv"),
        names=demand_col,
        dtype={
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "demand": int,
        },
    )
    dfs_present = pd.read_csv(
        input_dir.joinpath("input7.csv"),
        names=present_col,
        dtype={
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "company": str,
            "dispatch": int,
        },
    )
    dfs_optimized = pd.read_csv(
        output_dir.joinpath("output1.csv"),
        names=optimized_col,
        dtype={
            "company": str,
            "route_id": int,
            "day_of_the_week": str,
            "time_slot": str,
            "category": str,
            "optimized_demand": int,
            "optimized_dispatch": int,
        },
    )

    route_ids = sorted(dfs_demand["route_id"].unique())

    # ===================================現行サマリー===================================
    df_present_h = pd.pivot_table(
        dfs_present[dfs_present["route_id"].isin(route_ids)],
        index=["route_id", "day_of_the_week", "time_slot"],
        columns="company",
        values="dispatch",
    ).reset_index()
    df_present_h = pd.merge(
        dfs_demand[dfs_demand["route_id"].isin(route_ids)],
        df_present_h,
        on=["route_id", "day_of_the_week", "time_slot"],
    )
    companies = ["関越交通", "日本中央", "永井運輸", "群馬中央", "上信観光", "群馬バス"]
    df_present_h["total_dispatch"] = df_present_h[companies].sum(axis=1)

    # 乗客を按分
    for company in companies:
        df_present_h[company] = df_present_h["demand"] * (
            df_present_h[company] / df_present_h["total_dispatch"]
        )
        df_present_h[company] = df_present_h[company].fillna(0).astype(int)

    df_present_h = df_present_h.drop(columns=["demand", "total_dispatch"])
    df_present = pd.melt(
        df_present_h,
        id_vars=["route_id", "day_of_the_week", "time_slot"],
        var_name="company",
        value_name="demand",
    )
    df_present = pd.merge(
        dfs_present,
        df_present,
        on=["company", "route_id", "day_of_the_week", "time_slot"],
    )
    df_present["category"] = "現行"
    # ===============================================================================

    # ===================================シミュレーションサマリー============================
    df_optimized_h = pd.pivot_table(
        dfs_optimized.drop(columns="optimized_demand"),
        index=["route_id", "day_of_the_week", "time_slot", "category"],
        columns="company",
        values="optimized_dispatch",
    ).reset_index()
    df_optimized_h = pd.merge(
        dfs_optimized[
            ["route_id", "day_of_the_week", "time_slot", "category", "optimized_demand"]
        ].drop_duplicates(),
        df_optimized_h,
        on=["route_id", "day_of_the_week", "time_slot", "category"],
    )
    df_optimized_h["total_dispatch"] = df_optimized_h[companies].sum(axis=1)

    for company in companies:
        df_optimized_h[company] = df_optimized_h["optimized_demand"] * (
            df_optimized_h[company] / df_optimized_h["total_dispatch"]
        )
        df_optimized_h[company] = df_optimized_h[company].fillna(0).astype(int)
    df_optimized_h = df_optimized_h.drop(columns=["optimized_demand", "total_dispatch"])

    df_optimized = pd.melt(
        df_optimized_h,
        id_vars=["route_id", "day_of_the_week", "time_slot", "category"],
        var_name="company",
        value_name="demand",
    )
    df_optimized = pd.merge(
        dfs_optimized.drop(columns="optimized_demand").rename(
            columns={"optimized_dispatch": "dispatch"}
        ),
        df_optimized,
        on=["company", "route_id", "day_of_the_week", "time_slot", "category"],
    )
    # ================================================================================

    df_dispatch_demand_summary = pd.concat([df_present, df_optimized])
    df_dispatch_demand_summary = (
        df_dispatch_demand_summary.groupby(
            ["company", "route_id", "day_of_the_week", "category"]
        )[["demand", "dispatch"]]
        .sum()
        .reset_index()
    )
    df_dispatch_demand_summary.to_csv(
        output_dir.joinpath("output2.csv"), index=None, header=None
    )


def aggregate_money_summary(input_dir: Path, output_dir: Path):
    company_operation_col = [
        "company",
        "route_id",
        "fare",
        "capacity",
        "cost",
        "cost_present",
        "total_number_bus",
        "work_duration_dict",
        "working_distance",
        "parking_duration",
    ]
    common_operation_col = [
        "route_id",
        "bus_limit_window",
        "max_i",
        "growth_rate",
        "route_name",
        "route_length",
        "fare",
    ]
    dispatch_demand_summary_col = [
        "company",
        "route_id",
        "day_of_the_week",
        "category",
        "demand",
        "dispatch",
    ]

    dfs_company_operation = pd.read_csv(
        input_dir.joinpath("input1.csv"),
        names=company_operation_col,
        dtype={
            "company": str,
            "route_id": int,
            "fare": int,
            "capacity": int,
            "cost": int,
            "cost_present": int,
            "total_number_bus": int,
            "work_duration_dict": int,
            "working_distance": float,
            "parking_duration": int,
        },
    )
    dfs_common_operation = pd.read_csv(
        input_dir.joinpath("input3.csv"),
        names=common_operation_col,
        dtype={
            "route_id": int,
            "bus_limit_window": int,
            "max_i": int,
            "growth_rate": float,
            "route_name": str,
            "route_length": float,
            "fare": int,
        },
    )
    df_dispatch_demand_summary = pd.read_csv(
        output_dir.joinpath("output2.csv"),
        names=dispatch_demand_summary_col,
        dtype={
            "company": str,
            "route_id": int,
            "day_of_the_week": str,
            "category": str,
            "demand": int,
            "dispatch": int,
        },
    )
    # ===================================現行サマリー===================================
    df_summary_present = pd.merge(
        df_dispatch_demand_summary[df_dispatch_demand_summary["category"] == "現行"],
        dfs_common_operation[["route_id", "route_length", "fare"]],
        on="route_id",
    )
    df_summary_present = pd.merge(
        df_summary_present,
        dfs_company_operation[["company", "route_id", "cost_present"]],
        on=["company", "route_id"],
    )

    df_summary_present["売上"] = df_summary_present.apply(
        lambda x: x["fare"] * x["demand"] * 246
        if x["day_of_the_week"] == "平日"
        else x["fare"] * x["demand"] * 119,
        axis=1,
    )
    df_summary_present["支出"] = df_summary_present.apply(
        lambda x: x["dispatch"] * x["route_length"] * x["cost_present"] * 246
        if x["day_of_the_week"] == "平日"
        else x["dispatch"] * x["route_length"] * x["cost_present"] * 119,
        axis=1,
    )
    df_summary_present["収支"] = df_summary_present["売上"] - df_summary_present["支出"]
    # ===============================================================================

    # ===================================シミュレーションサマリー============================
    df_summary_simulated = pd.merge(
        df_dispatch_demand_summary[df_dispatch_demand_summary["category"] != "現行"],
        dfs_common_operation[["route_id", "route_length"]],
        on="route_id",
    )
    df_summary_simulated = pd.merge(
        df_summary_simulated,
        dfs_company_operation[["company", "route_id", "fare", "cost"]],
        on=["company", "route_id"],
    )
    df_summary_simulated["売上"] = df_summary_simulated.apply(
        lambda x: x["fare"] * x["demand"] * 246
        if x["day_of_the_week"] == "平日"
        else x["fare"] * x["demand"] * 119,
        axis=1,
    )
    df_summary_simulated["支出"] = df_summary_simulated.apply(
        lambda x: x["dispatch"] * x["route_length"] * x["cost"] * 246
        if x["day_of_the_week"] == "平日"
        else x["dispatch"] * x["route_length"] * x["cost"] * 119,
        axis=1,
    )
    df_summary_simulated["収支"] = df_summary_simulated["売上"] - df_summary_simulated["支出"]
    # ================================================================================

    df_summary_present = df_summary_present.drop(
        columns=["dispatch", "demand", "route_length", "fare", "cost_present"]
    )
    df_summary_simulated = df_summary_simulated.drop(
        columns=["dispatch", "demand", "route_length", "fare", "cost"]
    )

    df_summary_present = pd.melt(
        df_summary_present,
        id_vars=["company", "route_id", "day_of_the_week", "category"],
        var_name="type",
        value_name="money_present",
    ).drop(columns="category")
    df_summary_simulated = pd.melt(
        df_summary_simulated,
        id_vars=["company", "route_id", "day_of_the_week", "category"],
        var_name="type",
        value_name="money",
    )
    df_money_summary = pd.merge(
        df_summary_simulated,
        df_summary_present,
        on=["company", "route_id", "day_of_the_week", "type"],
    )
    df_money_summary["money"] = df_money_summary["money"].astype(int)
    df_money_summary["money_present"] = df_money_summary["money_present"].astype(int)
    df_money_summary.to_csv(output_dir.joinpath("output3.csv"), index=None, header=None)
