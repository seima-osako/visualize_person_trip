from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from exceptions import ModelInfeasible, SolutionNotFound


def optimize_dispatch_15min(
    df_demand: pd.DataFrame,
    df_common_operation: pd.DataFrame,
    df_company_operation: pd.DataFrame,
    df_bus_low_high: pd.DataFrame,
    growth_flag: str,
    solver_timeout: int,
):
    if growth_flag == "需要あり":
        growth_rate = df_common_operation["growth_rate"].values[0]
    else:
        growth_rate = 1
    # =============================================各種パラメータ==============================================
    d = [int(dd * growth_rate) for dd in list(df_demand["demand"])]
    total_time_slot = len(d)
    bus_limit_window = df_common_operation["bus_limit_window"].values[0]
    max_i = df_common_operation["max_i"].values[0]
    fare = df_company_operation["fare"].mean()
    companies = df_company_operation["company"].tolist()
    n_bus_companies = len(companies)
    costs = df_company_operation["cost"].tolist()
    capacities = df_company_operation["capacity"].tolist()
    total_number_buses = df_company_operation["total_number_bus"].tolist()
    lower_total_dispatch_buses = [
        df_bus_low_high[df_bus_low_high["company"] == company][
            "lower_total_dispatch"
        ].tolist()
        for company in companies
    ]
    upper_total_dispatch_buses = [
        df_bus_low_high[df_bus_low_high["company"] == company][
            "upper_total_dispatch"
        ].tolist()
        for company in companies
    ]
    # ======================================================================================================

    # =============================================運行本数最適化モデル=============================================
    model = cp_model.CpModel()

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for t in range(total_time_slot):
        for k in range(n_bus_companies):
            # 0 <= x[t, k] <= b[k]
            xx = model.NewIntVar(0, total_number_buses[k], f"x_{t}_{k}")
            x[(t, k)] = xx

    z: dict[tuple[int, int], cp_model.IntVar] = {}
    for t in range(total_time_slot):
        for i in range(max_i):
            # 0 <= z[t, i] <= d[t]
            zz = model.NewIntVar(0, d[t], f"z_{t}_{i}")
            z[(t, i)] = zz

    # sum_i z[t, i] <= d[t]
    for t in range(total_time_slot):
        expr = sum(z[(t, i)] for i in range(max_i)) <= d[t]
        model.Add(expr)

    # sum_i z[t-i, i] <= sum_zk x[t,k] * r[k]
    for t in range(total_time_slot):
        if t <= max_i - 2:
            range_ub = t + 1
        else:
            range_ub = max_i
        latent_demand = sum(z[t - i, i] for i in range(range_ub))
        actual_capacity = sum(x[t, k] * capacities[k] for k in range(n_bus_companies))
        model.Add(latent_demand <= actual_capacity)

    # sum_s x[t+s, k] <= b[k]
    for t in range(total_time_slot):
        for k in range(n_bus_companies):
            if t <= total_time_slot - bus_limit_window:
                range_ub = bus_limit_window
            else:
                range_ub = total_time_slot - t
            dispatched = sum(x[t + s, k] for s in range(range_ub))
            model.Add(dispatched <= total_number_buses[k])

    for k in range(n_bus_companies):
        for j in range(int(total_time_slot / 4)):
            dispatched = sum(x[t, k] for t in np.arange(j * 4, j * 4 + 4))
            model.Add(dispatched <= upper_total_dispatch_buses[k][j])
            model.Add(dispatched >= lower_total_dispatch_buses[k][j])

    # total revenue: TR(z) = P sum_t,i z[t, i]
    total_revenue = fare * sum(
        z[t, i] for t in range(total_time_slot) for i in range(max_i)
    )

    # total cost: TC(x) = sum_k c(k) sum_t x[t, k]
    total_cost = sum(
        costs[k] * sum(x[t, k] for t in range(total_time_slot))
        for k in range(n_bus_companies)
    )

    obj = total_revenue - total_cost
    model.Maximize(obj)
    # ===========================================================================================================

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_timeout
    status = solver.Solve(model)

    if status == cp_model.INFEASIBLE:
        raise ModelInfeasible()
    elif status == cp_model.UNKNOWN:
        raise SolutionNotFound()

    optimize_x: dict[tuple[int, int], int] = {}
    for t in range(total_time_slot):
        for k in range(n_bus_companies):
            optimize_x[(t, k)] = solver.Value(x[(t, k)])

    optimize_z: dict[tuple[int, int], int] = {}
    for t in range(total_time_slot):
        for i in range(max_i):
            optimize_z[(t, i)] = solver.Value(z[(t, i)])

    dispatch: list[list[int]] = []
    for t in range(total_time_slot):
        tmp_x = [optimize_x[(t, k)] for k in range(n_bus_companies)]
        dispatch.append(tmp_x)

    actual_pop: list[int] = []
    for t in range(total_time_slot):
        if t <= max_i - 2:
            range_ub = t + 1
        else:
            range_ub = max_i
        actual_pop.append(sum(optimize_z[t - i, i] for i in range(range_ub)))

    df_optimized = pd.concat([pd.DataFrame(actual_pop), pd.DataFrame(dispatch)], axis=1)
    df_optimized.columns = ["optimized_demand"] + companies
    df_optimized["category"] = growth_flag
    return df_optimized


def optimize_loop(input_dir: Path, output_dir: Path, solver_timeout: int):
    demand_col = ["route_id", "day_of_the_week", "time_slot", "demand"]
    company_bus_low_high_col = [
        "route_id",
        "company",
        "hour",
        "day_of_the_week",
        "lower_total_dispatch",
        "upper_total_dispatch",
    ]
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
    dfs_bus_low_high = pd.read_csv(
        input_dir.joinpath("input2.csv"),
        names=company_bus_low_high_col,
        dtype={
            "route_id": int,
            "company": str,
            "hour": int,
            "day_of_the_week": str,
            "lower_total_dispatch": int,
            "upper_total_dispatch": int,
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

    route_ids = sorted(dfs_demand["route_id"].unique())

    dfs_optimized = pd.DataFrame()

    for dow in ["平日", "土日祝"]:
        for route_id in route_ids:
            for growth_flag in ["需要あり", "需要なし"]:
                df_demand = dfs_demand[
                    (dfs_demand["day_of_the_week"] == dow)
                    & (dfs_demand["route_id"] == route_id)
                ]
                df_common_operation = dfs_common_operation[
                    dfs_common_operation["route_id"] == route_id
                ]
                df_company_operation = dfs_company_operation[
                    dfs_company_operation["route_id"] == route_id
                ]
                df_bus_low_high = dfs_bus_low_high[
                    (dfs_bus_low_high["day_of_the_week"] == dow)
                    & (dfs_bus_low_high["route_id"] == route_id)
                ]
                df_optimized = optimize_dispatch_15min(
                    df_demand,
                    df_common_operation,
                    df_company_operation,
                    df_bus_low_high,
                    growth_flag,
                    solver_timeout,
                )
                df_optimized = pd.concat(
                    [
                        df_demand[
                            ["route_id", "day_of_the_week", "time_slot"]
                        ].reset_index(drop=True),
                        df_optimized,
                    ],
                    axis=1,
                )
                dfs_optimized = pd.concat([dfs_optimized, df_optimized])

    dfs_optimized = pd.melt(
        dfs_optimized,
        id_vars=[
            "route_id",
            "day_of_the_week",
            "time_slot",
            "category",
            "optimized_demand",
        ],
        var_name="company",
        value_name="optimized_dispatch",
    )
    dfs_optimized = dfs_optimized[
        [
            "company",
            "route_id",
            "day_of_the_week",
            "time_slot",
            "category",
            "optimized_demand",
            "optimized_dispatch",
        ]
    ]
    dfs_optimized = dfs_optimized[dfs_optimized["optimized_dispatch"].notnull()]
    dfs_optimized["optimized_dispatch"] = dfs_optimized["optimized_dispatch"].astype(
        int
    )

    dfs_optimized.to_csv(output_dir.joinpath("output1.csv"), index=None, header=None)
