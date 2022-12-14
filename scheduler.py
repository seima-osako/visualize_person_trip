import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd

pd.options.mode.chained_assignment = None


@dataclasses.dataclass(frozen=True)
class SimulationParam:
    work_duration_dict: dict[int, int]
    offset_dict: dict[tuple[int, int], int]
    one_way_distance: dict[int, float]
    offset_distance_dict: dict[tuple[int, int], float]
    parking_duration: dict[int, int]
    route_id_dict: dict[int, int]


@dataclasses.dataclass(frozen=True)
class Task:
    time_slot: int
    direction: int


@dataclasses.dataclass
class Agent:
    id: str
    tasks: list[Task] = dataclasses.field(default_factory=list)

    def next_ready_time(self, direction: int, prm: SimulationParam):
        if len(self.tasks) == 0:
            return 0

        last_task = self.tasks[-1]

        i = direction
        j = prm.route_id_dict[last_task.direction]
        k = prm.route_id_dict[i]

        return (
            last_task.time_slot
            + prm.work_duration_dict[i]
            + prm.offset_dict.get((j, k), 1000)
        )


class SplitAgent(TypedDict):
    id: int
    driver_id: int
    tasks: list[Task]


def create_prm(
    optimized: pd.DataFrame,
    company_operation_param_file: pd.DataFrame,
    bidirectional_parameters_file: pd.DataFrame,
    company_name: str,
):

    route_ids = (
        optimized[optimized["company"] == company_name]["route_id"].unique().tolist()
    )

    # Upload route length parameters
    company_operation_param = company_operation_param_file
    company_operation_param = company_operation_param[
        (company_operation_param["company"] == company_name)
        & (company_operation_param["route_id"].isin(route_ids))
    ]
    company_operation_param.reset_index(drop=True, inplace=True)

    # Upload offset time and distance parameters
    bidirectional_parameters = bidirectional_parameters_file
    bidirectional_parameters = bidirectional_parameters[
        (bidirectional_parameters["company"] == company_name)
        & (bidirectional_parameters["previous_route_id"].isin(route_ids))
        & (bidirectional_parameters["next_route_id"].isin(route_ids))
    ]

    bidirectional_parameters["index"] = list(
        zip(
            bidirectional_parameters["previous_route_id"],
            bidirectional_parameters["next_route_id"],
        )
    )
    bidirectional_parameters.drop(
        columns=["previous_route_id", "next_route_id"], inplace=True
    )
    bidirectional_parameters.set_index("index", inplace=True)

    # Set parameters
    prm = SimulationParam(
        work_duration_dict=company_operation_param.to_dict()["work_duration_dict"],
        offset_dict=bidirectional_parameters["offset_time"],
        one_way_distance=company_operation_param.to_dict()["working_distance"],
        offset_distance_dict=bidirectional_parameters["offset_distance"],
        parking_duration=company_operation_param.to_dict()["parking_duration"],
        route_id_dict=company_operation_param.to_dict()["route_id"],
    )

    return prm


def get_restrictions(workinghours_parameters_file: pd.DataFrame, company: str):

    workinghours_parameters_file = workinghours_parameters_file.set_index("company")

    kousoku_hours = workinghours_parameters_file.loc[company, "kousoku_hours"]
    handle_hours = workinghours_parameters_file.loc[company, "handle_hours"]
    pause = workinghours_parameters_file.loc[company, "pause"]

    return kousoku_hours, handle_hours, pause


def clean_simulation_result(
    optimized: pd.DataFrame, company: str, weekday_or_weekend: str, category: str
):

    """
    Function that takes the simulation results as an input,
    cleans and outputs it in a form that can be read by the scheduler.
    """

    # Upload bus demand file and separate by weekday/weekend
    demand = optimized

    # Convert time format to datetime
    demand["time_slot"] = demand["time_slot"].astype("string")
    demand["time_slot"] = [
        datetime.strptime(x, "%H:%M:%S").time() for x in demand["time_slot"]
    ]

    demand_dataframe = demand[demand["category"] == category]
    demand_dataframe = demand_dataframe[
        demand_dataframe["day_of_the_week"] == weekday_or_weekend
    ]
    demand_dataframe = (
        demand_dataframe[demand_dataframe["company"] == company]
        .drop(columns=["category", "day_of_the_week", "optimized_demand", "company"])
        .reset_index(drop=True)
    )

    # Return to a pivot format for scheduler
    demand_cleaned = pd.pivot(demand_dataframe, index="time_slot", columns="route_id")[
        "optimized_dispatch"
    ]
    demand_cleaned.columns.name = None
    demand_cleaned.reset_index(inplace=True)

    # Separate time slots and demand per route
    time_slots = demand_cleaned.time_slot
    demand_cleaned = demand_cleaned.drop(columns=["time_slot"])

    # Convert demand per route to numpy format for scheduler
    scheduler_input = (
        demand_cleaned.swapaxes("index", "columns").fillna(0.0).astype(int).to_numpy()
    )

    return demand_cleaned, time_slots, scheduler_input


def assign_tasks(scheduler_input: list[list[int]], prm: SimulationParam):

    """
    Function to create agents (buses) and assign them to
    each task (bus leaving on specific route at specific timeslot)
    """

    # Input route/direction and timeslot from cleaned schedule data
    route_direction = len(scheduler_input)
    time_slot = len(scheduler_input[0])

    # Create empty list of agents
    agents: list[Agent] = []

    # Loop through each time slot and route/direction
    for t in range(time_slot):
        for r in range(route_direction):

            # Create a new task for each time slot/route/direction combination
            new_task = Task(t, r)

            # Demand is number of buses needed to meet demand
            demand = scheduler_input[r][t]

            # Assign an agent to each demand (each bus needing to leave at
            # each time slot/route/direction combination
            for j in range(demand):

                # If no agents created yet, create first agent
                if len(agents) == 0:
                    agents.append(Agent(str(0), [new_task]))

                # Otherwise, use available agent where next_ready_time
                # (see Agent class) is less than time slot
                else:
                    available_agents = [
                        a for a in agents if a.next_ready_time(r, prm) <= t
                    ]

                    # If no available agents, create new agent
                    if len(available_agents) == 0:
                        n = len(agents)
                        agents.append(Agent(str(n), [new_task]))

                    # If available agent, use agent with the least number of tasks??
                    else:
                        best_agent = min(available_agents, key=lambda a: len(a.tasks))
                        best_agent.tasks.append(new_task)

    return agents


def agent_statistics(
    tasks: list[Task],
    prm: SimulationParam
):

    service_driving_time = 0
    service_distance = 0.0
    non_service_driving_time = 0
    non_service_distance = 0.0

    if len(tasks) > 1:
        for i in range(1, len(tasks)):
            direction = tasks[i].direction
            last_direction = tasks[i - 1].direction
            direction_route = prm.route_id_dict[direction]
            last_direction_route = prm.route_id_dict[last_direction]

            service_driving_time += prm.work_duration_dict[direction]
            service_distance += prm.one_way_distance[direction]
            non_service_driving_time += prm.offset_dict[
                (last_direction_route, direction_route)
            ]
            non_service_distance += prm.offset_distance_dict[
                (last_direction_route, direction_route)
            ]

    start = tasks[0].time_slot - 1
    begin_route = tasks[0].time_slot
    end_route = tasks[-1].time_slot + prm.work_duration_dict[tasks[-1].direction]
    end = tasks[-1].time_slot + prm.work_duration_dict[tasks[-1].direction] + 1
    total_work_time = end - start

    total_distance = service_distance + non_service_distance

    return {
        "始業": start,
        "出庫": begin_route,
        "終業": end,
        "入庫": end_route,
        "拘束時間": total_work_time,
        "ハンドル時間": service_driving_time,
        "回送時間": non_service_driving_time,
        "実走距離": service_distance,
        "回送距離": non_service_distance,
        "総距離": total_distance,
    }


def create_agent_summary(
    agents: list[Agent],
    time_slots: list[str],
    prm: SimulationParam,
    time_interval: int = 15,
):

    dt_start = datetime.strptime(str(time_slots[0]), "%H:%M:%S")
    dt_min_interval = timedelta(minutes=time_interval)

    d = []

    for agent in agents:
        stats = agent_statistics(agent.tasks, prm)
        start_time = dt_start + dt_min_interval * (stats["始業"])
        end_time = dt_start + dt_min_interval * (stats["終業"])

        d.append(
            {
                "scenario_id": 0,
                "バス番号": agent.id,
                "運転手番号": 0,
                # "路線": '富士見温泉' if agent.tasks[0].direction <= 1 else '青柳温泉線',
                "始業時刻": "{:d}:{:02d}".format(start_time.hour, start_time.minute),
                "終業時刻": "{:d}:{:02d}".format(end_time.hour, end_time.minute),
                "ハンドル時間": stats["ハンドル時間"] * time_interval * 1.25 / 60,
                "拘束時間": stats["拘束時間"] * time_interval / 60,
                # "回送時間": len(agent.tasks) * 6 / 60,
                "実走距離": int(stats["実走距離"]),
                # "回送距離": stats["回送距離"],
                # "総距離": stats["総距離"],
                "トリップ数": len(agent.tasks),
            }
        )

    agent_summary = pd.DataFrame(d)
    return agent_summary


def create_split_agents_table(
    agents: list[Agent],
    agent_summary: pd.DataFrame,
    kousoku_hours: int = 13,
    handle_hours: int = 9,
    pause: int = 11,
):

    import math

    agents_over_kosoku_limit = []
    agents_not_over_kosoku_limit = []

    agents_split = []

    for index, row in agent_summary.iterrows():
        if row["拘束時間"] > kousoku_hours:
            agents_over_kosoku_limit.append(index)
        elif row["ハンドル時間"] > handle_hours:
            agents_over_kosoku_limit.append(index)
        elif 24 - row["拘束時間"] > pause:
            agents_over_kosoku_limit.append(index)
        else:
            agents_not_over_kosoku_limit.append(index)

    for agent in agents_over_kosoku_limit:
        first_driver_task_number = math.ceil(len(agents[agent].tasks) / 2)
        first_driver_tasks = agents[agent].tasks[:first_driver_task_number]

        agents_split.append(
            {"id": agents[agent].id, "driver_id": 0, "tasks": first_driver_tasks}
        )

        second_driver_task_number = len(agents[agent].tasks) - first_driver_task_number
        second_driver_tasks = agents[agent].tasks[-second_driver_task_number:]

        agents_split.append(
            {"id": agents[agent].id, "driver_id": 1, "tasks": second_driver_tasks}
        )

    for agent in agents_not_over_kosoku_limit:
        agents_split.append(
            {
                "id": agents[agent].id,
                "driver_id": 0,
                "tasks": agents[agent].tasks,
            }
        )

    return agents_split


def create_split_agent_summary(
    agents_split: list[SplitAgent],
    time_slots: list[str],
    prm: SimulationParam,
    time_interval: int = 15,
):

    dt_start = datetime.strptime(str(time_slots[0]), "%H:%M:%S")
    dt_min_interval = timedelta(minutes=time_interval)

    d = []

    for agent in agents_split:
        stats = agent_statistics(agent["tasks"], prm)
        start_time = dt_start + dt_min_interval * (stats["始業"])
        end_time = dt_start + dt_min_interval * (stats["終業"])

        d.append(
            {
                "scenario_id": 1,
                "バス番号": agent["id"],
                "運転手番号": agent["driver_id"],
                # "路線": '富士見温泉' if agent['tasks'][0].direction <= 1 else '青柳温泉線',
                "始業時刻": "{:d}:{:02d}".format(start_time.hour, start_time.minute),
                "終業時刻": "{:d}:{:02d}".format(end_time.hour, end_time.minute),
                "ハンドル時間": stats["ハンドル時間"] * time_interval * 1.25 / 60,
                "拘束時間": stats["拘束時間"] * time_interval / 60,
                # "回送時間": len(agent['tasks']) * 6 / 60,
                "実走距離": int(stats["実走距離"]),
                # "回送距離": stats["回送距離"],
                # "総距離": stats["総距離"],
                "トリップ数": len(agent["tasks"]),
            }
        )

    split_agent_summary = pd.DataFrame(d)
    return split_agent_summary


def create_agent_summary_by_company_weekday_category(
    optimized: pd.DataFrame,
    company: str,
    weekday_weekend: str,
    category: str,
    prm: SimulationParam,
    kousoku_hours: int = 13,
    handle_hours: int = 9,
    pause: int = 11,
    time_interval: int = 15,
):
    demand_cleaned, time_slots, scheduler_input = clean_simulation_result(
        optimized, company, weekday_weekend, category
    )
    agents = assign_tasks(scheduler_input, prm)
    agent_summary = create_agent_summary(agents, time_slots, prm, time_interval)
    agents_split = create_split_agents_table(
        agents, agent_summary, kousoku_hours, handle_hours, pause
    )
    split_agent_summary = create_split_agent_summary(
        agents_split, time_slots, prm, time_interval
    )
    shigyo = pd.concat([agent_summary, split_agent_summary], ignore_index=True)
    shigyo["company"] = company
    shigyo["曜日区分"] = weekday_weekend
    shigyo["category"] = category

    cols = [
        "scenario_id",
        "company",
        "曜日区分",
        "バス番号",
        "運転手番号",
        "category",
        "始業時刻",
        "終業時刻",
        "ハンドル時間",
        "拘束時間",
        "実走距離",
        "トリップ数",
    ]

    try:
        return shigyo[cols]
    except KeyError:
        return None


def create_final_agent_summary(input_dir: Path, output_dir: Path):
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
    bidirectional_col = [
        "company",
        "previous_route_id",
        "next_route_id",
        "offset_time",
        "offset_distance",
    ]
    workinghours_col = ["company", "kousoku_hours", "handle_hours", "pause"]
    optimized_col = [
        "company",
        "route_id",
        "day_of_the_week",
        "time_slot",
        "category",
        "optimized_demand",
        "optimized_dispatch",
    ]

    company_operation_param_file = pd.read_csv(
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
    bidirectional_parameters_file = pd.read_csv(
        input_dir.joinpath("input4.csv"),
        names=bidirectional_col,
        dtype={
            "company": str,
            "previous_route_id": int,
            "next_route_id": int,
            "offset_time": int,
            "offset_distance": float,
        },
    )
    workinghours_parameters_file = pd.read_csv(
        input_dir.joinpath("input5.csv"),
        names=workinghours_col,
        dtype={"company": str, "kousoku_hours": int, "handle_hours": int, "pause": int},
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
    companies = dfs_optimized["company"].unique()

    shigyo = pd.DataFrame()
    for weekday_weekend in ["平日", "土日祝"]:
        for company in companies:
            for category in ["需要あり", "需要なし"]:
                kousoku_hours, handle_hours, pause = get_restrictions(
                    workinghours_parameters_file, company
                )
                prm = create_prm(
                    dfs_optimized,
                    company_operation_param_file,
                    bidirectional_parameters_file,
                    company,
                )
                tmp_shigyo = create_agent_summary_by_company_weekday_category(
                    dfs_optimized,
                    company,
                    weekday_weekend,
                    category,
                    prm,
                    kousoku_hours,
                    handle_hours,
                    pause,
                )
                if tmp_shigyo is not None:
                    shigyo = pd.concat([shigyo, tmp_shigyo])

    shigyo.reset_index(drop=True, inplace=True)
    shigyo.to_csv(output_dir.joinpath("output5.csv"), index=None, header=None)


def create_wage_distribution(input_dir: Path, output_dir: Path):
    money_summary = [
        "company",
        "route_id",
        "day_of_the_week",
        "category",
        "type",
        "money",
        "money_present",
    ]
    shigyo_col = [
        "scenario_id",
        "company",
        "曜日区分",
        "バス番号",
        "運転手番号",
        "category",
        "始業時刻",
        "終業時刻",
        "ハンドル時間",
        "拘束時間",
        "実走距離",
        "トリップ数",
    ]

    money_summary = pd.read_csv(
        output_dir.joinpath("output3.csv"),
        names=money_summary,
        dtype={
            "company": str,
            "route_id": int,
            "day_of_the_week": str,
            "category": str,
            "type": str,
            "money": int,
            "money_present": int,
        },
    )

    shigyo = pd.read_csv(
        output_dir.joinpath("output5.csv"),
        names=shigyo_col,
        dtype={
            "scenario_id": int,
            "company": str,
            "曜日区分": str,
            "バス番号": int,
            "運転手番号": int,
            "category": str,
            "始業時刻": str,
            "終業時刻": str,
            "ハンドル時間": float,
            "拘束時間": float,
            "実走距離": int,
            "トリップ数": int,
        },
    )

    money_summary = (
        money_summary[money_summary.iloc[:, 4] == "売上"]
        .groupby(["company", "day_of_the_week", "category"])[["money"]]
        .sum()
    )

    unchin_bunpai = pd.DataFrame()

    for weekday_weekend in shigyo["曜日区分"].unique():
        for category in shigyo["category"].unique():

            df = (
                shigyo[
                    (shigyo["scenario_id"] == 0)
                    & (shigyo["曜日区分"] == weekday_weekend)
                    & (shigyo["category"] == category)
                ]
                .groupby(["company", "曜日区分", "category"])[["ハンドル時間", "トリップ数"]]
                .sum()
            )

            df["収入分配"] = round(
                (
                    (df["ハンドル時間"] / df["ハンドル時間"].sum()) * 0.1
                    + (df["トリップ数"] / df["トリップ数"].sum()) * 0.9
                ),
                2,
            )

            df["money"] = money_summary["money"]
            df["分配後収入"] = (df["収入分配"] * df["money"].sum()).astype(int)
            df["収入増減額"] = (df["分配後収入"] - df["money"]).astype(int)
            df.drop(columns=["ハンドル時間", "トリップ数", "money"], inplace=True)
            df.reset_index(inplace=True)

            cols = ["company", "曜日区分", "category", "収入分配", "収入増減額", "分配後収入"]
            df = df[cols]

            unchin_bunpai = pd.concat([unchin_bunpai, df])
            unchin_bunpai.reset_index(inplace=True, drop=True)

    unchin_bunpai.to_csv(output_dir.joinpath("output6.csv"), index=None, header=None)
