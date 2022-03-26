from math import ceil
from typing import List, Optional, Tuple

from avdata.data_structures.agent import AgentMetadata, AgentType


def exclude_types(no_types: Optional[List[AgentType]], agent_type: AgentType) -> bool:
    return no_types is not None and agent_type in no_types


def all_agents_excluded_types(
    no_types: Optional[List[AgentType]], agents: List[AgentMetadata]
) -> bool:
    return no_types is not None and all(
        agent_info.type in no_types for agent_info in agents
    )


def not_included_types(
    only_types: Optional[List[AgentType]], agent_type: AgentType
) -> bool:
    return only_types is not None and agent_type not in only_types


def no_agent_included_types(
    only_types: Optional[List[AgentType]], agents: List[AgentMetadata]
) -> bool:
    return only_types is not None and all(
        agent_info.type not in only_types for agent_info in agents
    )


def robot_future(incl_robot_future: bool, agent_name: str) -> bool:
    return incl_robot_future and agent_name == "ego"


def satisfies_history(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    if history_sec[0] is not None:
        min_history = ceil(history_sec[0] / dt)
        agent_history_satisfies = ts - agent_info.first_timestep >= min_history
    else:
        agent_history_satisfies = True

    return agent_history_satisfies


def satisfies_future(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    future_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    if future_sec[0] is not None:
        min_future = ceil(future_sec[0] / dt)
        agent_future_satisfies = agent_info.last_timestep - ts >= min_future
    else:
        agent_future_satisfies = True

    return agent_future_satisfies


def satisfies_times(
    agent_info: AgentMetadata,
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
) -> bool:
    agent_history_satisfies = satisfies_history(agent_info, ts, dt, history_sec)
    agent_future_satisfies = satisfies_future(agent_info, ts, dt, future_sec)
    return agent_history_satisfies and agent_future_satisfies


def no_agent_satisfies_time(
    ts: int,
    dt: float,
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
    agents: List[AgentMetadata],
) -> bool:
    return all(
        not satisfies_times(agent_info, ts, dt, history_sec, future_sec)
        for agent_info in agents
    )
