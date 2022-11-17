# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.constant import EPS
from qlib.rl.data.base import ProcessedDataProvider
from qlib.rl.interpreter import StateInterpreter
from qlib.rl.order_execution.state import SAOEState
from qlib.typehint import TypedDict
from qlib.utils import init_instance_by_config


def _to_int32(val):
    return np.array(int(val), dtype=np.int32)


def _to_float32(val):
    return np.array(val, dtype=np.float32)


def canonicalize(value: int | float | np.ndarray | pd.DataFrame | dict) -> np.ndarray | dict:
    """To 32-bit numeric types. Recursively."""
    if isinstance(value, pd.DataFrame):
        return value.to_numpy()
    if isinstance(value, (float, np.floating)) or (isinstance(value, np.ndarray) and value.dtype.kind == "f"):
        return np.array(value, dtype=np.float32)
    elif isinstance(value, (int, bool, np.integer)) or (isinstance(value, np.ndarray) and value.dtype.kind == "i"):
        return np.array(value, dtype=np.int32)
    elif isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    else:
        return value


class WholeDayObs(TypedDict):
    data_processed: Any
    data_processed_prev: Any
    acquiring: Any
    cur_tick: Any
    cur_step: Any
    num_step: Any
    target: Any
    position: Any
    position_history: Any
    tick_history: Any


class WholeDayStateInterpreter(StateInterpreter[SAOEState, WholeDayObs]):
    """The observation of all the history, including today (entire day), and yesterday.

    Parameters
    ----------
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    data_ticks
        Equal to the total number of records. For example, in SAOE per minute,
        the total ticks is the length of day in minutes.
    data_dim
        Number of dimensions in data.
    processed_data_provider
        Provider of the processed data.
    """

    def __init__(
        self,
        max_step: int,
        data_ticks: int,
        data_dim: int,
        processed_data_provider: dict | ProcessedDataProvider,
    ) -> None:
        super().__init__()

        self.max_step = max_step
        self.data_ticks = data_ticks
        self.data_dim = data_dim
        self.processed_data_provider: ProcessedDataProvider = init_instance_by_config(
            processed_data_provider,
            accept_types=ProcessedDataProvider,
        )

    def interpret(self, state: SAOEState) -> WholeDayObs:
        processed = self.processed_data_provider.get_data(
            stock_id=state.order.stock_id,
            date=pd.Timestamp(state.order.start_time.date()),
            feature_dim=self.data_dim,
            time_index=state.ticks_index,
        )

        position_history = np.full(self.max_step + 1, 0.0, dtype=np.float32)
        position_history[0] = state.order.amount
        position_history[1 : len(state.history_steps) + 1] = state.history_steps["position"].to_numpy()

        history_ticks = np.array([
            min(int(np.sum(state.ticks_index < tick)), self.data_ticks - 1)
            for tick in state.history_steps.index
        ], dtype=np.int32)
        history_ticks = np.concatenate((history_ticks, [0] * (self.max_step - len(history_ticks)))).astype(int)

        # The min, slice here are to make sure that indices fit into the range,
        # even after the final step of the simulator (in the done step),
        # to make network in policy happy.
        return cast(
            WholeDayObs,
            canonicalize(
                {
                    "data_processed": np.array(processed.today),
                    "data_processed_prev": np.array(processed.yesterday),
                    "acquiring": _to_int32(state.order.direction == state.order.BUY),
                    "cur_tick": _to_int32(min(int(np.sum(state.ticks_index < state.cur_time)), self.data_ticks - 1)),
                    "cur_step": _to_int32(min(state.cur_step, self.max_step - 1)),
                    "num_step": _to_int32(self.max_step),
                    "target": _to_float32(state.order.amount),
                    "position": _to_float32(state.position),
                    "position_history": _to_float32(position_history[: self.max_step]),
                    "tick_history": history_ticks,
                },
            ),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        space = {
            "data_processed": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "data_processed_prev": spaces.Box(-np.inf, np.inf, shape=(self.data_ticks, self.data_dim)),
            "acquiring": spaces.Discrete(2),
            "cur_tick": spaces.Box(0, self.data_ticks - 1, shape=(), dtype=np.int32),
            "cur_step": spaces.Box(0, self.max_step - 1, shape=(), dtype=np.int32),
            # TODO: support arbitrary length index
            "num_step": spaces.Box(self.max_step, self.max_step, shape=(), dtype=np.int32),
            "target": spaces.Box(-EPS, np.inf, shape=()),
            "position": spaces.Box(-EPS, np.inf, shape=()),
            "position_history": spaces.Box(-EPS, np.inf, shape=(self.max_step,)),
            "tick_history": spaces.Box(0, self.data_ticks - 1, shape=(self.max_step,), dtype=np.int32),
        }
        return spaces.Dict(space)
