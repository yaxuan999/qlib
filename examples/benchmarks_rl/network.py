# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import cast, Tuple

import torch
import torch.nn as nn
from tianshou.data import Batch

from examples.benchmarks_rl.interpreter import WholeDayObs


class PPONetwork(nn.Module):
    """Network forked from https://github.com/microsoft/qlib/blob/high-freq-execution/examples/trade/network/ppo.py.
    """
    def __init__(
        self,
        obs_space: WholeDayObs,
        max_step: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        cnn_shape: Tuple[int, int] = (30, 6),
    ) -> None:
        super().__init__()

        self.max_step = max_step
        self.output_dim = output_dim

        self.rnn = nn.GRU(64, hidden_dim, batch_first=True)
        self.rnn2 = nn.GRU(64, hidden_dim, batch_first=True)
        self.dnn = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),)
        self.cnn = nn.Sequential(nn.Conv1d(cnn_shape[1], 3, 3), nn.ReLU(),)
        self.raw_fc = nn.Sequential(nn.Linear((cnn_shape[0] - 2) * 3, 64), nn.ReLU(),)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.ReLU(),
        )

    def _get_private_state(self, obs: WholeDayObs) -> torch.Tensor:
        batch_size = obs["position_history"].shape[0]
        return torch.cat([  # TODO: The order of columns should be okay. Check with Yuge & Kan.
            obs["position_history"] / obs["target"].unsqueeze(-1),
            obs["tick_history"] / self.max_step,
            torch.zeros((batch_size, 2), dtype=torch.float32),
        ], dim=-1)

    def forward(self, batch: Batch) -> torch.Tensor:
        inp = cast(WholeDayObs, batch)

        public_state = inp["data_processed"]
        private_state = self._get_private_state(inp)
        seq_len = inp["num_step"].to(torch.long)
        batch_size = public_state.shape[0]  # B

        raw_in = public_state.reshape(batch_size, -1)  # [B, 1440]
        raw_in = torch.cat((torch.zeros(batch_size, 6 * 30), raw_in), dim=-1)  # [B, 1620]
        raw_in = raw_in.reshape(-1, 30, 6).transpose(1, 2)  # [B * 9, 6, 30]
        dnn_in = private_state.reshape(batch_size, 9, -1)  # [B, 9, 2]
        cnn_out = self.cnn(raw_in).view(batch_size, 9, -1)  # [B, 9, 3 * 28]
        assert not torch.isnan(cnn_out).any()
        rnn_in = self.raw_fc(cnn_out)  # [B, 9, 64]
        assert not torch.isnan(rnn_in).any()
        rnn2_in = self.dnn(dnn_in)  # [B, 9, 64]
        assert not torch.isnan(rnn2_in).any()
        rnn2_out = self.rnn2(rnn2_in)[0]  # [B, 9, hidden_size]
        assert not torch.isnan(rnn2_out).any()
        rnn_out = self.rnn(rnn_in)[0]  # [B, 9, hidden_size]
        assert not torch.isnan(rnn_out).any()
        rnn_out = rnn_out[torch.arange(rnn_out.size(0)), seq_len]  # [B, hidden_size]
        rnn2_out = rnn2_out[torch.arange(rnn2_out.size(0)), seq_len]  # [B, hidden_size]
        fc_in = torch.cat((rnn_out, rnn2_out), dim=-1)  # [B, hidden_size * 2]
        return self.fc(fc_in)  # [B, output_dim]
