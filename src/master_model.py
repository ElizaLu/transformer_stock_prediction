import torch
from torch import nn
from master_layers import PositionalEncoding, TAttention, SAttention, TemporalAttention, Gate


class MASTER(nn.Module):
    def __init__(
        self,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        T_dropout_rate,
        S_dropout_rate,
        gate_input_start_index,
        gate_input_end_index,
        beta,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index

        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.t_attn = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.s_attn = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temp_attn = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        x = self.feature_layer(src)
        x = self.pos_enc(x)
        x = self.t_attn(x)
        x = self.s_attn(x)
        x = self.temp_attn(x)
        x = self.decoder(x).squeeze(-1)
        return x