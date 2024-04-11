import torch.nn as nn
import torch
import math


class TIE(nn.Module):
    def __init__(self, d_model, dev):
        super().__init__()
        assert d_model % 6 == 0
        # [d/4]
        self.dev = dev
        self.theta = torch.exp(torch.arange(0, d_model, 6, device='cpu') * -(math.log(10000.0) / d_model)).to(self.dev)
        self.drop = nn.Dropout(p=0.2)

    def get_position(self, t_d, t_m, t_h):
        # [b, n, d/4]
        pos_day = t_d.unsqueeze(-1) * self.theta
        pos_hour = t_h.unsqueeze(-1) * self.theta
        pos_mon = t_m.unsqueeze(-1) * self.theta
        cos_day = torch.cos(pos_day)
        sin_day = torch.sin(pos_day)
        cos_mon = torch.cos(pos_mon)
        sin_mon = torch.sin(pos_mon)
        cos_hour = torch.cos(pos_hour)
        sin_hour = torch.sin(pos_hour)
        return cos_day, sin_day, cos_mon, sin_mon, cos_hour, sin_hour

    def forward(self, x, t_d, t_m, t_h):
        cos_day, sin_day, cos_mon, sin_mon, cos_hour, sin_hour = self.get_position(t_d, t_m, t_h)
        # [b, 1, n, d/4]
        x_1, x_2, x_3, x_4, x_5, x_6 = x[..., 0::6], x[..., 1::6], x[..., 2::6], x[..., 3::6], x[..., 4::6], x[..., 5::6]
        x_day_1 = x_1 * cos_day + x_2 * sin_day
        x_day_2 = x_2 * cos_day - x_1 * sin_day
        x_month_3 = x_3 * cos_mon + x_4 * sin_mon
        x_month_4 = x_4 * cos_mon - x_3 * sin_mon
        x_hour_5 = x_5 * cos_hour + x_6 * sin_hour
        x_hour_6 = x_6 * cos_hour - x_5 * sin_hour
        x = torch.cat([x_day_1, x_day_2, x_month_3, x_month_4, x_hour_5, x_hour_6], dim=-1)

        return self.drop(x)