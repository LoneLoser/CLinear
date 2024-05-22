import torch
from torch import nn
from einops import rearrange
from math import ceil

def get_activation(name):
    return {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU
    }[name]


class LongTermPeriodConv(nn.Module):

    def __init__(self, in_channels, out_channels, drop_prob, activation):
        super(LongTermPeriodConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_prob)
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, (1, 1), stride=1)

    def forward(self, x):
        return self.block(x) + self.conv1x1(x)


class CovariateConv(nn.Module):

    def __init__(self, n_vars, drop_prob, activation):
        super(CovariateConv, self).__init__()
        in_channels, out_channels = 1, n_vars
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (out_channels, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Dropout(drop_prob)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (out_channels, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_prob)
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, (out_channels, 1), stride=1)

    def forward(self, x):
        y = self.block1(x)
        y = rearrange(y, 'b n 1 l -> b 1 n l')
        y = self.block2(y) + self.conv1x1(x)
        return rearrange(y, 'b n 1 l -> b 1 n l')


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, drop_prob, activation):
        super(LinearLayer, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(out_features, out_features),
            nn.Dropout(drop_prob)
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.block(x) + self.linear(x)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.lookback_len = args.seq_len  # lookback length L
        self.horizon_len = args.pred_len  # prediction length H
        self.period = args.period
        self.channels = list(map(int, args.channels.split(',')))
        self.lb_period_num = ceil(self.lookback_len / self.period)
        self.hd_period_num = ceil(self.horizon_len / self.period)
        self.feedback_len = args.label_len
        self.fb_period_num = ceil(self.feedback_len / self.period)
        self.drop_prob = args.dropout
        self.n_vars = args.enc_in
        self.covar_layers = args.covar_layers
        self.conv_activation = get_activation(args.conv_activation)
        self.fc_activation = get_activation(args.fc_activation)

        self.conv_blocks = nn.Sequential(*[
            LongTermPeriodConv(self.channels[i], self.channels[i + 1], self.drop_prob, self.conv_activation) for i in range(len(self.channels) - 1)
        ])

        # if self.covar_layers > 0:
        #     self.covar_blocks = nn.Sequential(*[
        #         CovariateConv(self.n_vars, self.drop_prob, self.conv_activation) for _ in range(self.covar_layers)
        #     ])
        #     self.covar_linear = LinearLayer(self.feedback_len, 1, self.drop_prob, self.fc_activation)

        self.period_linear = LinearLayer(self.channels[-1] * self.lb_period_num, 1, self.drop_prob, self.fc_activation)
        self.lb_term_linear = LinearLayer(self.channels[-1] * self.period, 1, self.drop_prob, self.fc_activation)
        self.hd_term_linear = LinearLayer(self.lb_period_num, self.hd_period_num, self.drop_prob, self.fc_activation)
        self.residual_linear = LinearLayer(self.feedback_len, self.feedback_len + self.horizon_len, self.drop_prob, self.fc_activation)
        self.noise_linear = LinearLayer(self.feedback_len, self.horizon_len, self.drop_prob, self.fc_activation)

    def forward(self, batch_y):
        # batch_y: [Batch, Lookback length, Vars]
        y_lookback = rearrange(batch_y, 'b l n -> (b n) l')

        y_lookback_view = rearrange(y_lookback, 'b (l p) -> b 1 l p', l=self.lb_period_num, p=self.period)  # [b,1,l/p,p]
        y_conv_feat = self.conv_blocks(y_lookback_view)

        y_conv_period = rearrange(y_conv_feat, 'b c l p -> b p (c l)')
        y_period_feat = self.period_linear(y_conv_period)  # 周期规律[-1,p,1]
        y_period_feat = y_period_feat.squeeze(-1)
        y_feedback_period = y_period_feat.repeat(1, self.fb_period_num)[:, -self.feedback_len:]  # [-1,f]
        y_horizon_period = y_period_feat.repeat(1, self.hd_period_num)[:, :self.horizon_len]  # [-1,h]

        y_conv_term = rearrange(y_conv_feat, 'b c l p -> b l (c p)')
        y_lb_term_feat = self.lb_term_linear(y_conv_term)  # 长期趋势[-1,l/p,1]
        y_lb_term_feat = y_lb_term_feat.squeeze(-1)
        y_hb_term_feat = self.hd_term_linear(y_lb_term_feat)  # [-1,h/p]
        y_lb_term_feat = y_lb_term_feat.unsqueeze(-1).repeat(1, 1, self.period).flatten(1)
        y_hb_term_feat = y_hb_term_feat.unsqueeze(-1).repeat(1, 1, self.period).flatten(1)
        y_feedback_term = y_lb_term_feat[:, -self.feedback_len:]  # [-1,f]
        y_horizon_term = y_hb_term_feat[:, :self.horizon_len]  # [-1,h]

        y_residual_feat = self.residual_linear(y_lookback[:, -self.feedback_len:])  # 短期趋势[-1,f]
        y_feedback_residual = y_residual_feat[:, :self.feedback_len]
        y_horizon_residual = y_residual_feat[:, self.feedback_len:]

        y_feedback = y_feedback_period + y_feedback_term + y_feedback_residual
        y_horizon = y_horizon_period + y_horizon_term + y_horizon_residual

        # if self.covar_layers > 0:
        #     covar_lookback = rearrange(batch_y, 'b l n -> b 1 n l')[:, :, :, -self.feedback_len:]  # [b,1,n,f]
        #     y_covar_feat = self.covar_blocks(covar_lookback)  # 协方差特征[b,1,n,f]
        #
        #     y_covar_feat = self.covar_linear(y_covar_feat)  # [b,1,n,1]
        #     y_covar_feat = y_covar_feat.squeeze(1).repeat(1, 1, self.feedback_len + self.horizon_len)
        #     y_covar_feat = rearrange(y_covar_feat, 'b n l -> (b n) l')
        #     y_feedback_covar = y_covar_feat[:, :self.feedback_len]  # [-1,f]
        #     y_horizon_covar = y_covar_feat[:, self.feedback_len:]  # [-1,h]
        #
        #     y_feedback += y_feedback_covar
        #     y_horizon += y_horizon_covar

        y_feedback_noise = y_feedback - y_lookback[:, -self.feedback_len:]  # 噪音[-1,f]
        y_horizon_noise = self.noise_linear(y_feedback_noise)  # [-1,h]

        pred = y_horizon + y_horizon_noise
        pred = rearrange(pred, '(b n) h -> b h n', n=self.n_vars)
        return pred

