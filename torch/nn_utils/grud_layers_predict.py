import torch
import torch.nn as nn

class GRUDCell(nn.Module):
    def __init__(self, input_size, hidden_size, x_imputation='zero',
                 input_decay='exp_relu', hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None):
        super(GRUDCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        assert x_imputation in ['zero', 'forward', 'raw'], (
            'x_imputation {} argument is not supported.'.format(x_imputation)
        )
        self.x_imputation = x_imputation

        self.input_decay = self._get_activation(input_decay)
        self.hidden_decay = self._get_activation(hidden_decay)
        self.use_decay_bias = use_decay_bias

        assert (feed_masking or masking_decay is None or masking_decay == 'None'), (
            'Mask needs to be fed into GRU-D to enable the mask_decay.'
        )
        self.feed_masking = feed_masking
        if self.feed_masking:
            self.masking_decay = self._get_activation(masking_decay)

        if self.input_decay is not None or self.hidden_decay is not None:
            self.decay_initializer = nn.init.zeros_

        self.input_decay_weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        if self.use_decay_bias:
            self.input_decay_bias = nn.Parameter(torch.Tensor(input_size))

        self.hidden_decay_weight = nn.Parameter(torch.Tensor(input_size, hidden_size))
        if self.use_decay_bias:
            self.hidden_decay_bias = nn.Parameter(torch.Tensor(hidden_size))

        if self.feed_masking:
            self.masking_weight = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
            if self.masking_decay is not None:
                self.masking_decay_weight = nn.Parameter(torch.Tensor(input_size))
                if self.use_decay_bias:
                    self.masking_decay_bias = nn.Parameter(torch.Tensor(input_size))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.input_decay_weight)
        nn.init.orthogonal_(self.hidden_decay_weight)
        if self.use_decay_bias:
            nn.init.zeros_(self.input_decay_bias)
            nn.init.zeros_(self.hidden_decay_bias)
        if self.feed_masking:
            nn.init.orthogonal_(self.masking_weight)
            if self.masking_decay is not None:
                nn.init.orthogonal_(self.masking_decay_weight)
                if self.use_decay_bias:
                    nn.init.zeros_(self.masking_decay_bias)

    def _get_activation(self, activation):
        if activation == 'exp_relu':
            return nn.functional.relu
        elif activation == 'sigmoid':
            return nn.functional.sigmoid
        elif activation == 'tanh':
            return nn.functional.tanh
        else:
            return None

    def forward(self, input_x, input_m, input_s, states=None):
        if states is None:
            h_tm1 = torch.zeros(input_x.size(0), self.hidden_size, dtype=input_x.dtype, device=input_x.device)
            x_keep_tm1 = torch.zeros_like(input_x)
            s_prev_tm1 = torch.zeros(input_x.size(0), 1, dtype=input_x.dtype, device=input_x.device)
        else:
            h_tm1, x_keep_tm1, s_prev_tm1 = states

        input_1m = 1.0 - input_m
        input_d = torch.abs(input_s - s_prev_tm1)

                if self.input_decay is not None:
            x_predict = h_tm1 @ self.input_decay_weight.t()
            if self.use_decay_bias:
                x_predict = x_predict + self.input_decay_bias
        if self.hidden_decay is not None:
            gamma_dh = input_d @ self.hidden_decay_weight
            if self.use_decay_bias:
                gamma_dh = gamma_dh + self.hidden_decay_bias
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_weight
            if self.use_decay_bias:
                gamma_dm = gamma_dm + self.masking_decay_bias
            gamma_dm = self.masking_decay(gamma_dm)

        if self.input_decay is not None:
            x_keep_t = torch.where(input_m, input_x, x_keep_tm1)
            x_t = torch.where(input_m, input_x, x_predict)
        elif self.x_imputation == 'forward':
            x_t = torch.where(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = torch.where(input_m, input_x, torch.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError('No input decay or invalid x_imputation {}.'.format(self.x_imputation))

        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        z_t = x_t @ self.weight_ih + h_tm1d @ self.weight_hh
        r_t = x_t @ self.weight_ir + h_tm1d @ self.weight_hr
        hh_t = x_t @ self.weight_ihh

        if self.feed_masking:
            z_t += m_t @ self.masking_weight[:, :self.hidden_size]
            r_t += m_t @ self.masking_weight[:, self.hidden_size:self.hidden_size * 2]
            hh_t += m_t @ self.masking_weight[:, self.hidden_size * 2:]

        if self.use_bias:
            z_t += self.bias_ih + self.bias_hh
            r_t += self.bias_ir + self.bias_hr
            hh_t += self.bias_ihh

        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)

        h_tm1_r = r_t * h_tm1d
        hh_t = self.activation(hh_t + h_tm1_r @ self.weight_hhh)

        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        s_prev_t = torch.where(input_m, input_s.repeat(1, self.hidden_size), s_prev_tm1)

        return h_t, [h_t, x_keep_t, s_prev_t]

    def extra_repr(self):
        return 'input_size={}, hidden_size={}, x_imputation={}, input_decay={}, hidden_decay={}, use_decay_bias={}, feed_masking={}, masking_decay={}'.format(
            self.input_size, self.hidden_size, self.x_imputation, self.input_decay, self.hidden_decay, self.use_decay_bias,
            self.feed_masking, self.masking_decay
        )