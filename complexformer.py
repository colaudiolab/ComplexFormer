import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))  # args.n_series: number of series
        self.beta = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, batch_x, mode="norm", dec_inp=None):
        if mode == "norm":
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == "denorm":
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        self.avg = torch.mean(batch_x, axis=1, keepdim=True).detach()  # b*1*d
        self.var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d

    def forward_process(self, batch_input):
        temp = (batch_input - self.avg) / torch.sqrt(self.var + 1e-8)
        return temp.mul(self.gamma) + self.beta

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.var + 1e-8) + self.avg



class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ComplexAttention(nn.Module):
    def __init__(self, feature_dim, n_heads):
        super(ComplexAttention, self).__init__()
        self.feature_dim = feature_dim
        d_keys = feature_dim // n_heads
        d_values = feature_dim // n_heads

        # 定义用于复数的注意力线性层
        self.query = nn.Linear(feature_dim, d_keys * n_heads)
        self.key = nn.Linear(feature_dim, d_keys * n_heads)
        self.value = nn.Linear(feature_dim, d_values * n_heads)
        self.proj = nn.Linear(d_values * n_heads, feature_dim)
        self.n_heads = n_heads

    def forward(self, X):
        # x shape is (batch, seq_len, feature)
        X = torch.fft.ifft(X, dim=-1).real
        H = self.n_heads
        B, L, N = X.shape
        Q, K, V = self.query(X), self.key(X), self.value(X)
        Q, K, V = Q.view(B, L, H, -1), K.view(B, L, H, -1), V.view(B, L, H, -1)

        scale = self.feature_dim ** 0.5
        attention_score = torch.einsum("blhe,bshe->bhls", Q, K) / scale
        attention_score.masked_fill_(TriangularCausalMask(B, L, device=Q.device).mask, -torch.inf)
        attention_score = torch.softmax(attention_score, dim=-1)

        output = torch.einsum("bhls,bshd->blhd", attention_score, V).contiguous()
        output = output.view(B, L, -1)
        output = self.proj(output)

        return output


class ComplexEncoder(nn.Module):
    def __init__(self, feature_dim, n_heads, dropout=0.25):
        super(ComplexEncoder, self).__init__()
        self.feature_dim = feature_dim

        self.attention = ComplexAttention(feature_dim, n_heads)
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        new_x = self.attention(x)  # 若干噪声/周期/趋势  可以学，时域和频域各自发挥优势。
        gated = F.sigmoid(self.alpha(new_x)) + 1
        x = gated * x + (1 - gated) * self.dropout(new_x)   # 不需要norm，原始信号加和为1.

        return x


# class ComplexFormer(nn.Module):
#     def __init__(self, input_dim, seq_len, pred_len, d_model=128, e_layers=1, dropout=0.25, n_heads=4, **kwargs):
class Model(nn.Module):
    # def __init__(self, input_dim, seq_len, pred_len, d_model=128, e_layers=1, dropout=0.25, n_heads=4, **kwargs):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.e_layers = configs.e_layers
        self.dropout = nn.Dropout(configs.dropout)
        self.fc = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.pred_len = configs.pred_len
        self.proj = nn.Linear(configs.d_model, configs.pred_len)
        self.task_name = configs.task_name
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            self.real_fc = nn.Sequential(
            nn.Linear(configs.seq_len, configs.seq_len),
            nn.ReLU(),
        )
        # freq
        self.fft_blocks = nn.ModuleList([ComplexEncoder(configs.d_model, configs.n_heads) for _ in range(configs.e_layers)])
        self.enc_in = configs.enc_in

        if configs.use_gpu:
            self.device="cuda"
        else:
            self.device="cpu"    
        
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, std=1e-3)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)


    def base_forward(self, x):
        # (B, L, F)
        B, L, N = x.size()

        x = x.permute(0, 2, 1)  # (B, F, L)
        x = self.fc(x)
        x = torch.fft.fft(x, dim=-1)
        
        # Frequency
        for i in range(self.e_layers):
            x = self.fft_blocks[i](x)  # (B, F, H)

        x = torch.fft.ifft(x, dim=-1).real
        # merge
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        return x
    
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        
        # 使用revin
        revin_layer = RevIN(self.enc_in).to(self.device)
        x_enc = revin_layer(x_enc,"norm")

        dec_out  = self.base_forward(x_enc[0])

        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        dec_out = revin_layer(dec_out,"denorm").to(self.device)
        
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        # means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        # means = means.unsqueeze(1).detach()
        # x_enc = x_enc - means
        # x_enc = x_enc.masked_fill(mask == 0, 0)
        # stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
        #                    torch.sum(mask == 1, dim=1) + 1e-5)
        # stdev = stdev.unsqueeze(1).detach()
        # x_enc /= stdev

        revin_layer = RevIN(self.enc_in).to(self.device)
        x_enc = revin_layer(x_enc,"norm")

        dec_out  = self.base_forward(x_enc[0])

        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        dec_out = revin_layer(dec_out,"denorm").to(self.device)
        
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        revin_layer = RevIN(self.enc_in).to(self.device)
        x_enc = revin_layer(x_enc,"norm")
        dec_out  = self.base_forward(x_enc[0])
        
    
        dec_out = revin_layer(dec_out,"denorm").to(self.device)

        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        dec_out  = self.base_forward(x_enc)
        
        # Decoder
        output = self.flatten(dec_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
    
    
    
    
    
    
    
    
# 在复数域上与attention架构
#    
    
    
    