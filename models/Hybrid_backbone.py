import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vendor.vmamba import VSSBlock, LayerNorm2d, Permute

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True)), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True)), nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True)), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_Stages0123(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, stride=2)
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2]]
        all_blocks = []
        in_ch = 32
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                all_blocks.append(InvertedResidual(in_ch, c, stride, expand_ratio=t))
                in_ch = c
        self.blocks = nn.ModuleList(all_blocks)

    def load_mobilenet_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            elif 'model' in ckpt:
                sd = ckpt['model']
            else:
                sd = ckpt
        else:
            sd = ckpt
        new_sd = {}
        for k, v in sd.items():
            if not k.startswith('features.'):
                continue
            parts = k.split('.')
            feat_idx = int(parts[1])
            rest = '.'.join(parts[2:])
            if feat_idx == 0:
                new_sd[f'stem.{rest}'] = v
            elif 1 <= feat_idx <= 10:
                new_sd[f'blocks.{feat_idx - 1}.{rest}'] = v
        if not new_sd:
            raise ValueError('No MobileNetV2 feature weights were found in the checkpoint')
        model_sd = self.state_dict()
        for key, value in new_sd.items():
            if key in model_sd and value.shape != model_sd[key].shape:
                raise ValueError(f'MobileNetV2 weight shape mismatch: {key}')
        self.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        feat1 = self.blocks[2](x)
        x = self.blocks[3](feat1)
        x = self.blocks[4](x)
        feat2 = self.blocks[5](x)
        x = self.blocks[6](feat2)
        x = self.blocks[7](x)
        x = self.blocks[8](x)
        feat3 = self.blocks[9](x)
        return (feat1, feat2, feat3)

class ECAAttention(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class CoordAttention(nn.Module):

    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, ch, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.pool_h(x)
        w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = torch.cat([h, w], dim=2)
        hw = self.act(self.bn1(self.conv1(hw)))
        h_, w_ = hw.split([H, W], dim=2)
        w_ = w_.permute(0, 1, 3, 2)
        return x * self.conv_h(h_).sigmoid() * self.conv_w(w_).sigmoid()

class MultiScaleDetailFusion(nn.Module):

    def __init__(self, in_ch1=24, in_ch2=32, in_ch3=64, branch_ch=40, out_ch=120):
        super().__init__()
        assert branch_ch * 3 == out_ch
        self.down1 = nn.Sequential(nn.Conv2d(in_ch1, in_ch1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(in_ch1), nn.ReLU6(inplace=True), nn.Conv2d(in_ch1, in_ch1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(in_ch1), nn.ReLU6(inplace=True))
        self.proj1 = nn.Sequential(nn.Conv2d(in_ch1, branch_ch, 1, bias=False), nn.BatchNorm2d(branch_ch), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(in_ch2, in_ch2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(in_ch2), nn.ReLU6(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(in_ch2, branch_ch, 1, bias=False), nn.BatchNorm2d(branch_ch), nn.GELU())
        self.proj3 = nn.Sequential(nn.Conv2d(in_ch3, branch_ch, 1, bias=False), nn.BatchNorm2d(branch_ch), nn.GELU())
        self.fuse_conv = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
        self.attention = CoordAttention(out_ch)

    def forward(self, feat1, feat2, feat3):
        p1 = self.proj1(self.down1(feat1))
        p2 = self.proj2(self.down2(feat2))
        p3 = self.proj3(feat3)
        fused = torch.cat([p1, p2, p3], dim=1)
        fused = self.fuse_conv(fused)
        fused = self.attention(fused)
        return fused

class HaarDWT2D(nn.Module):

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return (ll, lh, hl, hh)

class HaarIDWT2D(nn.Module):

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return out

class SoftThresholdDenoise(nn.Module):

    def __init__(self, ch, exp=2):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.norm = nn.GroupNorm(ch, ch)
        self.pw1 = nn.Conv2d(ch, exp * ch, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(exp * ch, ch, 1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        t = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return torch.sign(x) * torch.relu(torch.abs(x) - t)

class LFMWB(nn.Module):

    def __init__(self, in_ch=120, mid_ch=192, use_daf=False):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.low_freq_gate = nn.Conv2d(in_ch, in_ch, kernel_size=1, groups=in_ch, bias=True)
        nn.init.zeros_(self.low_freq_gate.weight)
        nn.init.constant_(self.low_freq_gate.bias, 4.0)

        def make_hf_gate(ch):
            return nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.Conv2d(ch, ch, 1, bias=True))
        self.hf_gate_lh = make_hf_gate(in_ch)
        self.hf_gate_hl = make_hf_gate(in_ch)
        self.hf_gate_hh = make_hf_gate(in_ch)
        self.beta_lh = nn.Parameter(torch.zeros(1))
        self.beta_hl = nn.Parameter(torch.zeros(1))
        self.beta_hh = nn.Parameter(torch.zeros(1))
        self.alpha_logit = nn.Parameter(torch.full((1, in_ch, 1, 1), -2.0))
        self.denoise_lh = SoftThresholdDenoise(in_ch)
        self.denoise_hl = SoftThresholdDenoise(in_ch)
        self.denoise_hh = SoftThresholdDenoise(in_ch)
        self.expand_conv = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_ch)
        self.expand_act = nn.GELU()
        self.use_daf = use_daf
        if use_daf:
            self.daf_scale_lh = nn.Parameter(torch.zeros(1))
            self.daf_scale_hl = nn.Parameter(torch.zeros(1))
            self.daf_scale_hh = nn.Parameter(torch.zeros(1))

    def forward(self, x, delta_prior=None):
        ll, lh, hl, hh = self.dwt(x)
        w_low = torch.sigmoid(self.low_freq_gate(ll))
        ll_new = ll * w_low
        lh_enh = self.beta_lh * lh * torch.tanh(self.hf_gate_lh(lh))
        hl_enh = self.beta_hl * hl * torch.tanh(self.hf_gate_hl(hl))
        hh_enh = self.beta_hh * hh * torch.tanh(self.hf_gate_hh(hh))
        if self.use_daf and delta_prior is not None:
            dp = F.interpolate(delta_prior, size=ll.shape[-2:], mode='bilinear', align_corners=False)
            lh_enh = lh_enh * (1.0 + self.daf_scale_lh * dp)
            hl_enh = hl_enh * (1.0 + self.daf_scale_hl * dp)
            hh_enh = hh_enh * (1.0 + self.daf_scale_hh * dp)
        lh_new = lh + lh_enh
        hl_new = hl + hl_enh
        hh_new = hh + hh_enh
        lh_new = self.denoise_lh(lh_new)
        hl_new = self.denoise_hl(hl_new)
        hh_new = self.denoise_hh(hh_new)
        x_wavelet = self.idwt(ll_new, lh_new, hl_new, hh_new)
        alpha = torch.sigmoid(self.alpha_logit)
        x_fused = x + alpha * x_wavelet
        return self.expand_act(self.expand_bn(self.expand_conv(x_fused)))

class DifferenceAwareModule(nn.Module):

    def __init__(self, in_ch, reduction=4):
        super().__init__()
        mid_ch = max(8, in_ch // reduction)
        self.compress = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU(), nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True))

    def forward(self, pre, post):
        pre_n = F.normalize(pre, p=2, dim=1)
        post_n = F.normalize(post, p=2, dim=1)
        per_ch_sim = pre_n * post_n
        per_ch_dissim = (1.0 - per_ch_sim) * 0.5
        delta_prior = torch.sigmoid(self.compress(per_ch_dissim))
        return delta_prior

class Stage3Adapter(nn.Module):

    def __init__(self, in_ch=192, out_ch=192):
        super().__init__()
        assert in_ch == out_ch, 'Stage3Adapter requires equal input and output channels'
        self.norm = nn.LayerNorm(in_ch)
        self.act = nn.GELU()
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.act(self.norm(x))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dw_bn(self.dw_conv(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class PatchMerging2D(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_ch)
        self.reduction = nn.Linear(4 * in_ch, 2 * in_ch, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)

class VSSStage(nn.Module):

    def __init__(self, hidden_dim, num_blocks, norm_layer, ssm_act_layer, mlp_act_layer, drop_path_rate=0.1, **kwargs):
        super().__init__()
        dpr = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([VSSBlock(hidden_dim=hidden_dim, drop_path=dpr[i], norm_layer=norm_layer, channel_first=False, ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer, ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'], forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'], gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']) for i in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class VSSMStages34(nn.Module):

    def __init__(self, norm_layer, ssm_act_layer, mlp_act_layer, pretrained=None, drop_path_rate=0.1, **kwargs):
        super().__init__()
        self.stage3 = VSSStage(hidden_dim=192, num_blocks=4, norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, drop_path_rate=drop_path_rate, **kwargs)
        self.norm3 = nn.LayerNorm(192)
        self.patch_merging = PatchMerging2D(in_ch=192)
        self.stage4 = VSSStage(hidden_dim=384, num_blocks=2, norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, drop_path_rate=drop_path_rate, **kwargs)
        self.norm4 = nn.LayerNorm(384)
        if pretrained is not None:
            self.load_vssm_pretrained(pretrained)

    def load_vssm_pretrained(self, ckpt_path, key='model'):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt[key] if key in ckpt else ckpt
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith('layers.2.blocks.'):
                parts = k.split('.')
                block_idx = int(parts[3])
                if block_idx < 4:
                    rest = '.'.join(parts[4:])
                    mapped_sd[f'stage3.blocks.{block_idx}.{rest}'] = v
            elif k.startswith('layers.3.blocks.'):
                parts = k.split('.')
                block_idx = int(parts[3])
                rest = '.'.join(parts[4:])
                mapped_sd[f'stage4.blocks.{block_idx}.{rest}'] = v
        model_sd = self.state_dict()
        new_sd = {}
        for k, v_ckpt in mapped_sd.items():
            if k not in model_sd:
                continue
            v_model = model_sd[k]
            if v_ckpt.shape == v_model.shape:
                new_sd[k] = v_ckpt.clone()
            elif len(v_ckpt.shape) == len(v_model.shape) and all(a >= b for a, b in zip(v_ckpt.shape, v_model.shape)):
                slices = tuple((slice(0, s) for s in v_model.shape))
                new_sd[k] = v_ckpt[slices].clone()
        if not new_sd:
            raise ValueError('No compatible VSSM weights were found in the checkpoint')
        self.load_state_dict(new_sd, strict=False)

    def forward(self, z):
        o3 = self.stage3(z)
        o3n = self.norm3(o3)
        out3 = o3n.permute(0, 3, 1, 2).contiguous()
        x4 = self.patch_merging(o3)
        o4 = self.stage4(x4)
        o4n = self.norm4(o4)
        out4 = o4n.permute(0, 3, 1, 2).contiguous()
        return (out3, out4)

class HybridEncoder(nn.Module):
    dims = [24, 32, 192, 384]
    channel_first = True

    def __init__(self, norm_layer, ssm_act_layer, mlp_act_layer, mobilenet_pretrained=None, vssm_pretrained=None, msdf_out_ch=120, lfmwb_mid_ch=192, drop_path_rate=0.1, exchange_interval=2, **kwargs):
        super().__init__()
        self.exchange_interval = exchange_interval
        self.cnn = MobileNetV2_Stages0123()
        if mobilenet_pretrained is not None:
            self.cnn.load_mobilenet_pretrained(mobilenet_pretrained)
        self.msdf = MultiScaleDetailFusion(in_ch1=24, in_ch2=32, in_ch3=64, branch_ch=msdf_out_ch // 3, out_ch=msdf_out_ch)
        self.dam = DifferenceAwareModule(in_ch=msdf_out_ch, reduction=4)
        self.lfmwb = LFMWB(in_ch=msdf_out_ch, mid_ch=lfmwb_mid_ch, use_daf=True)
        self.stage3_adapter = Stage3Adapter(in_ch=lfmwb_mid_ch, out_ch=lfmwb_mid_ch)
        self.vss = VSSMStages34(norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, pretrained=vssm_pretrained, drop_path_rate=drop_path_rate, **kwargs)
        self.cnn_residual_proj = nn.Sequential(nn.Conv2d(64, lfmwb_mid_ch, kernel_size=1, bias=False), nn.BatchNorm2d(lfmwb_mid_ch))

    def _channel_exchange(self, pre, post):
        C = pre.shape[-1]
        p = max(int(self.exchange_interval), 1)
        idx = torch.arange(C, device=pre.device)
        swap_mask = (idx % p == 0).view(1, 1, 1, C)
        pre_x = torch.where(swap_mask, post, pre)
        post_x = torch.where(swap_mask, pre, post)
        return (pre_x, post_x)

    def forward(self, pre_data, post_data):
        pre_f1, pre_f2, pre_f3_cnn = self.cnn(pre_data)
        post_f1, post_f2, post_f3_cnn = self.cnn(post_data)
        pre_msdf = self.msdf(pre_f1, pre_f2, pre_f3_cnn)
        post_msdf = self.msdf(post_f1, post_f2, post_f3_cnn)
        delta_prior = self.dam(pre_msdf, post_msdf)
        pre_z_mid = self.lfmwb(pre_msdf, delta_prior)
        post_z_mid = self.lfmwb(post_msdf, delta_prior)
        pre_z = self.stage3_adapter(pre_z_mid)
        post_z = self.stage3_adapter(post_z_mid)
        pre_o3 = self.vss.stage3(pre_z)
        post_o3 = self.vss.stage3(post_z)
        pre_feat3 = self.vss.norm3(pre_o3).permute(0, 3, 1, 2).contiguous()
        post_feat3 = self.vss.norm3(post_o3).permute(0, 3, 1, 2).contiguous()
        pre_feat3 = pre_feat3 + self.cnn_residual_proj(pre_f3_cnn)
        post_feat3 = post_feat3 + self.cnn_residual_proj(post_f3_cnn)
        pre_x4 = self.vss.patch_merging(pre_o3)
        post_x4 = self.vss.patch_merging(post_o3)
        pre_x4, post_x4 = self._channel_exchange(pre_x4, post_x4)
        pre_o4 = self.vss.stage4(pre_x4)
        post_o4 = self.vss.stage4(post_x4)
        pre_feat4 = self.vss.norm4(pre_o4).permute(0, 3, 1, 2).contiguous()
        post_feat4 = self.vss.norm4(post_o4).permute(0, 3, 1, 2).contiguous()
        return ([pre_f1, pre_f2, pre_feat3, pre_feat4], [post_f1, post_f2, post_feat3, post_feat4], delta_prior)
