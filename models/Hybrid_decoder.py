import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vmamba import VSSBlock, LayerNorm2d, Permute

class CNNToMambaAdapter(nn.Module):

    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.adapt = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.adapt(x)

class RepConvBN(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.deploy = False
        self.rbr_dense = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False), nn.BatchNorm2d(out_ch))
        self.rbr_identity = nn.BatchNorm2d(in_ch) if in_ch == out_ch and stride == 1 else None
        self.rbr_reparam = None

    def forward(self, x):
        if self.deploy and self.rbr_reparam is not None:
            return self.rbr_reparam(x)
        idt = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + idt

    def _fuse_bn(self, branch):
        if branch is None:
            return (0, 0)
        if isinstance(branch, nn.Sequential):
            conv, bn = (branch[0], branch[1])
            kernel = conv.weight
        else:
            bn = branch
            kdim = self.in_ch
            kernel = torch.zeros(kdim, kdim, 3, 3, device=bn.weight.device)
            for i in range(kdim):
                kernel[i, i, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return (kernel * t, bn.bias - bn.running_mean * bn.weight / std)

    @staticmethod
    def _pad_1x1(k):
        if isinstance(k, int):
            return 0
        return F.pad(k, [1, 1, 1, 1])

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn(self.rbr_dense)
        k1, b1 = self._fuse_bn(self.rbr_1x1)
        kid, bid = self._fuse_bn(self.rbr_identity)
        return (k3 + self._pad_1x1(k1) + kid, b3 + b1 + bid)

    @torch.no_grad()
    def reparameterize(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, 3, self.stride, 1, bias=True).to(kernel.device)
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)
        for attr in ['rbr_dense', 'rbr_1x1', 'rbr_identity']:
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.rep1 = RepConvBN(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.rep2 = RepConvBN(out_channels, out_channels, 1)

    def forward(self, x):
        out = self.relu(self.rep1(x))
        out = self.rep2(out)
        return self.relu(out + x)

class DirectedGatedFusion(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.gate_conv = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, bias=True)

    def forward(self, pre_f, post_f):
        gate = torch.sigmoid(self.gate_conv(torch.cat([pre_f, post_f], dim=1)))
        post_enhanced = post_f + gate * pre_f
        return torch.cat([pre_f, post_enhanced], dim=1)

class CrossLevelGate(nn.Module):

    def __init__(self, ch=128, reduction=4):
        super().__init__()
        hid = max(ch // reduction, 8)
        self.gate = nn.Sequential(nn.Conv2d(ch, hid, kernel_size=1, bias=False), nn.Conv2d(hid, hid, kernel_size=3, padding=1, groups=hid, bias=False), nn.GELU(), nn.Conv2d(hid, ch, kernel_size=1, bias=True))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 4.0)

    def forward(self, high_up, low):
        return low * torch.sigmoid(self.gate(high_up))

def make_vss_path(in_ch, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
    return nn.Sequential(nn.Conv2d(in_ch, 128, kernel_size=1, bias=False), Permute(0, 2, 3, 1), VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=False, ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer, ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'], forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'], gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']), Permute(0, 3, 1, 2))

def make_dw_path(in_ch):
    return nn.Sequential(nn.Conv2d(in_ch, 128, kernel_size=1, bias=False), nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False), nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.GELU())

def make_fuse_layer():
    return nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

class MaskGuideAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.adapt = nn.Sequential(nn.Conv2d(128 + 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, feat, prev_mask, target_size):
        mask_up = F.interpolate(prev_mask, size=target_size, mode='bilinear', align_corners=False)
        return self.adapt(torch.cat([feat, mask_up], dim=1))

class EdgeBranch(nn.Module):

    def __init__(self, d1=24, d2=32, mid_ch=32):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(d1, mid_ch, kernel_size=1, bias=False), nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU())
        self.branch2 = nn.Sequential(nn.Conv2d(d2, mid_ch, kernel_size=1, bias=False), nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU(), nn.Conv2d(mid_ch, 1, kernel_size=1))

    def forward(self, diff1, diff2):
        e1 = self.branch1(diff1)
        e2 = self.branch2(diff2)
        e2 = F.interpolate(e2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([e1, e2], dim=1))

class ChangeDecoder_Hybrid(nn.Module):

    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, use_cross_gate=True, use_mask_guide=False, **kwargs):
        super().__init__()
        self.use_cross_gate = use_cross_gate
        self.use_mask_guide = use_mask_guide
        d1, d2, d3, d4 = encoder_dims
        self.cnn_adapt_2 = CNNToMambaAdapter(in_ch=d2)
        self.cnn_adapt_1 = CNNToMambaAdapter(in_ch=d1)
        vss_kw = dict(norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, **kwargs)
        self.gate4 = DirectedGatedFusion(d4)
        self.path4_cat = make_vss_path(d4 * 2, **vss_kw)
        self.path4_diff = make_vss_path(d4, **vss_kw)
        self.fuse_layer_4 = make_fuse_layer()
        self.aux_head4 = nn.Conv2d(128, 2, kernel_size=1)
        if self.use_mask_guide:
            self.mask_guide3 = MaskGuideAdapter()
        self.gate3 = DirectedGatedFusion(d3)
        self.path3_cat = make_vss_path(d3 * 2, **vss_kw)
        self.path3_diff = make_vss_path(d3, **vss_kw)
        self.fuse_layer_3 = make_fuse_layer()
        self.smooth_layer_3 = ResBlock(128, 128)
        self.aux_head3 = nn.Conv2d(128, 2, kernel_size=1)
        if self.use_mask_guide:
            self.mask_guide2 = MaskGuideAdapter()
        self.gate2 = DirectedGatedFusion(d2)
        self.path2_cat = make_dw_path(d2 * 2)
        self.path2_diff = make_dw_path(d2)
        self.fuse_layer_2 = make_fuse_layer()
        self.smooth_layer_2 = ResBlock(128, 128)
        self.aux_head2 = nn.Conv2d(128, 2, kernel_size=1)
        if self.use_mask_guide:
            self.mask_guide1 = MaskGuideAdapter()
        self.gate1 = DirectedGatedFusion(d1)
        self.path1_cat = make_dw_path(d1 * 2)
        self.path1_diff = make_dw_path(d1)
        self.fuse_layer_1 = make_fuse_layer()
        self.smooth_layer_1 = ResBlock(128, 128)
        if self.use_cross_gate:
            self.xgate3 = CrossLevelGate(128)
            self.xgate2 = CrossLevelGate(128)
            self.xgate1 = CrossLevelGate(128)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _gated_upsample_add(self, high, low, gate):
        high_up = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        return high_up + gate(high_up, low)

    def _calibrated_diff(self, pre_f, post_f):
        eps = 1e-06
        pre_n = F.normalize(pre_f, p=2, dim=1)
        post_n = F.normalize(post_f, p=2, dim=1)
        cos = (pre_n * post_n).sum(dim=1, keepdim=True)
        dissim = (1.0 - cos) * 0.5
        return torch.abs(pre_f - post_f) * (1.0 + dissim)

    def _two_path_process(self, gate, path_cat, path_diff, fuse, pre_f, post_f):
        p1 = path_cat(gate(pre_f, post_f))
        p2 = path_diff(self._calibrated_diff(pre_f, post_f))
        return fuse(torch.cat([p1, p2], dim=1))

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, RepConvBN):
                m.reparameterize()

    def forward(self, pre_features, post_features):
        pre_f1, pre_f2, pre_f3, pre_f4 = pre_features
        post_f1, post_f2, post_f3, post_f4 = post_features
        pre_f2_a = self.cnn_adapt_2(pre_f2)
        post_f2_a = self.cnn_adapt_2(post_f2)
        pre_f1_a = self.cnn_adapt_1(pre_f1)
        post_f1_a = self.cnn_adapt_1(post_f1)
        p4 = self._two_path_process(self.gate4, self.path4_cat, self.path4_diff, self.fuse_layer_4, pre_f4, post_f4)
        _need_aux = self.training or self.use_mask_guide
        mask4 = self.aux_head4(p4) if _need_aux else None
        p3 = self._two_path_process(self.gate3, self.path3_cat, self.path3_diff, self.fuse_layer_3, pre_f3, post_f3)
        if self.use_cross_gate:
            p3 = self._gated_upsample_add(p4, p3, self.xgate3)
        else:
            p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
        if self.use_mask_guide:
            p3 = self.mask_guide3(p3, mask4, target_size=p3.shape[-2:])
        mask3 = self.aux_head3(p3) if _need_aux else None
        p2 = self._two_path_process(self.gate2, self.path2_cat, self.path2_diff, self.fuse_layer_2, pre_f2_a, post_f2_a)
        if self.use_cross_gate:
            p2 = self._gated_upsample_add(p3, p2, self.xgate2)
        else:
            p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
        if self.use_mask_guide:
            p2 = self.mask_guide2(p2, mask3, target_size=p2.shape[-2:])
        mask2 = self.aux_head2(p2) if _need_aux else None
        p1 = self._two_path_process(self.gate1, self.path1_cat, self.path1_diff, self.fuse_layer_1, pre_f1_a, post_f1_a)
        if self.use_cross_gate:
            p1 = self._gated_upsample_add(p2, p1, self.xgate1)
        else:
            p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        if self.use_mask_guide:
            p1 = self.mask_guide1(p1, mask2, target_size=p1.shape[-2:])
        return (p1, mask2, mask3, mask4)
