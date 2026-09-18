import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vmamba import LayerNorm2d
from models.Hybrid_backbone import HybridEncoder
from models.Hybrid_decoder import ChangeDecoder_Hybrid

class FreqRefine(nn.Module):

    def __init__(self, channels=128, reduction=4):
        super().__init__()
        self.channels = channels
        self.eps = 1e-06
        self.amp_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.amp_conv.weight)
        nn.init.zeros_(self.amp_conv.bias)
        self.cos_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.cos_conv.weight)
        nn.init.zeros_(self.cos_conv.bias)
        self.sin_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.sin_conv.weight)
        nn.init.zeros_(self.sin_conv.bias)
        self.spatial_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        nn.init.zeros_(self.spatial_dw.weight)
        nn.init.zeros_(self.spatial_dw.bias)
        hidden = max(channels // reduction, 4)
        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(hidden, channels, kernel_size=1, bias=True), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        identity = x
        x_fft = torch.fft.fft2(x, norm='ortho')
        real, imag = (x_fft.real, x_fft.imag)
        amp = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        cos_p = real / (amp + self.eps)
        sin_p = imag / (amp + self.eps)
        amp_refined = amp + self.amp_conv(amp)
        cos_ref = cos_p + self.cos_conv(cos_p)
        sin_ref = sin_p + self.sin_conv(sin_p)
        pnorm = torch.sqrt(cos_ref ** 2 + sin_ref ** 2 + self.eps)
        cos_ref = cos_ref / pnorm
        sin_ref = sin_ref / pnorm
        real_ref = amp_refined * cos_ref
        imag_ref = amp_refined * sin_ref
        x_fft_refined = torch.complex(real_ref, imag_ref)
        branch1 = torch.fft.ifft2(x_fft_refined, norm='ortho').real
        branch2 = self.spatial_dw(x)
        fused = branch1 + branch2
        ca_weight = self.ca_fc(self.ca_pool(fused))
        modulated = fused * ca_weight
        return identity + self.gamma * modulated

class LightUpsampleRefine(nn.Module):

    def __init__(self, in_ch=128, mid_ch=64, out_ch=32):
        super().__init__()
        self.up1_conv = nn.Conv2d(in_ch, mid_ch * 4, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.ref1 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU())
        self.up2_conv = nn.Conv2d(mid_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.ref2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
        self._icnr(self.up1_conv, scale=2)
        self._icnr(self.up2_conv, scale=2)

    @staticmethod
    def _icnr(conv, scale=2):
        ni, nf, h, w = conv.weight.shape
        sub = ni // (scale * scale)
        k = torch.zeros([sub, nf, h, w])
        nn.init.kaiming_normal_(k)
        k = k.repeat_interleave(scale * scale, dim=0)
        with torch.no_grad():
            conv.weight.copy_(k)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        x = self.ref1(self.ps1(self.up1_conv(x)))
        x = self.ref2(self.ps2(self.up2_conv(x)))
        return x

def get_boundary_label(mask, dilation=1):
    m = mask.float().unsqueeze(1)
    lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=mask.device).view(1, 1, 3, 3)
    lap = F.conv2d(m, lap_kernel, padding=1)
    boundary = (lap.abs() > 0.5).float()
    if dilation > 1:
        dilate_kernel = torch.ones(1, 1, dilation * 2 + 1, dilation * 2 + 1, device=mask.device)
        boundary = (F.conv2d(boundary, dilate_kernel, padding=dilation) > 0).float()
    return boundary

class LSMambaCD(nn.Module):

    def __init__(self, mobilenet_pretrained=None, vssm_pretrained=None, msdf_out_ch=120, lfmwb_mid_ch=192, use_cross_gate=True, use_mask_guide=False, **kwargs):
        super().__init__()
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)
        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.encoder = HybridEncoder(norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, mobilenet_pretrained=mobilenet_pretrained, vssm_pretrained=vssm_pretrained, msdf_out_ch=msdf_out_ch, lfmwb_mid_ch=lfmwb_mid_ch, **clean_kwargs)
        self.decoder = ChangeDecoder_Hybrid(encoder_dims=self.encoder.dims, channel_first=self.encoder.channel_first, norm_layer=norm_layer, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer, use_cross_gate=use_cross_gate, use_mask_guide=use_mask_guide, **clean_kwargs)
        self.upsample_refine = LightUpsampleRefine(in_ch=128, mid_ch=64, out_ch=32)
        self.refine_head = nn.Conv2d(32, 2, kernel_size=1)
        nn.init.zeros_(self.refine_head.weight)
        nn.init.zeros_(self.refine_head.bias)
        self.main_clf = nn.Conv2d(128, 2, kernel_size=1)
        self.freq_refine = FreqRefine(channels=128)

    def reparameterize(self):
        if hasattr(self.decoder, 'reparameterize'):
            self.decoder.reparameterize()

    def forward(self, pre_data, post_data):
        img_size = pre_data.shape[-2:]
        pre_features, post_features, delta_prior = self.encoder(pre_data, post_data)
        p1, mask2, mask3, mask4 = self.decoder(pre_features, post_features)
        p1 = self.freq_refine(p1)
        base_logits = self.main_clf(p1)
        base_up = F.interpolate(base_logits, size=img_size, mode='bilinear', align_corners=False)
        hr_feat = self.upsample_refine(p1)
        refine = self.refine_head(hr_feat)
        if refine.shape[-2:] != base_up.shape[-2:]:
            refine = F.interpolate(refine, size=img_size, mode='bilinear', align_corners=False)
        main_up = base_up + refine
        out = {'main': main_up}
        if mask2 is not None:
            out['aux2'] = F.interpolate(mask2, size=img_size, mode='bilinear', align_corners=False)
            out['aux3'] = F.interpolate(mask3, size=img_size, mode='bilinear', align_corners=False)
            out['aux4'] = F.interpolate(mask4, size=img_size, mode='bilinear', align_corners=False)
        if self.training:
            out['delta_prior'] = delta_prior
        return out
