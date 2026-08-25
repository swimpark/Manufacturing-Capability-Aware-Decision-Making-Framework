# autoencoder_model_Unet10_lat48.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2)

def GN(C: int, prefer_groups: int = 8) -> nn.GroupNorm:
    g = min(prefer_groups, C)
    while C % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, C)

# 128^3 → 64^3 → 32^3 → 16^3 → 8^3
class ShapeEncoder(nn.Module):
    def __init__(self, in_channels=1, dim=128, out_conv_channels=512, hidden_dim=32,
                 normalize_shape: bool = True):
        super().__init__()
        torch.manual_seed(3)
        self.normalize_shape = bool(normalize_shape)
        c1, c2, c3, c4 = out_conv_channels//8, out_conv_channels//4, out_conv_channels//2, out_conv_channels
        self.out_dim = dim // 16  # 8

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, c1, 4, 2, 1, bias=False),  # 128 → 64
            GN(c1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(c1, c2, 4, 2, 1, bias=False),           # 64 → 32
            GN(c2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(c2, c3, 4, 2, 1, bias=False),           # 32 → 16
            GN(c3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(c3, c4, 4, 2, 1, bias=False),           # 16 → 8
            GN(c4),
        )

        self.vector = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(c4 * self.out_dim**3, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b = x.size(0)
        x1 = F.leaky_relu(self.conv1(x),  0.2)  # [B,  64, 64,64,64]
        x2 = F.leaky_relu(self.conv2(x1), 0.2)  # [B, 128, 32,32,32]
        x3 = F.leaky_relu(self.conv3(x2), 0.2)  # [B, 256, 16,16,16]
        x4 = F.leaky_relu(self.conv4(x3), 0.2)  # [B, 512,  8, 8, 8]

        v  = x4.view(b, -1)
        zs = self.vector(v)                    # [B,32]

        if self.normalize_shape:
            m = zs.mean(dim=1, keepdim=True)
            s = zs.std(dim=1, keepdim=True)
            zs = (zs - m) / (s + 1e-8)

        skips = {
            "s64": x1,  # [B,  64, 64,64,64]
            "s32": x2,  # [B, 128, 32,32,32]
            "s16": x3,  # [B, 256, 16,16,16]
            "s08": x4,  # [B, 512,  8, 8, 8]
        }
        return zs, skips


class MetricsEncoder(nn.Module):
    def __init__(self, hidden_dim=8, out_dim=16):
        super().__init__()
        torch.manual_seed(3)
        def mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        self.time_mlp      = mlp(1)
        self.cost_mlp      = mlp(1)
        self.quantity_mlp  = mlp(1)
        self.tolerance_mlp = mlp(1)
        self.materials_mlp = mlp(3)

    def forward(self, t, c, q, tol, m):
        b = t.size(0)
        zt   = self.time_mlp(t.view(b,1))        # [B,16]
        zc   = self.cost_mlp(c.view(b,1))        # [B,16]
        zq   = self.quantity_mlp(q.view(b,1))    # [B,16]
        ztol = self.tolerance_mlp(tol.view(b,1)) # [B,16]
        zm   = self.materials_mlp(m)             # [B,16]
        return zt, zc, zq, ztol, zm


class ShapeDecoder(nn.Module):
    """
    z_dim = 48
    """
    def __init__(self, in_channels=512, dim=128, z_dim=48, out_channels=1):
        super().__init__()
        torch.manual_seed(3)
        self.out_dim = dim // 16  # 8

        self.linear = nn.Linear(z_dim, in_channels * self.out_dim**3)

        self.red_s16 = nn.Conv3d(256, 32, 1, bias=False)
        self.red_s32 = nn.Conv3d(128, 16, 1, bias=False)
        self.red_s64 = nn.Conv3d( 64,  8, 1, bias=False)

        self.gate_from_z16 = nn.Sequential(nn.Linear(z_dim, 32), nn.Sigmoid())
        self.gate_from_z32 = nn.Sequential(nn.Linear(z_dim, 16), nn.Sigmoid())
        self.gate_from_z64 = nn.Sequential(nn.Linear(z_dim,  8), nn.Sigmoid())

        self.skip_do = nn.Dropout3d(p=0.2)

        def zproj(out_ch):
            return nn.Sequential(
                nn.Linear(z_dim, out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.zinj16 = zproj(64)
        self.zinj32 = zproj(32)
        self.zinj64 = zproj(16)

        self.up16_up = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),  # 8 → 16
            GN(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up32_up = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),  # 16 → 32
            GN(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up64_up = nn.Sequential(
            nn.ConvTranspose3d(128,  64, 4, 2, 1, bias=False),  # 32 → 64
            GN(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up16_fuse = nn.Sequential(
            nn.Conv3d(352, 256, 3, 1, 1, bias=False),  # 256 + 32 + 64
            GN(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up32_fuse = nn.Sequential(
            nn.Conv3d(176, 128, 3, 1, 1, bias=False),  # 128 + 16 + 32
            GN(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up64_fuse = nn.Sequential(
            nn.Conv3d( 88,  64, 3, 1, 1, bias=False),  #  64 +  8 + 16
            GN(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.to128 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, 2, 1, bias=False),
            GN(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, out_channels, 3, 1, 1, bias=False),
        )

        self.use_s16 = True
        self.use_s32 = True
        self.use_s64 = True

        self.skip_drop_p = 0.6
        self.skip_drop_p16 = 0.6
        self.skip_drop_p32 = 0.6
        self.skip_drop_p64 = 0.6

    @torch.no_grad()
    def set_skip_drop_p(self, p: float):
        p = float(max(0.0, min(1.0, p)))
        self.skip_drop_p = p
        self.skip_drop_p16 = p
        self.skip_drop_p32 = p
        self.skip_drop_p64 = p

    @torch.no_grad()
    def set_skip_drop_ps(self, p16: float, p32: float, p64: float):
        self.skip_drop_p16 = float(max(0.0, min(1.0, p16)))
        self.skip_drop_p32 = float(max(0.0, min(1.0, p32)))
        self.skip_drop_p64 = float(max(0.0, min(1.0, p64)))

    def _maybe_drop(self, t, p=0.5):
        if self.training and torch.rand(1, device=t.device) < p:
            return torch.zeros_like(t)
        return t

    def forward(self, z, skips):
        b = z.size(0)

        h = self.linear(z).view(b, 512, self.out_dim, self.out_dim, self.out_dim)

        # 8 → 16
        h   = self.up16_up(h)  # [B,256,16,16,16]
        zin = self.zinj16(z).view(b, 64, 1, 1, 1).expand(-1, 64, h.size(2), h.size(3), h.size(4))
        if self.use_s16:
            rs16 = self.red_s16(skips["s16"])                 # [B,32,16,16,16]
            g16  = self.gate_from_z16(z).view(b,32,1,1,1).expand_as(rs16)
            s16  = self._maybe_drop(self.skip_do(g16 * rs16), p=self.skip_drop_p16)
        else:
            s16  = torch.zeros(b, 32, h.size(2), h.size(3), h.size(4),
                               device=h.device, dtype=h.dtype)
        h = self.up16_fuse(torch.cat([h, s16, zin], dim=1))   # [B,256,16,16,16]

        # 16 → 32
        h   = self.up32_up(h)  # [B,128,32,32,32]
        zin = self.zinj32(z).view(b, 32, 1, 1, 1).expand(-1, 32, h.size(2), h.size(3), h.size(4))
        if self.use_s32:
            rs32 = self.red_s32(skips["s32"])                 # [B,16,32,32,32]
            g32  = self.gate_from_z32(z).view(b,16,1,1,1).expand_as(rs32)
            s32  = self._maybe_drop(self.skip_do(g32 * rs32), p=self.skip_drop_p32)
        else:
            s32  = torch.zeros(b, 16, h.size(2), h.size(3), h.size(4),
                               device=h.device, dtype=h.dtype)
        h = self.up32_fuse(torch.cat([h, s32, zin], dim=1))   # [B,128,32,32,32]

        # 32 → 64
        h   = self.up64_up(h)  # [B,64,64,64,64]
        zin = self.zinj64(z).view(b, 16, 1, 1, 1).expand(-1, 16, h.size(2), h.size(3), h.size(4))
        if self.use_s64:
            rs64 = self.red_s64(skips["s64"])                 # [B,8,64,64,64]
            g64  = self.gate_from_z64(z).view(b, 8,1,1,1).expand_as(rs64)
            s64  = self._maybe_drop(self.skip_do(g64 * rs64), p=self.skip_drop_p64)
        else:
            s64  = torch.zeros(b, 8, h.size(2), h.size(3), h.size(4),
                               device=h.device, dtype=h.dtype)
        h = self.up64_fuse(torch.cat([h, s64, zin], dim=1))   # [B,64,64,64,64]

        # 64 → 128
        logits = self.to128(h)                                # [B,1,128,128,128]
        return logits

    @torch.no_grad()
    def forward_from_latent(self, z):
        """
        """
        self.eval()
        b  = z.size(0)
        d8 = self.out_dim        # 8
        d16, d32, d64 = 16, 32, 64
        dev = z.device
        dtype = torch.float32

        zero_skips = {
            "s16": torch.zeros(b, 256, d16, d16, d16, device=dev, dtype=dtype),
            "s32": torch.zeros(b, 128, d32, d32, d32, device=dev, dtype=dtype),
            "s64": torch.zeros(b,  64, d64, d64, d64, device=dev, dtype=dtype),
        }
        return self.forward(z, zero_skips)


class MetricsDecoder(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=8):
        super().__init__()
        torch.manual_seed(3)
        def make_dec(out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, out_dim),
            )
        self.time_dec      = make_dec(1)
        self.cost_dec      = make_dec(1)
        self.quantity_dec  = make_dec(1)
        self.tolerance_dec = make_dec(1)
        self.materials_dec = make_dec(3)

    def forward(self, z):
        return (
            self.time_dec(z),      # [B,1]
            self.cost_dec(z),      # [B,1]
            self.quantity_dec(z),  # [B,1]
            self.tolerance_dec(z), # [B,1]
            self.materials_dec(z)  # [B,3]
        )


class MultiModalAutoencoder(nn.Module):
    """
      x_voxel, t, c, q, tol, m
      x_logits, (t_hat, c_hat, q_hat, tol_hat, m_hat), zlat
    """
    def __init__(self, normalize_shape: bool = True):
        super().__init__()
        self.shape_enc   = ShapeEncoder(hidden_dim=32, normalize_shape=normalize_shape)
        self.metrics_enc = MetricsEncoder()
        self.shape_dec   = ShapeDecoder(z_dim=48)
        self.metrics_dec = MetricsDecoder(in_dim=48)
        self.to_latent   = nn.Sequential(
            nn.Linear(32 + 16*5, 48),  # 112 → 48
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_voxel, t, c, q, tol, m):
        zs, skips = self.shape_enc(x_voxel)                       # [B,32]
        zt, zc, zq, ztol, zm = self.metrics_enc(t, c, q, tol, m)
        zcat = torch.cat([zs, zt, zc, zq, ztol, zm], dim=1)       # [B,112]
        zlat = self.to_latent(zcat)                               # [B,48]

        x_logits = self.shape_dec(zlat, skips)                    # [B,1,128,128,128]
        t_hat, c_hat, q_hat, tol_hat, m_hat = self.metrics_dec(zlat)
        return x_logits, (t_hat, c_hat, q_hat, tol_hat, m_hat), zlat

    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor):
        """
        """
        self.eval()
        # shape
        if hasattr(self.shape_dec, "forward_from_latent"):
            x_logits = self.shape_dec.forward_from_latent(z)
        else:
            raise RuntimeError("Operation failed or required precondition is missing.")

        # metrics
        t_hat, c_hat, q_hat, tol_hat, m_hat = self.metrics_dec(z)
        return x_logits, (t_hat, c_hat, q_hat, tol_hat, m_hat)
