"""
消融版 RSLQ —— 去掉每层可学习缩放（layer_scale）的 ResidualSphericalLeech。

对比实验目的：验证 semantic_layer_scale / layer_scales 这两个逐层可学习标量
（原 rslq.py 中把每层 up 投影输出乘上一个可训练系数）对残差量化性能的影响。

与 bimtokenizer/modules/quantizer/rslq.py 的 ResidualSphericalLeech 唯一区别：
  - 删除 self.semantic_layer_scale（原 nn.Parameter，语义层 up 输出的缩放）
  - 删除 self.layer_scales（原 nn.Parameter[num_rvq]，各 RVQ 声学层 up 输出的缩放）
  - forward / encode / decode 三处的 `e_* = up(...) * scale` 改为 `e_* = up(...)`
其余（码本、投影、mask、stats 聚合、encode/decode 契约）与原类逐字一致，
接口（forward 返回元组、encode/decode 形状）完全对齐，可被 codec 无缝替换。

基础量化器 SphericalLeechQuantizer 与常量直接复用 rslq.py，不重复实现，
避免两份实现漂移。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bimtokenizer.modules.quantizer.rslq import (
    SphericalLeechQuantizer,
    LEECH_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Residual Λ24-SQ （无 layer_scale 消融版）
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualSphericalLeechNoScale(nn.Module):
    """ResidualSphericalLeech 的消融版本：移除逐层可学习缩放系数。

    构造参数与 ResidualSphericalLeech 完全一致（配置可直接复用），
    仅内部不再创建 semantic_layer_scale / layer_scales 参数，
    各层 up 投影输出直接进入残差累加，不做缩放。
    """

    def __init__(
        self,
        num_codebooks: int,
        embedding_dim: int,
        codebook_size: int,
        compute_dtype: torch.dtype = torch.float32,
        codebook_path: str = None,
        leech_type: str = "full",
        semantic_codebook_size: int = None,
        semantic_codebook_path: str = None,
        semantic_leech_type: str = None,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embedding_dim = embedding_dim
        self.latent_dim = LEECH_DIM
        self.compute_dtype = compute_dtype
        self.codebook_size = codebook_size
        self.num_rvq = num_codebooks - 1

        # ── 声学 RVQ 层共用码本（所有 RVQ 残差层共享同一固定 Leech 码本）──
        self.acoustic_quantizer = SphericalLeechQuantizer(
            codebook_size=self.codebook_size,
            codebook_dim=self.latent_dim,
            codebook_path=codebook_path,
            leech_type=leech_type,
        )

        # ── 语义层码本（默认与声学层共用，传 semantic_codebook_* 则异构独立）──
        self.heterogeneous = (
            semantic_codebook_size is not None
            or semantic_codebook_path is not None
            or semantic_leech_type is not None
        )
        if self.heterogeneous:
            self.semantic_codebook_size = (
                int(semantic_codebook_size)
                if semantic_codebook_size is not None
                else self.codebook_size
            )
            self.semantic_quantizer = SphericalLeechQuantizer(
                codebook_size=self.semantic_codebook_size,
                codebook_dim=self.latent_dim,
                codebook_path=semantic_codebook_path,
                leech_type=(semantic_leech_type
                            if semantic_leech_type is not None else leech_type),
            )
        else:
            self.semantic_codebook_size = self.codebook_size
            self.semantic_quantizer = self.acoustic_quantizer

        # ── Semantic 层投影（无 semantic_layer_scale）──
        self.semantic_down = nn.Linear(self.embedding_dim, self.latent_dim, dtype=compute_dtype)
        self.semantic_up = nn.Linear(self.latent_dim, self.embedding_dim, dtype=compute_dtype)

        # ── RVQ 声学层投影（无 layer_scales）──
        self.downs = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.latent_dim, dtype=compute_dtype)
            for _ in range(self.num_rvq)
        ])
        self.ups = nn.ModuleList([
            nn.Linear(self.latent_dim, self.embedding_dim, dtype=compute_dtype)
            for _ in range(self.num_rvq)
        ])
        # ✂ 消融：原 rslq.py 此处有
        #       self.layer_scales = nn.Parameter(torch.ones(self.num_rvq))
        #   本版本删除，各 RVQ 层 up 输出不再缩放。

    # ───────────────────────────── forward ────────────────────────────────────
    def forward(
        self,
        x,
        input_length,
        n_codebooks=None,
    ):
        """
        Args / Returns 与 ResidualSphericalLeech.forward 完全一致（见 rslq.py）。
        """
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        B, T, C = x.shape
        device = x.device

        if n_codebooks is None:
            n_codebooks = torch.full(
                (B,), self.num_codebooks, device=device, dtype=torch.long
            )
        n_codebooks = n_codebooks.clamp(min=0, max=self.num_codebooks)

        residuals = x.to(self.compute_dtype)
        rvq_embed = torch.zeros_like(residuals)

        with torch.no_grad():
            arange_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            time_mask = (arange_t < input_length.unsqueeze(1))            # [B, T]
            time_mask_f = time_mask.unsqueeze(-1).to(residuals.dtype)

            n_mask = (
                torch.arange(self.num_codebooks, device=device).unsqueeze(0)
                < n_codebooks.unsqueeze(1)
            )  # [B, num_codebooks]

        indices_list = []

        # ── Semantic 层 (并行, 不参与残差) ──
        masked_x = residuals * time_mask_f
        z_sem = self.semantic_down(masked_x)
        z_sem_flat = F.normalize(z_sem.reshape(-1, self.latent_dim), dim=-1, eps=1e-8)
        q_sem_flat, idx_sem_flat = self.semantic_quantizer(z_sem_flat)
        q_sem = q_sem_flat.view(B, T, self.latent_dim)
        e_sem = self.semantic_up(q_sem)                    # ✂ 无 * semantic_layer_scale
        e_sem = e_sem * time_mask_f

        sem_layer_mask = n_mask[:, 0].view(B, 1, 1).to(e_sem.dtype)
        rvq_embed = rvq_embed + e_sem * sem_layer_mask

        sem_valid_mask = time_mask & n_mask[:, 0].unsqueeze(1)
        idx_sem = idx_sem_flat.view(B, T).masked_fill(~sem_valid_mask, 0)
        indices_list.append(idx_sem)

        # ── RVQ 声学层 (串行残差) ──
        for i in range(self.num_rvq):
            masked_residual = residuals * time_mask_f

            latent = self.downs[i](masked_residual)
            latent_flat = F.normalize(
                latent.reshape(-1, self.latent_dim), dim=-1, eps=1e-8
            )

            q_i_flat, idx_i_flat = self.acoustic_quantizer(latent_flat)
            q_i = q_i_flat.view(B, T, self.latent_dim)
            e_i = self.ups[i](q_i)                         # ✂ 无 * layer_scales[i]
            e_i = e_i * time_mask_f

            cb_idx = i + 1
            layer_mask = n_mask[:, cb_idx].view(B, 1, 1).to(e_i.dtype)
            update_mask = time_mask_f * layer_mask

            residuals = residuals - e_i * update_mask
            rvq_embed = rvq_embed + e_i * update_mask

            idx_i = idx_i_flat.view(B, T)
            layer_valid_mask = time_mask & n_mask[:, cb_idx].unsqueeze(1)
            idx_i = idx_i.masked_fill(~layer_valid_mask, 0)
            indices_list.append(idx_i)

        indices = torch.stack(indices_list, dim=0)

        return (
            rvq_embed.permute(0, 2, 1),
            indices,
        )

    # ───────────────────────────── encode / decode ────────────────────────────
    def encode(self, x, input_length=None, n_codebooks=None):
        """x: [B, C, T]  →  indices [nq, B, T]"""
        x = x.permute(0, 2, 1)
        B, T, C = x.shape
        device = x.device

        if n_codebooks is None:
            n_codebooks = torch.full(
                (B,), self.num_codebooks, device=device, dtype=torch.long
            )
        n_codebooks = n_codebooks.clamp(min=0, max=self.num_codebooks)
        n_max = int(n_codebooks.max().item())

        if input_length is not None:
            arange_t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            time_mask = (arange_t < input_length.unsqueeze(1))
        else:
            time_mask = torch.ones(B, T, device=device, dtype=torch.bool)
        time_mask_f = time_mask.unsqueeze(-1).to(x.dtype)

        residuals = x.to(self.compute_dtype)
        indices_list = []

        # Semantic
        masked_x = residuals * time_mask_f
        z_sem = self.semantic_down(masked_x)
        z_sem_flat = F.normalize(z_sem.reshape(-1, self.latent_dim), dim=-1, eps=1e-8)
        _, idx_sem = self.semantic_quantizer(z_sem_flat)
        idx_sem = idx_sem.view(B, T).masked_fill(~time_mask, 0)
        indices_list.append(idx_sem)

        # RVQ
        for i in range(min(n_max - 1, self.num_rvq)):
            masked_residual = residuals * time_mask_f
            latent = self.downs[i](masked_residual)
            latent_flat = F.normalize(
                latent.reshape(-1, self.latent_dim), dim=-1, eps=1e-8
            )

            _, idx_i = self.acoustic_quantizer(latent_flat)
            idx_i = idx_i.view(B, T).masked_fill(~time_mask, 0)
            indices_list.append(idx_i)

            q_i = self.acoustic_quantizer.decode_code(idx_i.reshape(-1)).view(B, T, self.latent_dim)
            e_i = self.ups[i](q_i)                         # ✂ 无 * layer_scales[i]
            e_i = e_i * time_mask_f

            layer_valid = ((i + 1) < n_codebooks).unsqueeze(1).unsqueeze(-1).to(e_i.dtype)
            residuals = residuals - e_i * layer_valid

        return torch.stack(indices_list, dim=0)  # [nq, B, T]

    def decode(self, indices, n_codebooks=None):
        """indices: [nq, B, T]  →  [B, C, T]"""
        indices = indices.permute(1, 2, 0)  # (B, T, nq)
        B, T, N = indices.shape
        device = indices.device

        if n_codebooks is None:
            n_codebooks = torch.full((B,), N, device=device, dtype=torch.long)
        n_codebooks = n_codebooks.clamp(min=0, max=N)
        n_max = int(n_codebooks.max().item())

        rvq_embed = torch.zeros(
            B, T, self.embedding_dim, device=device, dtype=self.compute_dtype
        )

        with torch.no_grad():
            layer_mask = (
                torch.arange(n_max, device=device).unsqueeze(0)
                < n_codebooks.unsqueeze(1)
            )  # [B, n_max]

        # Semantic (layer 0)
        q_sem = self.semantic_quantizer.decode_code(indices[:, :, 0].reshape(-1))
        e_sem = self.semantic_up(q_sem.view(B, T, self.latent_dim))   # ✂ 无 * semantic_layer_scale
        rvq_embed = rvq_embed + e_sem * layer_mask[:, 0].view(B, 1, 1).to(e_sem.dtype)

        # RVQ layers (1..n_max-1)
        for i in range(min(n_max - 1, self.num_rvq)):
            cb_idx = i + 1
            q_i = self.acoustic_quantizer.decode_code(indices[:, :, cb_idx].reshape(-1))
            q_i = q_i.view(B, T, self.latent_dim)
            e_i = self.ups[i](q_i)                         # ✂ 无 * layer_scales[i]
            rvq_embed = rvq_embed + e_i * layer_mask[:, cb_idx].view(B, 1, 1).to(e_i.dtype)

        return rvq_embed.permute(0, 2, 1)  # (B, C, T)
