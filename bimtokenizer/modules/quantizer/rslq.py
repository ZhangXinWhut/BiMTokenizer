"""
Spherical Leech Quantization (Λ24-SQ) for Speech Tokenizer
Based on: https://arxiv.org/abs/2512.14697

Key properties from the paper:
  1. NO commitment loss, NO entropy loss needed
     — the densest sphere packing (δ_min = √3/2 ≈ 0.866) ensures
       near-equal Voronoi cells, implicitly maximizing entropy
  2. Pipeline: z ∈ R^d → L2_normalize → Q_Λ(z̃) on S^23
  3. Fixed codebook: 196,560 vectors from the first shell of the
     Leech lattice, normalized to unit length
  4. d-itwise representation: each lattice point has 24 integer
     coordinates in {-4,...,4}, enabling 24×9-way factorized prediction
     for downstream language models
  5. Gradient: straight-through estimator (STE)
  6. Can combine with residual quantization (Section 3.5)

Adaptation to speech:
  - Input: (B, L, C) from speech encoder (e.g., EnCodec/DAC encoder)
  - project_in maps C→24, L2 normalize, quantize on S^23, project_out maps 24→C
  - Training loss: reconstruction only (e.g., ℓ1 + mel-spec + discriminator)
    NO auxiliary quantizer losses needed
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LEECH_DIM = 24
# Leech 第一壳层向量的平方模长(在标准整数坐标系下).
# 已归一化码字 z_q 乘以 sqrt(LEECH_NORM_SQ) 即还原到 {-4,...,4}^24 的整数坐标.
LEECH_NORM_SQ = 32
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# ═══════════════════════════════════════════════════════════════════════════════
#  Λ24-SQ quantizer
# ═══════════════════════════════════════════════════════════════════════════════
class SphericalLeechQuantizer(nn.Module):
    """
    Λ24-SQ: Spherical Leech Quantization.

    Follows Eq. (11) of the paper:
        z ∈ R^d  →  project_in  →  L2_normalize  →  Q_Λ  →  project_out

    The codebook is FIXED (no gradient). The only learnable parameters are
    the projection layers. No commitment loss or entropy loss is needed.

    Args:
        codebook_size:  number of Leech lattice vectors to use
                        (max 196560 for full first shell;
                         subsets: type2=97152, type3=98304, type4=1104)
        codebook_dim:   dimension of the Leech lattice vectors
        codebook_path:  path to .npy file with pre-normalized Leech vectors
        leech_type:     which subset of the Leech lattice FIRST shell to use.
                        ‼ 三个 type 不是不同壳层, 而是【同一个第一壳层】里按 shape 分的三类
                          (论文 Table 3, 数量 97152+98304+1104=196560). 全部范数都 = sqrt(32),
                          归一化后都在同一个球面 S^23 上, 最近对 d_min 都 = 1(packing 相同).
                        full  : 196560 = 完整第一壳层(全部 minimal vectors).
                        "2"   : 97152  = shape 类 Λ24(2)_2, 形状 (2^8 0^16).
                        "3"   : 98304  = shape 类 Λ24(2)_3, 形状 (3 1^23).
                        "4"   : 1104   = shape 类 Λ24(2)_4, 形状 (4^2 0^22), 最对称的一类.

                        ── 关于选码本大小(实测, S^23 均匀采样下的期望量化 MSE, 越低越好)──
                          1104 → 0.81, 2048 → 0.72, 4096 → 0.67, 32768 → 0.54, 196560 → 0.45.
                        量化误差随 |C| 单调下降; d_min(packing) 对所有子集都相同, 真正影响
                        重建的是 covering/量化 MSE, 所以越大越好(代价是每 token 比特数上升).
                        2048/4096 等非壳层完整数的子集没有对称解, 用 make_leech_subuset.py
                        从 full 均匀随机抽即可(196560 点近乎均匀, 随机子集 usage≈0.999).
                        注意: 随机子集会破坏整数坐标闭合性, 故 compute_bit_indices /
                        factorized d-itwise 预测只对 full 或完整 shape 类有效; 走 argmax
                        检索的 codec 不受影响.
    """
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int = LEECH_DIM,
        codebook_path: str = None,
        leech_type: str = "full",
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.register_buffer(
            "codebook",
            torch.zeros(codebook_size, codebook_dim, dtype=torch.float32),
        )

        if codebook_path is None:
            codebook_path = {
                "full": os.path.join(_CACHE_DIR, "leech_lattices_normalized.npy"),
                "2": os.path.join(_CACHE_DIR, "leech_lattices_type2_normalized.npy"),
                "3": os.path.join(_CACHE_DIR, "leech_lattices_type3_normalized.npy"),
                "4": os.path.join(_CACHE_DIR, "leech_lattices_type4_normalized.npy"),
            }.get(leech_type, os.path.join(_CACHE_DIR, "leech_lattices_normalized.npy"))
        self.codebook_path = codebook_path
        self._load_codebook()

    def _load_codebook(self):
        try:
            import numpy as np
            weights = np.load(self.codebook_path)
        except FileNotFoundError:
            logging.warning(
                f"[SphericalLeechQuantizer] Codebook file {self.codebook_path} not found, "
                "using random unit-sphere initialization for testing."
            )
            nn.init.trunc_normal_(self.codebook, std=0.02)
            self.codebook.copy_(F.normalize(self.codebook, dim=-1))
            return

        assert weights.shape[1] == self.codebook_dim
        if weights.shape[0] < self.codebook_size:
            raise ValueError(
                f"Codebook has {weights.shape[0]} vectors but need {self.codebook_size}"
            )
        if weights.shape[0] > self.codebook_size:
            logging.info(
                f"[SphericalLeechQuantizer] Using first {self.codebook_size} of "
                f"{weights.shape[0]} codebook vectors"
            )
        self.codebook.copy_(torch.from_numpy(weights[: self.codebook_size]).float())

    def forward(self, z):
        z = z.to(self.codebook.dtype)
        sim = z @ self.codebook.t()
        indices = sim.argmax(dim=-1)
        z_q = F.embedding(indices, self.codebook)
        z_q = z + (z_q - z).detach()  # STE
        return z_q, indices

    def decode_code(self, indices):
        return F.embedding(indices, self.codebook)


# ═══════════════════════════════════════════════════════════════════════════════
#  ResidualSphericalLeech — 1 semantic codebook layer + (N-1) RVQ acoustic layers
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualSphericalLeech(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        embedding_dim: int,
        codebook_size: int,
        compute_dtype: torch.dtype = torch.float32,
        codebook_path: str = None,
        leech_type: str = "full",
        # ── 语义层独立码本（异构码本：语义/声学可不同大小/不同来源）──
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

        # ── 语义层码本 ──
        # 默认与声学层共用（向后兼容：不传 semantic_codebook_* 时退化为单 quantizer 语义）。
        # 传入 semantic_codebook_path/size 即启用独立语义码本（例如 32768 子集 vs 声学 type4=1104）。
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
            # 同构：语义层直接复用声学码本实例（省一份 buffer，行为与旧版完全一致）。
            self.semantic_codebook_size = self.codebook_size
            self.semantic_quantizer = self.acoustic_quantizer

        # ── Semantic 层投影 ──
        self.semantic_down = nn.Linear(self.embedding_dim, self.latent_dim, dtype=compute_dtype)
        self.semantic_up = nn.Linear(self.latent_dim, self.embedding_dim, dtype=compute_dtype)
        self.semantic_layer_scale = nn.Parameter(torch.ones((), dtype=compute_dtype))

        # ── RVQ 声学层投影 ──
        self.downs = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.latent_dim, dtype=compute_dtype)
            for _ in range(self.num_rvq)
        ])
        self.ups = nn.ModuleList([
            nn.Linear(self.latent_dim, self.embedding_dim, dtype=compute_dtype)
            for _ in range(self.num_rvq)
        ])
        self.layer_scales = nn.Parameter(
            torch.ones(self.num_rvq, dtype=compute_dtype)
        )

    # ───────────────────────────── forward ────────────────────────────────────
    def forward(
        self,
        x,
        input_length,
        n_codebooks=None,
    ):
        """
        Args:
            x:             [B, C, T]
            input_length:  [B]
            n_codebooks:   [B] or None

        Returns:
            rvq_embed      [B, C, T]
            indices        [nq, B, T]
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
        e_sem = self.semantic_up(q_sem) * self.semantic_layer_scale
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
            e_i = self.ups[i](q_i) * self.layer_scales[i]
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
        """
        x: [B, C, T]  →  indices [nq, B, T]
        """
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
            e_i = self.ups[i](q_i) * self.layer_scales[i]
            e_i = e_i * time_mask_f

            layer_valid = ((i + 1) < n_codebooks).unsqueeze(1).unsqueeze(-1).to(e_i.dtype)
            residuals = residuals - e_i * layer_valid

        return torch.stack(indices_list, dim=0)  # [nq, B, T]

    def decode(self, indices, n_codebooks=None):
        """
        indices: [nq, B, T]  →  [B, C, T]
        """
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
        e_sem = self.semantic_up(q_sem.view(B, T, self.latent_dim)) * self.semantic_layer_scale
        rvq_embed = rvq_embed + e_sem * layer_mask[:, 0].view(B, 1, 1).to(e_sem.dtype)

        # RVQ layers (1..n_max-1)
        for i in range(min(n_max - 1, self.num_rvq)):
            cb_idx = i + 1
            q_i = self.acoustic_quantizer.decode_code(indices[:, :, cb_idx].reshape(-1))
            q_i = q_i.view(B, T, self.latent_dim)
            e_i = self.ups[i](q_i) * self.layer_scales[i]
            rvq_embed = rvq_embed + e_i * layer_mask[:, cb_idx].view(B, 1, 1).to(e_i.dtype)
        return rvq_embed.permute(0, 2, 1)  # (B, C, T)
