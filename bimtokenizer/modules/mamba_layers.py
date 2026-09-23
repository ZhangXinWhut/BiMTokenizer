"""Mamba Layer Stacks.

Unified module for stacking multiple BiMamba / BiMamba2 blocks
with PreNorm residual connections and optional GatedMLP FFN.

Replaces the old MambaBlocksSequential and Mamba2BlocksSequential
with a single MambaLayers class using nn.ModuleList.
"""

import math
import torch
import torch.nn as nn

from functools import partial
from typing import Optional

from mamba_ssm import Mamba
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from .bimamba import BiMamba, BiMamba2, Block
from .selective_scan_interface import build_valid_time_flip_indices
from mamba_ssm.modules.mlp import GatedMLP

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ============================================================================
# Block factory & weight initialization
# ============================================================================

def _create_block(
    d_model: int,
    d_intermediate: int,
    ssm_cls,
    ssm_cfg: Optional[dict] = None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    device=None,
    dtype=None,
) -> Block:
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(ssm_cls, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon, **factory_kwargs,
    )
    mlp_cls = (
        nn.Identity if d_intermediate == 0
        else partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    )
    block = Block(
        d_model, mixer_cls, mlp_cls,
        norm_cls=norm_cls, fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """GPT-2 style weight initialization with prenorm residual rescaling."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "out_proj_b.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ============================================================================
# MambaLayers: unified stack of Mamba1 / Mamba2 blocks
# ============================================================================

class MambaLayers(nn.Module):
    """Stack of Mamba / BiMamba / BiMamba2 blocks.

    Unified replacement for MambaBlocksSequential and Mamba2BlocksSequential.
    Uses nn.ModuleList for explicit iteration control.

    Args:
        n_mamba: number of blocks to stack
        mamba_version: 1 for Mamba1, 2 for Mamba2 (SSD)
        bidirectional: whether to use bidirectional variants
        bimamba_type: "v2" (InnBiMamba) or "v3" (ExtBiMamba)
        d_model: model / hidden dimension
        d_state: SSM state dimension (16 for Mamba1, 64 for Mamba2)
        d_intermediate: GatedMLP FFN dimension; 0 = no FFN
        expand: expansion factor for d_inner
        d_conv: convolution kernel size
        dt_rank: (Mamba1 only) rank of dt projection, "auto" = d_model//16
        headdim: (Mamba2 only) dimension per head
        ngroups: (Mamba2 only) number of groups for B,C matrices
        chunk_size: (Mamba2 only) chunk size for SSD algorithm
        use_mem_eff_path: (Mamba2 only) use fused Triton kernels
        conv_bias / bias: bias settings
        fused_add_norm: use Triton fused add+norm
        rms_norm: use RMSNorm instead of LayerNorm
        norm_epsilon: epsilon for normalization
        initializer_cfg: override for weight init params
        residual_in_fp32: keep residual in fp32
    """

    def __init__(
        self,
        n_mamba: int,
        mamba_version: int = 1,
        bidirectional: bool = False,
        bimamba_type: str = "v3",
        d_model: int = 768,
        d_state: int = 16,
        d_intermediate: int = 0,
        expand: int = 2,
        d_conv: int = 4,
        # Mamba1 only
        dt_rank: str = "auto",
        # Mamba2 only
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        # Common
        conv_bias: bool = True,
        bias: bool = False,
        fused_add_norm: bool = True,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        assert mamba_version in [1, 2], "mamba_version must be 1 or 2"
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Select SSM class and build config.
        # bimamba_type=="v1": unidirectional path through BiMamba(v1) — keeps a
        # uniform class interface. (Equivalent in math to upstream Mamba but routed
        # through our wrapper so future changes stay in one place.)
        if mamba_version == 1:
            if bidirectional or bimamba_type == "v1":
                ssm_cls = BiMamba
            else:
                ssm_cls = Mamba
            ssm_cfg = {
                "d_state": d_state, "expand": expand, "d_conv": d_conv,
                "dt_rank": dt_rank, "conv_bias": conv_bias, "bias": bias,
            }
        else:
            ssm_cls = BiMamba2 if bidirectional else Mamba2Simple
            ssm_cfg = {
                "d_state": d_state, "expand": expand, "d_conv": d_conv,
                "headdim": headdim, "ngroups": ngroups,
                "chunk_size": chunk_size, "use_mem_eff_path": use_mem_eff_path,
                "conv_bias": conv_bias, "bias": bias,
            }
        if ssm_cls in (BiMamba, BiMamba2):
            ssm_cfg["bimamba_type"] = bimamba_type

        self.layers = nn.ModuleList([
            _create_block(
                d_model=d_model,
                d_intermediate=d_intermediate,
                ssm_cls=ssm_cls,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
            )
            for i in range(n_mamba)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_mamba,
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x: torch.Tensor, keep_to=None, inference_params=None, seq_lens=None) -> torch.Tensor:
        """
        Args:
            x: (batch, seqlen, d_model)
            keep_to: unused, kept for API compatibility
            inference_params: optional inference cache params
            seq_lens: (batch,) valid lengths for padded batches
        Returns:
            (batch, seqlen, d_model)
        """
        hidden_states = x
        residual = None

        flip_indices = None
        if (
            seq_lens is not None
            and len(self.layers) > 0
            and isinstance(self.layers[0].mixer, BiMamba)
            and self.layers[0].mixer.bimamba_type == "v3"
        ):
            flip_indices = build_valid_time_flip_indices(
                seq_lens, hidden_states.shape[1], hidden_states.device
            )

        for layer in self.layers:
            mixer_kwargs = {"seq_lens": seq_lens}
            if flip_indices is not None:
                mixer_kwargs["flip_indices"] = flip_indices
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params,
                **mixer_kwargs,
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight, self.norm_f.bias,
                eps=self.norm_f.eps, residual=residual,
                prenorm=False, residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states
