"""Bidirectional Mamba Modules.

Contains:
- BiMamba:  Mamba1-based bidirectional SSM (v2: InnBiMamba, v3: ExtBiMamba)
- BiMamba2: Mamba2-based bidirectional SSM (v2: InnBiMamba2, v3: ExtBiMamba2)
- Block:    PreNorm residual block wrapper shared by both
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from .selective_scan_interface import (
        selective_scan_fn,
        mamba_inner_fn,
        mamba_inner_fn_no_out_proj,
        inn_bimamba_inner_fn,
        ext_bimamba_inner_fn,
        flip_valid_time,
    )
except ImportError:
    selective_scan_fn = None
    mamba_inner_fn = None
    mamba_inner_fn_no_out_proj = None
    inn_bimamba_inner_fn = None
    ext_bimamba_inner_fn = None
    flip_valid_time = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.modules.mlp import GatedMLP


# ============================================================================
# BiMamba (Mamba1-based)
# ============================================================================

class BiMamba(nn.Module):
    """Bidirectional Mamba1 SSM.

    Supports:
    - v1 (UniMamba):  original unidirectional Mamba1 (no backward params)
    - v2 (InnBiMamba): shared in_proj/out_proj, separate conv/SSM params per direction
    - v3 (ExtBiMamba): fully independent parameters for each direction
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v3",
        if_devide_out=True,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        assert bimamba_type in ["v1", "v2", "v3"]

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones((d_model)), requires_grad=True
            )

        # ---- Forward direction ----
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # ---- Backward direction (only for v2/v3, skipped for v1 unidirectional) ----
        if bimamba_type != "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n", d=self.d_inner,
            ).contiguous()
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                self.d_inner, self.d_inner, bias=conv_bias,
                kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

            # ---- v3-only: separate in_proj_b, out_proj_b, and dt_proj_b init ----
            if bimamba_type == "v3":
                self.in_proj_b = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
                self.out_proj_b = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

                if dt_init == "constant":
                    nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
                elif dt_init == "random":
                    nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)

                dt_b = torch.exp(
                    torch.rand(self.d_inner, **factory_kwargs)
                    * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                ).clamp(min=dt_init_floor)
                inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
                with torch.no_grad():
                    self.dt_proj_b.bias.copy_(inv_dt_b)
                self.dt_proj_b.bias._no_reinit = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, seq_lens=None):
        """
        Args:
            hidden_states: (batch, seqlen, d_model)
            seq_lens: (batch,) valid sequence lengths; None = legacy full flip on padding
        Returns:
            (batch, seqlen, d_model)
        """
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float()) if self.bimamba_type != "v1" else None

        if self.use_fast_path and inference_params is None:
            if self.bimamba_type == "v1":
                # Original unidirectional Mamba1 (matches mamba_ssm.modules.mamba_simple.Mamba)
                xz = rearrange(
                    self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                    "d (b l) -> b d l", l=seqlen,
                )
                if self.in_proj.bias is not None:
                    xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight, self.conv1d.bias,
                    self.x_proj.weight, self.dt_proj.weight,
                    self.out_proj.weight, self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
            elif self.bimamba_type == "v2":
                xz = rearrange(
                    self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                    "d (b l) -> b d l", l=seqlen,
                )
                if self.in_proj.bias is not None:
                    xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
                out = inn_bimamba_inner_fn(
                    xz,
                    self.conv1d.weight, self.conv1d.bias,
                    self.x_proj.weight, self.dt_proj.weight,
                    A, self.D.float(), self.dt_proj.bias.float(),
                    self.conv1d_b.weight, self.conv1d_b.bias,
                    self.x_proj_b.weight, self.dt_proj_b.weight,
                    A_b, self.D_b.float(), self.dt_proj_b.bias.float(),
                    self.out_proj.weight, self.out_proj.bias,
                    if_devide_out=self.if_devide_out,
                    seq_lens=seq_lens,
                )
            else:  # v3
                out = ext_bimamba_inner_fn(
                    hidden_states,
                    self.in_proj.weight, self.in_proj.bias,
                    self.conv1d.weight, self.conv1d.bias,
                    self.x_proj.weight, self.dt_proj.weight,
                    A, self.D.float(), self.dt_proj.bias.float(),
                    self.out_proj.weight, self.out_proj.bias,
                    self.in_proj_b.weight, self.in_proj_b.bias,
                    self.conv1d_b.weight, self.conv1d_b.bias,
                    self.x_proj_b.weight, self.dt_proj_b.weight,
                    A_b, self.D_b.float(), self.dt_proj_b.bias.float(),
                    self.out_proj_b.weight, self.out_proj_b.bias,
                    if_devide_out=self.if_devide_out,
                    seq_lens=seq_lens,
                )
        else:
            # Slow path (inference with cache or no causal_conv1d)
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l", l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(),
                z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x, conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D,
                z=z, dt_bias=self.dt_proj.bias, dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv,
            device=device, dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state,
            device=device, dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv,
                device=self.conv1d.weight.device, dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_state,
                device=self.dt_proj.weight.device, dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


# ============================================================================
# BiMamba2 (Mamba2/SSD-based)
# ============================================================================

class BiMamba2(nn.Module):
    """Bidirectional Mamba2 SSM using SSD (State Space Duality) algorithm.

    Supports:
    - v2 (InnBiMamba2): shared in_proj/out_proj, separate conv/SSM params per direction
    - v3 (ExtBiMamba2): fully independent parameters for each direction
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        bimamba_type="v3",
        if_devide_out=True,
        init_layer_scale=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        assert bimamba_type in ["v2", "v3"], "bimamba_type must be 'v2' or 'v3'"

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones((d_model)), requires_grad=True
            )

        # ---- Forward direction ----
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        assert RMSNormGated is not None, "RMSNormGated is required for BiMamba2"
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        # ---- Backward direction ----
        self.conv1d_b = nn.Conv1d(
            conv_dim, conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d_b.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states_b = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states_b._no_weight_decay = True

        dt_b = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias_b = nn.Parameter(dt_b + torch.log(-torch.expm1(-dt_b)))
        self.dt_bias_b._no_weight_decay = True

        A_b = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_b_log = nn.Parameter(torch.log(A_b).to(dtype=dtype))
        self.A_b_log._no_weight_decay = True

        self.D_b = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_b._no_weight_decay = True

        self.norm_b = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        # ---- v3-only: separate in_proj_b, out_proj_b ----
        if bimamba_type == "v3":
            self.in_proj_b = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.out_proj_b = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None, inference_params=None, seq_lens=None):
        """
        Args:
            u: (batch, seqlen, d_model)
            seq_lens: (batch,) valid lengths for padded batches; None = legacy full flip
        Returns:
            (batch, seqlen, d_model)
        """
        batch, seqlen, dim = u.shape

        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())

        initial_states = None
        initial_states_b = None
        if self.learnable_init_states:
            initial_states = repeat(self.init_states, "... -> b ...", b=batch)
            initial_states_b = repeat(self.init_states_b, "... -> b ...", b=batch)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}

        if self.bimamba_type == "v2":
            out = self._forward_v2(u, A, A_b, initial_states, initial_states_b, seq_idx, dt_limit_kwargs, seq_lens)
        else:
            out = self._forward_v3(u, A, A_b, initial_states, initial_states_b, seq_idx, dt_limit_kwargs, seq_lens)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

    def _flip_bwd_seq_idx(self, seq_idx, seq_lens):
        if seq_idx is None:
            return None
        if flip_valid_time is None:
            return seq_idx.flip(dims=[1])
        return flip_valid_time(seq_idx.unsqueeze(-1), seq_lens).squeeze(-1)

    def _forward_v2(self, u, A, A_b, initial_states, initial_states_b, seq_idx, dt_limit_kwargs, seq_lens=None):
        """v2 (InnBiMamba2): shared in_proj, separate processing, shared out_proj."""
        zxbcdt = self.in_proj(u)

        if self.use_mem_eff_path and mamba_split_conv1d_scan_combined is not None:
            out_f = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias,
                self.dt_bias, A, D=self.D,
                chunk_size=self.chunk_size, seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.eps,
                outproj_weight=None, outproj_bias=None,
                headdim=self.headdim, ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states, return_final_states=False,
                **dt_limit_kwargs,
            )
            zxbcdt_flip = flip_valid_time(zxbcdt, seq_lens, channel_first=True) if flip_valid_time else zxbcdt.flip(dims=[1])
            out_b = mamba_split_conv1d_scan_combined(
                zxbcdt_flip,
                rearrange(self.conv1d_b.weight, "d 1 w -> d w"), self.conv1d_b.bias,
                self.dt_bias_b, A_b, D=self.D_b,
                chunk_size=self.chunk_size,
                seq_idx=self._flip_bwd_seq_idx(seq_idx, seq_lens),
                activation=self.activation,
                rmsnorm_weight=self.norm_b.weight, rmsnorm_eps=self.norm_b.eps,
                outproj_weight=None, outproj_bias=None,
                headdim=self.headdim, ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states_b, return_final_states=False,
                **dt_limit_kwargs,
            )
            out_b = flip_valid_time(out_b, seq_lens) if flip_valid_time else out_b.flip(dims=[1])
        else:
            out_f = self._forward_core(
                zxbcdt, self.conv1d, self.dt_bias, A, self.D,
                self.norm, initial_states, seq_idx, dt_limit_kwargs,
            )
            zxbcdt_flip = flip_valid_time(zxbcdt, seq_lens, channel_first=True) if flip_valid_time else zxbcdt.flip(dims=[1])
            out_b = self._forward_core(
                zxbcdt_flip, self.conv1d_b, self.dt_bias_b, A_b, self.D_b,
                self.norm_b, initial_states_b,
                self._flip_bwd_seq_idx(seq_idx, seq_lens),
                dt_limit_kwargs,
            )
            out_b = flip_valid_time(out_b, seq_lens) if flip_valid_time else out_b.flip(dims=[1])

        combined = (0.5 * out_f + 0.5 * out_b) if self.if_devide_out else (out_f + out_b)
        return self.out_proj(combined)

    def _forward_v3(self, u, A, A_b, initial_states, initial_states_b, seq_idx, dt_limit_kwargs, seq_lens=None):
        """v3 (ExtBiMamba2): fully separate parameters per direction."""
        zxbcdt_f = self.in_proj(u)
        u_flip = flip_valid_time(u, seq_lens) if flip_valid_time else u.flip(dims=[1])
        zxbcdt_b = self.in_proj_b(u_flip)

        if self.use_mem_eff_path and mamba_split_conv1d_scan_combined is not None:
            out_f = mamba_split_conv1d_scan_combined(
                zxbcdt_f,
                rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias,
                self.dt_bias, A, D=self.D,
                chunk_size=self.chunk_size, seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias,
                headdim=self.headdim, ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states, return_final_states=False,
                **dt_limit_kwargs,
            )
            out_b = mamba_split_conv1d_scan_combined(
                zxbcdt_b,
                rearrange(self.conv1d_b.weight, "d 1 w -> d w"), self.conv1d_b.bias,
                self.dt_bias_b, A_b, D=self.D_b,
                chunk_size=self.chunk_size,
                seq_idx=self._flip_bwd_seq_idx(seq_idx, seq_lens),
                activation=self.activation,
                rmsnorm_weight=self.norm_b.weight, rmsnorm_eps=self.norm_b.eps,
                outproj_weight=self.out_proj_b.weight, outproj_bias=self.out_proj_b.bias,
                headdim=self.headdim, ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states_b, return_final_states=False,
                **dt_limit_kwargs,
            )
            out_b = flip_valid_time(out_b, seq_lens) if flip_valid_time else out_b.flip(dims=[1])
        else:
            out_f = self._forward_core(
                zxbcdt_f, self.conv1d, self.dt_bias, A, self.D,
                self.norm, initial_states, seq_idx, dt_limit_kwargs,
            )
            out_f = self.out_proj(out_f)

            out_b = self._forward_core(
                zxbcdt_b, self.conv1d_b, self.dt_bias_b, A_b, self.D_b,
                self.norm_b, initial_states_b,
                self._flip_bwd_seq_idx(seq_idx, seq_lens),
                dt_limit_kwargs,
            )
            out_b = self.out_proj_b(out_b)
            out_b = flip_valid_time(out_b, seq_lens) if flip_valid_time else out_b.flip(dims=[1])

        return (0.5 * out_f + 0.5 * out_b) if self.if_devide_out else (out_f + out_b)

    def _forward_core(self, zxbcdt, conv1d, dt_bias, A, D, norm,
                       initial_states, seq_idx, dt_limit_kwargs):
        """Core single-direction forward (non-fused path)."""
        batch, seqlen, _ = zxbcdt.shape

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + dt_bias)

        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(conv1d(xBC.transpose(1, 2)).transpose(1, 2))[:, :seqlen, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(conv1d.weight, "d 1 w -> d w"),
                bias=conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size, D=D, z=None,
            seq_idx=seq_idx, initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        return norm(y, z)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        conv_state_f = torch.zeros(batch_size, conv_dim, self.d_conv, device=device, dtype=conv_dtype)
        ssm_state_f = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=conv_dtype)
        conv_state_b = torch.zeros(batch_size, conv_dim, self.d_conv, device=device, dtype=conv_dtype)
        ssm_state_b = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=conv_dtype)
        return (conv_state_f, ssm_state_f), (conv_state_b, ssm_state_b)


# ============================================================================
# Block: PreNorm residual wrapper (shared by BiMamba & BiMamba2)
# ============================================================================

class Block(nn.Module):
    """PreNorm residual block: Add -> LN -> Mixer -> (optional MLP).

    Supports fused add+norm via Triton kernels for performance.
    """

    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **mixer_kwargs,
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states, self.norm2.weight, self.norm2.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
