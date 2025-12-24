#!/usr/bin/env python3
# Vectorized B-spline KAN head for prosody prediction
# Based on Liu et al. 2024 KAN formulation
#
# follows https://alexzhang13.github.io/blog/2024/annotated-kan/ for:
# - B-spline basis via Cox-de Boor recursion (implementation variant, not textbook-exact)
# - clamped knot vector with spline_order repeated endpoints
# - learnable spline coefficients as nn.Parameter
# - per-feature linear residual (simpler than original silu_x, no nonlinearity)
#
# differs from full KAN (NASM architecture instead):
# - single layer only: y = sum_i phi_i(x_i) + bias
# - no stacking: full KAN does y = Phi_L(...Phi_2(Phi_1(x)))
# - no feature interactions: each input has independent spline
# - fully interpretable: can plot phi_i(x) for each feature
#
# why NASM for prosody prediction:
# - interpretability: can see exactly how each linguistic feature affects F0/duration/energy
# - additive structure matches linguistic intuition (features contribute independently)
# - parameter efficient: ~813 params for 30 inputs × 3 outputs × 8 basis (vs thousands for stacked KAN)
# - Kolmogorov-Arnold theorem only guarantees 2-layer form anyway
#
# tensor dimensions used throughout:
#   b = batch size
#   s = sequence length (phonemes per utterance)
#   i = input features (30 linguistic features)
#   o = output features (3 prosody targets: F0, duration, energy)
#   k = spline basis functions (grid_size, default 8)
#
# core computation (einsum 'bsik,iok->bso'):
#   B[b,s,i,k] = B-spline basis evaluated at input x[b,s,i]
#   coef[i,o,k] = learned coefficients defining phi_i for output o
#   output[b,s,o] = sum_i sum_k B[b,s,i,k] * coef[i,o,k]
#                 = sum_i phi_i(x_i)  (the additive model)

import torch
import torch.nn as nn
from typing import Tuple

    
class BSplineBasis(nn.Module):
    # B-spline basis functions using Cox-de Boor recursion
    
    def __init__(self, spline_order: int = 3, grid_size: int = 8, grid_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range
        
        # knot vector: [left^k, interior, right^k]
        num_knots = grid_size + spline_order + 1
        interior = torch.linspace(grid_range[0], grid_range[1], grid_size - spline_order + 1)
        knots = torch.cat([
            torch.full((spline_order,), grid_range[0]),
            interior,
            torch.full((spline_order,), grid_range[1])
        ])
        self.register_buffer("knots", knots)
        assert len(self.knots) == num_knots
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) → B: (..., in_features, grid_size)
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        orig_shape = x.shape
        in_features = orig_shape[-1]
        batch_size = x.numel() // in_features
        
        x_flat = x.reshape(batch_size, in_features)
        x_exp = x_flat.unsqueeze(-1)
        knots = self.knots.to(x.device)
        
        # degree-0 basis (indicator functions)
        left = knots[:-1].unsqueeze(0).unsqueeze(0)
        right = knots[1:].unsqueeze(0).unsqueeze(0)
        B = ((x_exp >= left) & (x_exp < right)).to(x.dtype)
        B[..., -1] += (x_exp.squeeze(-1) == knots[-1]).to(x.dtype)
        
        # Cox-de Boor recursion for higher degrees
        for k in range(1, self.spline_order + 1):
            num_basis = len(knots) - k - 1
            if num_basis <= 0:
                break
            
            idx = torch.arange(num_basis, device=x.device)
            denom1 = (knots[idx + k] - knots[idx]).clamp(min=1e-6).unsqueeze(0).unsqueeze(0)
            denom2 = (knots[idx + k + 1] - knots[idx + 1]).clamp(min=1e-6).unsqueeze(0).unsqueeze(0)
            
            left_num = x_exp - knots[idx].unsqueeze(0).unsqueeze(0)
            right_num = knots[idx + k + 1].unsqueeze(0).unsqueeze(0) - x_exp
            
            left_term = (left_num / denom1) * B[..., :num_basis]
            right_term = (right_num / denom2) * B[..., 1:num_basis+1]
            B = torch.clamp(left_term + right_term, -1e6, 1e6)
            
        return B.reshape(list(orig_shape) + [self.grid_size])


class TrueKANHead(nn.Module):
    # Pure additive spline model: y = sum_i phi_i(x_i) + bias
    # This is a single KAN layer without stacking (interpretable)
    
    def __init__(self, in_features: int = None, out_features: int = 1, grid_size: int = 8, 
                 spline_order: int = 3, grid_range: Tuple[float, float] = (-1, 1),
                 enable_interpretability: bool = True, learn_base_linear: bool = False,
                 # aliases for backward compatibility
                 in_dim: int = None, out_dim: int = None, num_basis: int = None, degree: int = None):
        super().__init__()
        
        # handle aliases for backward compatibility
        in_features = in_dim if in_dim is not None else in_features
        out_features = out_dim if out_dim is not None else out_features
        if in_features is None:
            raise ValueError("in_features (or in_dim) is required")
        grid_size = num_basis if num_basis is not None else grid_size
        spline_order = degree if degree is not None else spline_order
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.enable_interpretability = enable_interpretability
        
        # aliases for backward compatibility
        self.in_dim = in_features
        self.out_dim = out_features
        self.num_basis = grid_size
        self.degree = spline_order
        
        self.basis = BSplineBasis(spline_order, grid_size, grid_range)
        
        # spline coefficients: coef[i,o,k] for input i, output o, basis k
        self.coef = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.01)
        
        # per-feature linear (residual connection like silu_x in original KAN)
        self.scale = nn.Parameter(torch.zeros(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # optional base linear (breaks additivity, not used by default)
        self.base_linear = nn.Linear(in_features, out_features) if learn_base_linear else None
        
        self.dropout = nn.Dropout(0.1)
        self.l1_lambda = 5e-4
        
        # backward compatibility aliases
        self.coefficients = self.coef
        self.linear_weights = self.scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_features) → (batch, seq, out_features)
        
        # map input [0,1] → grid_range (out-of-range values clamped in basis)
        lo, hi = self.grid_range
        x_mapped = x * (hi - lo) + lo
        
        # spline output: sum_i phi_i(x_i)
        B = self.basis(x_mapped)  # (b, s, i, k) = basis functions per input feature
        # einsum 'bsik,iok->bso': for each output o, sum over inputs i and basis k
        # this computes sum_i (sum_k B[i,k] * coef[i,o,k]) = sum_i phi_i(x_i)
        spline = torch.einsum('bsik,iok->bso', B, self.coef)
        spline = torch.clamp(spline, -10, 10)
        
        # residual linear: sum_i scale_i * x_i (like silu_x in original KAN)
        # einsum 'bsi,io->bso': weighted sum of inputs per output
        residual = torch.einsum('bsi,io->bso', x, self.scale)
        residual = torch.clamp(residual, -10, 10)
        
        output = spline + residual + self.bias
        
        # optional base linear (multivariate, breaks additivity)
        if self.base_linear is not None:
            output = output + torch.clamp(self.base_linear(x), -10, 10)
        
        output = torch.clamp(output, -50, 50)
        output = self.dropout(output)
        
        # NaN safety
        if torch.isnan(output).any():
            output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
        return output
    
    def get_l1_loss(self, l1_lambda=None):
        # L1 regularization on spline coefficients (encourages sparsity)
        l1_lambda = l1_lambda or self.l1_lambda
        return l1_lambda * torch.sum(torch.abs(self.coef))
    
    def get_feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
        # returns phi_i(x_i) for each feature: (batch, seq, in_features, out_features)
        if not self.enable_interpretability:
            raise ValueError("Interpretability not enabled")
        
        lo, hi = self.grid_range
        x_mapped = x * (hi - lo) + lo
        B = self.basis(x_mapped)
        
        # einsum 'bsik,iok->bsio': keep i dimension to see each feature's contribution
        # output[b,s,i,o] = sum_k B[b,s,i,k] * coef[i,o,k] = phi_i(x_i) for output o
        spline_contrib = torch.einsum('bsik,iok->bsio', B, self.coef)
        linear_contrib = x.unsqueeze(-1) * self.scale.unsqueeze(0).unsqueeze(0)
        
        return spline_contrib + linear_contrib  # (b, s, i, o)
    
    def get_curve_for_plotting(self, feature_idx: int, output_idx: int = 0, n_points: int = 200):
        # returns (xs, ys) for plotting the learned univariate function phi_i
        xs = torch.linspace(self.grid_range[0], self.grid_range[1], n_points, device=self.coef.device)
        B = self.basis(xs.view(1, n_points, 1))
        coeffs = self.coef[feature_idx, output_idx, :].view(1, 1, -1)
        ys = (B.squeeze(2) * coeffs).sum(-1).squeeze(0)
        return xs.cpu(), ys.detach().cpu()


if __name__ == "__main__":
    # smoke test
    x = torch.randn(2, 10, 30)
    kan = TrueKANHead(in_features=30, out_features=3, grid_size=8, spline_order=3)
    out = kan(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in kan.parameters()):,}")
    
    # test backward compatibility
    kan2 = TrueKANHead(in_dim=30, out_dim=3, num_basis=8, degree=3)
    out2 = kan2(x)
    print(f"Backward compat test: {out2.shape}")
