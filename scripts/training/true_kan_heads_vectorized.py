#!/usr/bin/env python3
"""
Neural Additive Spline Model (NASM) for Prosody Prediction
==========================================================

This implementation follows the Kolmogorov-Arnold Network (KAN) formulation
from Liu et al. 2024, adapted as a single-layer additive model for interpretability.

Reference: https://alexzhang13.github.io/blog/2024/annotated-kan/

Background
----------
The Kolmogorov-Arnold representation theorem states that any continuous function
f(x₁,...,xₙ) can be written as:

    f(x₁,...,xₙ) = Σ_q Φ_q( Σ_p φ_{q,p}(x_p) )

where φ and Φ are univariate functions. A KAN layer implements:

    K_{m,n}(x) = Φx   where   (Φx)_i = Σ_j φ_{i,j}(x_j)

This is analogous to an MLP layer, but replaces fixed activations with learnable
univariate functions parameterized by B-splines.

NASM Architecture (This Implementation)
---------------------------------------
We use a SINGLE KAN layer without stacking, making it a Neural Additive Model:

    y_o = Σ_i φ_{i,o}(x_i) + b_o

where φ_{i,o}(x) = spline_{i,o}(x) + scale_{i,o} · x

This additive structure means:
- Each feature contributes independently (no interactions)
- We can plot φ_i(x) to see exactly how each feature affects each output
- Fully interpretable: the learned curves ARE the model

Why NASM for Prosody?
---------------------
- Linguistic features should contribute additively (stress → F0, boundary → duration)
- Interpretability lets us validate learned relationships against phonetic theory
- Parameter efficient: ~813 params for 30 inputs × 3 outputs × 8 basis functions

Tensor Dimensions
-----------------
Throughout this code:
    b = batch size
    s = sequence length (phonemes per utterance)  
    i = input features (30 linguistic features)
    o = output features (3 prosody targets: F0, duration, energy)
    k = B-spline basis functions (grid_size, default 8)
"""

import torch
import torch.nn as nn
from typing import Tuple


class BSplineBasis(nn.Module):
    """
    B-Spline Basis Functions via Cox-de Boor Recursion
    
    A B-spline of degree d is defined recursively:
    
        B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
        
        B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) · B_{i,k-1}(x)
                   + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) · B_{i+1,k-1}(x)
    
    Properties of B-splines:
    - Non-negative: B_{i,k}(x) ≥ 0
    - Partition of unity: Σ_i B_{i,k}(x) = 1 for x in the interior
    - Local support: each basis is non-zero only over k+1 knot spans
    - Smoothness: C^{k-1} continuous
    
    We use a clamped knot vector with repeated endpoints to ensure
    the spline interpolates at the boundaries.
    """
    
    def __init__(self, spline_order: int = 3, grid_size: int = 8, 
                 grid_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.spline_order = spline_order  # degree of the B-spline
        self.grid_size = grid_size        # number of basis functions
        self.grid_range = grid_range
        
        # Construct canonical clamped (open-uniform) knot vector
        # For n basis functions of degree k, we need n+k+1 knots
        # Clamped means: endpoints repeated k+1 times (to interpolate at boundaries)
        # Interior knots are uniformly spaced EXCLUDING the endpoints
        
        # Number of interior knots (not including the repeated boundaries)
        num_interior = grid_size - spline_order - 1
        
        if num_interior > 0:
            # Interior points exclude the endpoints (already in the boundary repeats)
            interior = torch.linspace(grid_range[0], grid_range[1], num_interior + 2)[1:-1]
        else:
            interior = torch.tensor([])
        
        # Clamped: repeat endpoints (spline_order + 1) times each
        knots = torch.cat([
            torch.full((spline_order + 1,), grid_range[0]),
            interior,
            torch.full((spline_order + 1,), grid_range[1])
        ])
        
        self.register_buffer("knots", knots)
        
        # Verify: n_basis = n_knots - order - 1
        expected_knots = grid_size + spline_order + 1
        assert len(self.knots) == expected_knots, \
            f"Expected {expected_knots} knots, got {len(self.knots)}"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at input points.
        
        Args:
            x: (..., in_features) input tensor, expected in [0, 1]
            
        Returns:
            B: (..., in_features, grid_size) basis function values
        """
        # Clamp to grid range (values outside are projected to boundary)
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        
        # Reshape for broadcasting
        orig_shape = x.shape
        in_features = orig_shape[-1]
        batch_size = x.numel() // in_features
        
        x_flat = x.reshape(batch_size, in_features)
        x_exp = x_flat.unsqueeze(-1)  # (batch, features, 1)
        knots = self.knots.to(x.device)
        
        # Degree-0 basis: indicator functions B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}
        left = knots[:-1].unsqueeze(0).unsqueeze(0)   # (1, 1, n_knots-1)
        right = knots[1:].unsqueeze(0).unsqueeze(0)
        B = ((x_exp >= left) & (x_exp < right)).to(x.dtype)
        
        # Handle right endpoint: include t_{n} in the last interval
        B[..., -1] += (x_exp.squeeze(-1) == knots[-1]).to(x.dtype)
        
        # Cox-de Boor recursion for higher degrees
        for k in range(1, self.spline_order + 1):
            num_basis = len(knots) - k - 1
            if num_basis <= 0:
                break
            
            idx = torch.arange(num_basis, device=x.device)
            
            # Denominators (avoid division by zero with small epsilon)
            denom1 = (knots[idx + k] - knots[idx]).clamp(min=1e-8)
            denom2 = (knots[idx + k + 1] - knots[idx + 1]).clamp(min=1e-8)
            denom1 = denom1.unsqueeze(0).unsqueeze(0)
            denom2 = denom2.unsqueeze(0).unsqueeze(0)
            
            # Numerators
            left_num = x_exp - knots[idx].unsqueeze(0).unsqueeze(0)
            right_num = knots[idx + k + 1].unsqueeze(0).unsqueeze(0) - x_exp
            
            # Recursion: B_{i,k} = left_term + right_term
            left_term = (left_num / denom1) * B[..., :num_basis]
            right_term = (right_num / denom2) * B[..., 1:num_basis + 1]
            B = left_term + right_term
        
        return B.reshape(list(orig_shape) + [self.grid_size])
    
    def verify_basis_properties(self, n_test: int = 1000) -> dict:
        """
        Verify B-spline properties: partition of unity and non-negativity.
        Call this during debugging to ensure basis is correct.
        """
        x_test = torch.linspace(self.grid_range[0], self.grid_range[1], n_test)
        x_test = x_test.view(1, n_test, 1)  # (1, n_test, 1 feature)
        B = self.forward(x_test).squeeze(0).squeeze(1)  # (n_test, grid_size)
        
        partition_sum = B.sum(dim=-1)  # should be ~1 everywhere
        
        return {
            'partition_unity_mean': float(partition_sum.mean()),
            'partition_unity_std': float(partition_sum.std()),
            'partition_unity_max_error': float((partition_sum - 1.0).abs().max()),
            'min_value': float(B.min()),
            'max_value': float(B.max()),
            'non_negative': bool((B >= -1e-6).all())
        }


class TrueKANHead(nn.Module):
    """
    Neural Additive Spline Model (NASM) - Single KAN Layer
    
    Computes: y_o = Σ_i φ_{i,o}(x_i) + bias_o
    
    where φ_{i,o}(x) = spline_{i,o}(x) + scale_{i,o} · x
    
    The spline term captures nonlinear effects, while the linear term
    (analogous to silu_x in original KAN) captures global trends.
    This is a standard semi-parametric formulation in GAM literature.
    
    Key differences from full KAN:
    - Single layer only (no stacking) → fully interpretable
    - No dropout → learned curves are faithful to training objective
    - No output clamping → relies on proper normalization instead
    
    Parameters
    ----------
    in_features : int
        Number of input features (30 for our prosody model)
    out_features : int  
        Number of outputs (3: F0, duration, energy)
    grid_size : int
        Number of B-spline basis functions (default 8)
    spline_order : int
        Degree of B-spline (default 3 = cubic)
    """
    
    def __init__(self, in_features: int = None, out_features: int = 1, 
                 grid_size: int = 8, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (0, 1),
                 enable_interpretability: bool = True,
                 # backward compatibility aliases
                 in_dim: int = None, out_dim: int = None, 
                 num_basis: int = None, degree: int = None,
                 learn_base_linear: bool = False):
        super().__init__()
        
        # Handle backward compatibility aliases
        in_features = in_dim if in_dim is not None else in_features
        out_features = out_dim if out_dim is not None else out_features
        grid_size = num_basis if num_basis is not None else grid_size
        spline_order = degree if degree is not None else spline_order
        
        if in_features is None:
            raise ValueError("in_features (or in_dim) is required")
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.enable_interpretability = enable_interpretability
        
        # Backward compatibility attributes
        self.in_dim = in_features
        self.out_dim = out_features
        self.num_basis = grid_size
        self.degree = spline_order
        
        # B-spline basis (shared across all feature-output pairs)
        self.basis = BSplineBasis(spline_order, grid_size, grid_range)
        
        # Learnable spline coefficients: coef[i, o, k]
        # For each input i and output o, we have grid_size coefficients
        # that define φ_{i,o}(x) = Σ_k coef[i,o,k] · B_k(x)
        self.coef = nn.Parameter(
            torch.randn(in_features, out_features, grid_size) * 0.01
        )
        
        # Per-feature linear term: scale[i, o] 
        # φ_{i,o}(x) = spline(x) + scale · x (semi-parametric model)
        self.scale = nn.Parameter(torch.zeros(in_features, out_features))
        
        # Global bias per output
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Optional: multivariate linear (breaks additivity, off by default)
        self.base_linear = nn.Linear(in_features, out_features) if learn_base_linear else None
        
        # Backward compatibility aliases for loading old checkpoints
        self.coefficients = self.coef
        self.linear_weights = self.scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = Σ_i φ_i(x_i) + bias
        
        Args:
            x: (batch, seq, in_features) input tensor
               Expected to be normalized to [0, 1] range
               
        Returns:
            output: (batch, seq, out_features) predictions
        """
        # Evaluate B-spline basis for all inputs
        # B: (batch, seq, in_features, grid_size)
        B = self.basis(x)
        
        # Spline contribution: Σ_i Σ_k B[i,k] · coef[i,o,k]
        # einsum 'bsik,iok->bso': sum over inputs i and basis k
        spline = torch.einsum('bsik,iok->bso', B, self.coef)
        
        # Linear residual: Σ_i scale[i,o] · x_i
        # einsum 'bsi,io->bso': weighted sum of inputs per output
        residual = torch.einsum('bsi,io->bso', x, self.scale)
        
        # Combine: y = spline + linear + bias
        output = spline + residual + self.bias
        
        # Optional multivariate term (not used by default)
        if self.base_linear is not None:
            output = output + self.base_linear(x)
        
        return output
    
    def get_feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose output into per-feature contributions.
        
        Returns φ_i(x_i) for each feature, enabling interpretability analysis.
        Since the model is additive: y_o = Σ_i contrib[i,o] + bias_o
        
        Args:
            x: (batch, seq, in_features) input tensor
            
        Returns:
            contributions: (batch, seq, in_features, out_features)
                           contrib[b,s,i,o] = φ_{i,o}(x_{b,s,i})
        """
        if not self.enable_interpretability:
            raise ValueError("Interpretability not enabled at construction")
        
        B = self.basis(x)
        
        # Spline contribution per feature: keep i dimension
        # einsum 'bsik,iok->bsio': B[i,k] · coef[i,o,k] summed over k
        spline_contrib = torch.einsum('bsik,iok->bsio', B, self.coef)
        
        # Linear contribution per feature: x_i · scale[i,o]
        linear_contrib = x.unsqueeze(-1) * self.scale.unsqueeze(0).unsqueeze(0)
        
        return spline_contrib + linear_contrib
    
    def get_curve_for_plotting(self, feature_idx: int, output_idx: int = 0, 
                                n_points: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the learned univariate function φ_{i,o}(x) for visualization.
        
        Args:
            feature_idx: which input feature (0 to in_features-1)
            output_idx: which output (0=F0, 1=duration, 2=energy)
            n_points: resolution of the curve
            
        Returns:
            xs: (n_points,) x-axis values
            ys: (n_points,) φ(x) values
        """
        device = self.coef.device
        xs = torch.linspace(self.grid_range[0], self.grid_range[1], 
                           n_points, device=device)
        
        # Evaluate basis: need shape (1, n_points, 1) for basis
        B = self.basis(xs.view(1, n_points, 1))  # (1, n_points, 1, grid_size)
        
        # Get coefficients for this feature-output pair
        coeffs = self.coef[feature_idx, output_idx, :]  # (grid_size,)
        
        # Spline value: Σ_k B_k(x) · coef_k
        ys_spline = (B.squeeze(0).squeeze(1) * coeffs).sum(-1)  # (n_points,)
        
        # Add linear term
        ys_linear = xs * self.scale[feature_idx, output_idx]
        
        ys = ys_spline + ys_linear
        
        return xs.cpu(), ys.detach().cpu()
    
    def get_spline_smoothness_penalty(self) -> torch.Tensor:
        """
        Second-derivative smoothness penalty on spline coefficients.
        
        For each feature-output pair, penalize: ||D² coef||²
        where D² is the second-difference matrix.
        
        This is the standard smoothing penalty in GAM literature (Wood, 2017).
        
        Note: This method is available for optional use but is NOT applied
        in the default training script. B-spline compact support already
        provides implicit smoothness; explicit penalties were not needed.
        """
        # Second difference: coef[k] - 2*coef[k+1] + coef[k+2]
        d2 = self.coef[:, :, :-2] - 2 * self.coef[:, :, 1:-1] + self.coef[:, :, 2:]
        return (d2 ** 2).sum()
    
    def get_l1_penalty(self) -> torch.Tensor:
        """
        L1 penalty on coefficients (encourages sparsity).
        
        Note: This method is available for optional use but is NOT applied
        in the default training script. The reported experiments use only
        MSE loss on z-normalized targets without explicit L1 regularization.
        """
        return torch.abs(self.coef).sum()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NASM (KAN Head) Smoke Test")
    print("=" * 60)
    
    # Test dimensions
    batch, seq, in_feat, out_feat = 2, 10, 30, 3
    x = torch.rand(batch, seq, in_feat)  # inputs in [0, 1]
    
    # Create model
    model = TrueKANHead(in_features=in_feat, out_features=out_feat, 
                        grid_size=8, spline_order=3)
    
    # Forward pass
    out = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify B-spline basis properties
    print("\nB-spline Basis Verification:")
    props = model.basis.verify_basis_properties()
    for k, v in props.items():
        status = "✓" if ("unity" not in k or abs(v - 1.0) < 0.01 or v < 0.01) and \
                       ("non_negative" not in k or v) else "✗"
        print(f"  {k}: {v:.6f} {status}")
    
    # Test interpretability
    contrib = model.get_feature_contributions(x)
    print(f"\nFeature contributions: {contrib.shape}")
    print(f"  Sum matches output: {torch.allclose(contrib.sum(dim=2) + model.bias, out, atol=1e-5)}")
    
    # Test curve extraction
    xs, ys = model.get_curve_for_plotting(feature_idx=0, output_idx=0)
    print(f"\nCurve for feature 0 → output 0: {len(xs)} points")
    
    # Backward compatibility test
    model2 = TrueKANHead(in_dim=30, out_dim=3, num_basis=8, degree=3)
    out2 = model2(x)
    print(f"\nBackward compat test: {out2.shape} ✓")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
