"""
Shared compute kernels used across the TAGI library.

Contains the fused variance-forward and backward-delta matmuls that are the
computational workhorses behind the Linear layer. These were originally fused
Triton kernels; this is a pure-PyTorch port that runs on CPU (and CUDA/MPS if
available), keeping the exact same math and function signatures.

Variance formula
----------------
Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb

The two-matmul grouping mirrors cuTAGI's inner loop which accumulates
    sum_var += (mw²+Sw)*Sa + Sw*ma²
per input element k, minimising fp32 accumulation differences vs cuTAGI's
FMA-fused kernel.
"""

import torch


# ======================================================================
#  Fused variance forward
#  Computes  Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb
# ======================================================================


def triton_fused_var_forward(ma, Sa, mw, Sw, Sb):
    """
    Fused variance forward pass.

    Computes the pre-activation variance:
        Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb

    Parameters
    ----------
    ma : Tensor (B, K)   activation means
    Sa : Tensor (B, K)   activation variances
    mw : Tensor (K, N)   weight means
    Sw : Tensor (K, N)   weight variances
    Sb : Tensor (1, N)   bias variances

    Returns
    -------
    Sz : Tensor (B, N)   pre-activation variances
    """
    # ma² @ Sw  and  Sa @ (mw² + Sw): the two-matmul grouping mirrors cuTAGI.
    Sz = torch.matmul(ma * ma, Sw) + torch.matmul(Sa, mw * mw + Sw)
    Sz = Sz + Sb.view(1, -1)
    return Sz


# ======================================================================
#  Fused backward-delta
#  Computes  d_ma = dmz @ mw^T       (mean delta propagation)
#            d_Sa = dSz @ (mw²)^T    (var  delta propagation)
# ======================================================================


def triton_fused_backward_delta(dmz, dSz, mw):
    """
    Fused backward delta propagation.

    Computes the input-space deltas by back-projecting through weights:
        d_ma = dmz @ mw^T        (mean delta)
        d_Sa = dSz @ (mw²)^T     (variance delta)

    Parameters
    ----------
    dmz : Tensor (B, N)   mean delta from next layer
    dSz : Tensor (B, N)   variance delta from next layer
    mw  : Tensor (K, N)   weight means

    Returns
    -------
    d_ma : Tensor (B, K)  mean delta to propagate backward
    d_Sa : Tensor (B, K)  variance delta to propagate backward
    """
    d_ma = torch.matmul(dmz, mw.t())
    d_Sa = torch.matmul(dSz, (mw * mw).t())
    return d_ma, d_Sa


# ======================================================================
#  Fused weight-gradient
#  Computes  grad_mw = a^T  @  dmz          (weight mean gradient)
#            grad_Sw = (a²)^T @ dSz         (weight variance gradient)
# ======================================================================


def triton_fused_weight_grad(a, dmz, dSz):
    """
    Fused weight-gradient computation.

        grad_mw = a^T  @ dmz        (weight mean gradient)
        grad_Sw = (a²)^T @ dSz      (weight variance gradient)

    Parameters
    ----------
    a   : Tensor (M, K)  activations cached from forward (ma_in or patches_ma)
    dmz : Tensor (M, N)  output mean deltas
    dSz : Tensor (M, N)  output variance deltas

    Returns
    -------
    grad_mw : Tensor (K, N)
    grad_Sw : Tensor (K, N)
    """
    grad_mw = torch.matmul(a.t(), dmz)
    grad_Sw = torch.matmul((a * a).t(), dSz)
    return grad_mw, grad_Sw
