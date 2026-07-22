"""
Parameter initialization for TAGI neural networks.

Mirrors the cuTAGI C++ initialization (param_init.h / param_init.cpp):
    - He initialization   (He et al., 2015)
    - Xavier initialization (Glorot & Bengio, 2010)
    - Gaussian parameter init  (mean drawn from N(0, scale), variance = (gain * scale)^2)

Each layer type has its own `init_weight_bias_*` helper that computes the
appropriate fan-in / fan-out, picks the scale, and returns (mu_w, var_w, mu_b, var_b).
"""

import math

import torch

# ======================================================================
#  Scale functions
# ======================================================================


def he_init(fan_in: float) -> float:
    """He initialization scale matching cuTAGI's param_init.cpp.

    cuTAGI defines scale = sqrt(1 / fan_in), so Sw = scale^2 = 1/fan_in
    (with gain_w = 1.0).  This matches the formula in:
        Delving Deep into Rectifiers (He et al., 2015)
    as implemented in cuTAGI.

    Args:
        fan_in: Number of input connections.

    Returns:
        scale: Standard deviation for the weight distribution.
    """
    return math.sqrt(1.0 / fan_in)


def xavier_init(fan_in: float, fan_out: float) -> float:
    """Xavier / Glorot initialization scale (Glorot & Bengio, 2010).

    Args:
        fan_in:  Number of input connections.
        fan_out: Number of output connections.

    Returns:
        scale: Standard deviation for the weight distribution.
    """
    return math.sqrt(2.0 / (fan_in + fan_out))


# ======================================================================
#  Gaussian parameter initialization
# ======================================================================


def gaussian_param_init(scale: float, gain: float, shape, device="cpu", generator=None):
    """Initialize TAGI parameters with Gaussian mean and constant variance.

    Matches cuTAGI's ``gaussian_param_init``:
        m[i] ~ N(0, scale)
        S[i] = (gain * scale)^2

    Args:
        scale: Standard deviation for the mean distribution.
        gain:  Multiplication factor for the variance.
        shape: Tensor shape (tuple or int).
        device: Torch device.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        m: Mean tensor.
        S: Variance tensor.
    """
    m = torch.randn(shape, device=device, generator=generator) * scale
    S = torch.full(
        shape if isinstance(shape, tuple) else (shape,), (gain * scale) ** 2, device=device
    )
    return m, S


# ======================================================================
#  Layer-specific initialization (matching cuTAGI)
# ======================================================================


def init_weight_bias_linear(
    input_size: int,
    output_size: int,
    init_method: str = "He",
    gain_w: float = 1.0,
    gain_b: float = 1.0,
    bias: bool = True,
    device="cpu",
    generator=None,
):
    """Initialize weights and biases for a Linear (fully-connected) layer.

    Matches cuTAGI ``init_weight_bias_linear``.

    Args:
        input_size:  Number of input features (fan_in).
        output_size: Number of output features (fan_out).
        init_method: "He" or "Xavier".
        gain_w: Gain multiplier for weight variance.
        gain_b: Gain multiplier for bias variance.
        bias:   Whether to create a bias.
        device: Torch device.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        mu_w, var_w, mu_b, var_b
    """
    if init_method.lower() == "xavier":
        scale = xavier_init(input_size, output_size)
    elif init_method.lower() == "he":
        scale = he_init(input_size)
    else:
        raise ValueError(f"Unsupported init method: {init_method}")

    mu_w, var_w = gaussian_param_init(scale, gain_w, (input_size, output_size), device, generator)

    if bias:
        mu_b, var_b = gaussian_param_init(scale, gain_b, (1, output_size), device, generator)
    else:
        mu_b = torch.zeros(1, output_size, device=device)
        var_b = torch.zeros(1, output_size, device=device)

    return mu_w, var_w, mu_b, var_b


def init_weight_bias_conv2d(
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    init_method: str = "He",
    gain_w: float = 1.0,
    gain_b: float = 1.0,
    device="cpu",
):
    """Initialize weights and biases for a Conv2D layer.

    Matches cuTAGI ``init_weight_bias_conv2d``.

    Args:
        kernel_size:  Spatial kernel size (square).
        in_channels:  Input channels.
        out_channels: Output channels (filters).
        init_method:  "He" or "Xavier".
        gain_w: Gain multiplier for weight variance.
        gain_b: Gain multiplier for bias variance.
        device: Torch device.

    Returns:
        mu_w, var_w, mu_b, var_b
    """
    fan_in = kernel_size**2 * in_channels
    fan_out = kernel_size**2 * out_channels

    if init_method.lower() == "xavier":
        scale = xavier_init(fan_in, fan_out)
    elif init_method.lower() == "he":
        scale = he_init(fan_in)
    else:
        raise ValueError(f"Unsupported init method: {init_method}")

    K = in_channels * kernel_size * kernel_size  # weight matrix rows
    mu_w, var_w = gaussian_param_init(scale, gain_w, (K, out_channels), device)
    mu_b, var_b = gaussian_param_init(scale, gain_b, (1, out_channels), device)

    return mu_w, var_w, mu_b, var_b


def init_weight_bias_norm(
    num_features: int, gain_w: float = 1.0, gain_b: float = 1.0, device="cpu"
):
    """Initialize parameters for a normalization layer (BatchNorm).

    Matches cuTAGI ``init_weight_bias_norm``:
        gamma: mean = 1.0,  variance = scale * gain_w^2
        beta:  mean = 0.0,  variance = scale * gain_b^2

    Args:
        num_features: Number of channels.
        gain_w: Gain multiplier for gamma variance.
        gain_b: Gain multiplier for beta variance.
        device: Torch device.

    Returns:
        mu_gamma, var_gamma, mu_beta, var_beta
    """
    # cuTAGI uses scale = 2 / (in + out); for BN, in == out == num_features
    scale = 2.0 / (num_features + num_features)

    mu_gamma = torch.ones(num_features, device=device)
    var_gamma = torch.full((num_features,), scale * gain_w**2, device=device)

    mu_beta = torch.zeros(num_features, device=device)
    var_beta = torch.full((num_features,), scale * gain_b**2, device=device)

    return mu_gamma, var_gamma, mu_beta, var_beta
