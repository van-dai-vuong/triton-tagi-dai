"""
triton-tagi: Tractable Approximate Gaussian Inference on Triton
================================================================

A minimal, GPU-accelerated Python reimplementation of cuTAGI
(https://github.com/lhnguyen102/cuTAGI) using fused Triton kernels.

The surface is deliberately small: the layer set mirrors what is needed to
reproduce cuTAGI's headline examples (regression, MNIST MLP/CNN, CIFAR-10 CNN
and ResNet-18) with numerical parity. Additional layers, optimizers, and
diagnostics live under ``_archive/`` at the repository root.

Modules
-------
- ``layers``  : Bayesian layers
- ``update``  : observation innovation and parameter update rules
- ``network`` : ``Sequential`` network builder
- ``kernels`` : low-level Triton kernels

Numerical precision
-------------------
TF32 matmul is disabled at import time. cuTAGI uses scalar FMA loops
(``__fmaf_rn``) with near-fp64 accuracy; leaving TF32 enabled in
PyTorch/Triton would introduce systematic ~1e-3 errors in the variance
forward pass and break numerical parity.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False

from .base import Layer, LearnableLayer
from .checkpoint import RunDir, load_model
from .hrc_softmax import (
    HierarchicalSoftmax,
    class_to_obs,
    get_predicted_labels,
    labels_to_hrc,
    obs_to_class_probs,
)
from .inference_init import inference_init
from .layers import (
    Add,
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Embedding,
    EvenSoftplus,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2D,
    MultiheadAttentionV2,
    PositionalEncoding,
    RMSNorm,
    ReLU,
    Remax,
    ResBlock,
)
from .network import Sequential
from .param_init import (
    gaussian_param_init,
    he_init,
    init_weight_bias_conv2d,
    init_weight_bias_linear,
    init_weight_bias_norm,
    xavier_init,
)

__version__ = "0.2.0"
__all__ = [
    # ABCs
    "Layer",
    "LearnableLayer",
    # Network
    "Sequential",
    # Layers
    "Add",
    "AvgPool2D",
    "BatchNorm2D",
    "Conv2D",
    "Embedding",
    "EvenSoftplus",
    "Flatten",
    "LayerNorm",
    "Linear",
    "MaxPool2D",
    "MultiheadAttentionV2",
    "PositionalEncoding",
    "RMSNorm",
    "ReLU",
    "Remax",
    "ResBlock",
    # Parameter initialisation
    "he_init",
    "xavier_init",
    "gaussian_param_init",
    "init_weight_bias_linear",
    "init_weight_bias_conv2d",
    "init_weight_bias_norm",
    "inference_init",
    # Hierarchical softmax
    "HierarchicalSoftmax",
    "class_to_obs",
    "labels_to_hrc",
    "obs_to_class_probs",
    "get_predicted_labels",
    # Run management
    "RunDir",
    "load_model",
]
