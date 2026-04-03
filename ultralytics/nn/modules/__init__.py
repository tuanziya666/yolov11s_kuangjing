# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3k2Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
)
from .akconv import AKConv, Bottleneck_AKConv, C3k_AKConv, C3k2_AKConv
from .carafe import CARAFE
from .conv import (
    CBAM,
    BiFPNConcat2,
    BiFPNConcat3,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .decm import DECM
from .dysample import DySample
from .ema import EMA
from .fcanet import FcaNet, MultiSpectralAttentionLayer, MultiSpectralDCTLayer
from .head import Classify, Detect, DyDetect, OBB, Pose, RTDETRDecoder, Segment, TDDetect, WorldDetect, v10Detect
from .hspan_dysample import Add, ChannelAttention_HSFPN, Multiply
from .illum_dwt import AIE, DWTDown
from .LSKA import C2PSA_LSKA, LSKA, PSABlock_LSKA
from .lsccm import LSCCM
from .masa import MaSA
from .minelgl import MineLGL
from .msee import C3k2MSEE, EdgeEnhancer, MSEE
from .sfem import HaarDWT, HaarIDWT, SFEM
from .spdgf_ds import C3k2DSConv, DSConv, GFBlock, GlobalFilter, SPDConv, SPDGFDown, SpaceToDepth
from .wfu import HaarWavelet, WFU
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AKConv",
    "Bottleneck_AKConv",
    "C3k_AKConv",
    "C3k2_AKConv",
    "CARAFE",
    "DECM",
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "BiFPNConcat2",
    "BiFPNConcat3",
    "DySample",
    "ChannelAttention_HSFPN",
    "Multiply",
    "Add",
    "EMA",
    "FcaNet",
    "MultiSpectralAttentionLayer",
    "MultiSpectralDCTLayer",
    "AIE",
    "DWTDown",
    "LSKA",
    "PSABlock_LSKA",
    "C2PSA_LSKA",
    "LSCCM",
    "MaSA",
    "MineLGL",
    "EdgeEnhancer",
    "MSEE",
    "HaarDWT",
    "HaarIDWT",
    "SFEM",
    "HaarWavelet",
    "WFU",
    "SpaceToDepth",
    "SPDConv",
    "GlobalFilter",
    "GFBlock",
    "SPDGFDown",
    "DSConv",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "C3k2Ghost",
    "C3k2MSEE",
    "C3k2DSConv",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "DyDetect",
    "TDDetect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
)
