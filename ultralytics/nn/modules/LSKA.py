import torch
import torch.nn as nn
from .block import PSABlock,C2PSA  #确保能引用到原YOLO的类

#----------------------------------------------------------
#1.LSKA核心模块（动态版）
class LSKA(nn.Module):
    def __init__(self, dim,k_size=11,dilation=2):
        """
        LSKA注意力机制:大核可分离卷积注意力
        
        参数：
            dim: 输入通道数
            k_size: 最大感受也大小(k)
            dilation: 膨胀率(d)，控制感受野扩展程度
        """
        super().__init__()
        self.k_size = k_size
        self.d = dilation

        #--第一步，计算卷积核参数---
        # 深度卷积核大小(ks_dw) = 2d - 1
        ks_dw = 2 * self.d - 1
        # 空洞卷积核大小(ks_dilated) = floor(k / d)
        ks_dilated = k_size // self.d

        #--第二步，计算padding(保证输入输出尺寸一致)---
        pad_dw = (ks_dw - 1) // 2
        pad_dilated = self.d * (ks_dilated - 1) // 2

        # --第三步，定义网络层---
        # 1.深度卷积(捕捉局部特征)
        # 水平方向(1 * ks)
        self.conv0_h = nn.Conv2d(dim, dim, kernel_size=(1, ks_dw), padding=(0, pad_dw), groups=dim)
        # 垂直方向(ks * 1)
        self.conv0_v = nn.Conv2d(dim, dim, kernel_size=(ks_dw, 1), padding=(pad_dw, 0), groups=dim)

        #2.深度空洞卷积(捕捉长距离/全局特征)
        # 水平方向(1 * ks)，带膨胀
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, ks_dilated), stride=1,
                                        padding=(0, pad_dilated), groups=dim, dilation=self.d)
        
        # 垂直方向(ks * 1)，带膨胀
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(ks_dilated, 1), stride=1,
                                        padding=(pad_dilated, 0), groups=dim, dilation=self.d)
        
        #3.通道融合(1*1卷积)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()  # 保存输入，用于最后的注意力加权
        attn = self.conv0_h(x)  # 水平局部卷积
        attn = self.conv0_v(attn)  # 垂直局部卷积
        attn = self.conv_spatial_h(attn)  # 水平空洞卷积
        attn = self.conv_spatial_v(attn)  # 垂直空洞卷积
        attn = self.conv1(attn)  # 通道融合
        return u * attn  # 输出 = 输入 * 注意力图
        
class PSABlock_LSKA(PSABlock):
    def __init__(self, c, qk_dim=16,pdim=32,shortcut=True) -> None:
        """继承自YOLO 的PSABlock,将内部的注意力替换为LSKA"""
        super().__init__(c, shortcut=shortcut)

        self.ffn = LSKA(c, k_size=11, dilation=2)  # 替换原有的注意力机制为LSKA
                

class C2PSA_LSKA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        """
        C2PSA模块的LSKA版本
        参数：
            c1: 输入通道数
            c2: 输出通道数
            n: 堆叠层数
            e: 扩张比例
        """
        super().__init__(c1, c2, n=n, e=e)
        assert c1 == c2 # 输入输出通道数必须相同以适配LSKA模块
        self.c = int(c1 * e)  # 中间通道数

        # 使用列表推导式生成 n 个PSABlock_LSKA
        self.m = nn.Sequential(*(PSABlock_LSKA(self.c,qk_dim=16,pdim=32) for _ in range(n)))
            
