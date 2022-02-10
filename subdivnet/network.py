from typing import List
import jittor as jt
import jittor.nn as nn

from .mesh_ops import MeshAdaptivePool
from .mesh_ops import MeshBatchNorm
from .mesh_ops import MeshConv
from .mesh_ops import MeshDropout
from .mesh_ops import MeshLinear
from .mesh_ops import MeshPool
from .mesh_ops import MeshUnpool
from .mesh_ops import MeshReLU
from .mesh_ops import mesh_concat
from .mesh_ops import mesh_add


class MeshConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        self.mconv1 = MeshConv(in_channels, out_channels, dilation=dilation, bias=False)
        self.mconv2 = MeshConv(out_channels, out_channels, dilation=dilation, bias=False)
        self.bn1 = MeshBatchNorm(out_channels)
        self.bn2 = MeshBatchNorm(out_channels)
        self.relu1 = MeshReLU()
        self.relu2 = MeshReLU()

    def execute(self, mesh):
        mesh = self.mconv1(mesh)
        mesh = self.bn1(mesh)
        mesh = self.relu1(mesh)
        mesh = self.mconv2(mesh)
        mesh = self.bn2(mesh)
        mesh = self.relu2(mesh)
        return mesh


class MeshResIdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = MeshLinear(in_channels, out_channels)
        self.bn1 = MeshBatchNorm(out_channels)
        self.relu = MeshReLU()
        self.conv2 = MeshConv(out_channels, out_channels, dilation=dilation)
        self.bn2 = MeshBatchNorm(out_channels)
        self.conv3 = MeshLinear(out_channels, out_channels)
        self.bn3 = MeshBatchNorm(out_channels)

    def execute(self, mesh):
        identity = mesh

        mesh = self.conv1(mesh)
        mesh = self.bn1(mesh)
        mesh = self.relu(mesh)
        mesh = self.conv2(mesh)
        mesh = self.bn2(mesh)
        mesh = self.conv3(mesh)
        mesh = self.bn3(mesh)

        mesh.feats += identity.feats
        mesh = self.relu(mesh)

        return mesh


class MeshResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv0 = MeshLinear(in_channels, out_channels)
        self.bn0 = MeshBatchNorm(out_channels)
        self.conv1 = MeshLinear(out_channels, out_channels)
        self.bn1 = MeshBatchNorm(out_channels)
        self.relu = MeshReLU()
        self.conv2 = MeshConv(out_channels, out_channels, dilation=dilation)
        self.bn2 = MeshBatchNorm(out_channels)
        self.conv3 = MeshLinear(out_channels, out_channels)
        self.bn3 = MeshBatchNorm(out_channels)

    def execute(self, mesh):
        mesh = self.conv0(mesh)
        mesh = self.bn0(mesh)
        identity = mesh

        mesh = self.conv1(mesh)
        mesh = self.bn1(mesh)
        mesh = self.relu(mesh)
        mesh = self.conv2(mesh)
        mesh = self.bn2(mesh)
        mesh = self.conv3(mesh)
        mesh = self.bn3(mesh)

        mesh.feats += identity.feats
        mesh = self.relu(mesh)

        return mesh


class MeshBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = MeshConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MeshBatchNorm(planes)
        self.conv2 = MeshConv(planes, planes, kernel_size=3, stride=stride, dilation=dilation, bias=False)
        self.bn2 = MeshBatchNorm(planes)
        self.conv3 = MeshConv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MeshBatchNorm(planes * 4)
        self.relu = MeshReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MeshNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, 
        layer_channels: List[int], residual=False, blocks=None, n_dropout=1):
        super(MeshNet, self).__init__()
        self.fc = MeshLinear(in_channels, layer_channels[0])
        self.relu = MeshReLU()

        self.convs = nn.Sequential()
        for i in range(depth):
            if residual:
                self.convs.append(MeshResConvBlock(layer_channels[i], layer_channels[i + 1]))
                for _ in range(blocks[i] - 1):
                    self.convs.append(MeshResConvBlock(layer_channels[i + 1], layer_channels[i + 1]))
            else:
                self.convs.append(MeshConvBlock(layer_channels[i], 
                                                layer_channels[i + 1]))
            self.convs.append(MeshPool('max'))
        self.convs.append(MeshConv(layer_channels[-1], 
                                   layer_channels[-1], 
                                   bias=False))
        self.global_pool = MeshAdaptivePool('max')

        if n_dropout >= 2:
            self.dp1 = nn.Dropout(0.5)

        self.linear1 = nn.Linear(layer_channels[-1], layer_channels[-1], bias=False)
        self.bn = nn.BatchNorm1d(layer_channels[-1])

        if n_dropout >= 1:
            self.dp2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(layer_channels[-1], out_channels)
    

    def execute(self, mesh):
        mesh = self.fc(mesh)
        mesh = self.relu(mesh)

        mesh = self.convs(mesh)

        x = self.global_pool(mesh)

        if hasattr(self, 'dp1'):
            x = self.dp1(x)
        x = nn.relu(self.bn(self.linear1(x)))

        if hasattr(self, 'dp2'):
            x = self.dp2(x)
        x = self.linear2(x)

        return x
