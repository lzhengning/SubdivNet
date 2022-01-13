from typing import Optional
import jittor as jt
import jittor.nn as nn

from .mesh_tensor import MeshTensor
from .mesh_ops import MeshAdaptivePool
from .mesh_ops import MeshBatchNorm
from .mesh_ops import MeshConv
from .mesh_ops import MeshDropout
from .mesh_ops import MeshLinear
from .mesh_ops import MeshPool
from .mesh_ops import MeshUnpool
from .mesh_ops import MeshReLU
from .mesh_ops import mesh_concat


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


class MeshVanillaUnet(nn.Module):
    def __init__(self, in_channels, out_channels, upsample='nearest') -> None:
        super().__init__()

        self.fc1 = MeshLinear(in_channels, 32)
        self.relu = MeshReLU()
        self.pool = MeshPool('max')
        self.unpool = MeshUnpool(upsample)

        self.enc1 = MeshConvBlock(32, 64)
        self.enc2 = MeshConvBlock(64, 128)
        self.enc3 = MeshConvBlock(128, 128)

        self.mid_conv = MeshConvBlock(128, 128)

        self.dec3 = MeshConvBlock(256, 128)
        self.dec2 = MeshConvBlock(256, 64)
        self.dec1 = MeshConvBlock(128, 64)

        self.fc2 = nn.Sequential(
            MeshDropout(0.5),
            MeshLinear(64, 64),
            MeshDropout(0.1),
            MeshLinear(64, out_channels)
        )

    def execute(self, mesh):
        mesh = self.fc1(mesh)
        mesh = self.relu(mesh)

        enc_mesh1 = self.enc1(mesh)
        enc_mesh2 = self.enc2(self.pool(enc_mesh1))
        enc_mesh3 = self.enc3(self.pool(enc_mesh2))

        mid_mesh = self.pool(enc_mesh3)
        mid_mesh = self.mid_conv(mid_mesh)

        dec_mesh3 = self.unpool(mid_mesh, ref_mesh=enc_mesh3)
        dec_mesh3 = self.dec3(mesh_concat([dec_mesh3, enc_mesh3]))
        dec_mesh2 = self.unpool(dec_mesh3, ref_mesh=enc_mesh2)
        dec_mesh2 = self.dec2(mesh_concat([dec_mesh2, enc_mesh2]))
        dec_mesh1 = self.unpool(dec_mesh2, ref_mesh=enc_mesh1)
        dec_mesh1 = self.dec1(mesh_concat([dec_mesh1, enc_mesh1]))

        out_mesh = self.fc2(dec_mesh1)

        return out_mesh.feats


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[MeshPool] = None
    ):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = MeshConv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation)
        self.bn1 = MeshBatchNorm(planes)
        self.relu = MeshReLU()
        self.conv2 = MeshConv(planes, planes, kernel_size=3, dilation=dilation)
        self.bn2 = MeshBatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, mesh):
        identity = mesh

        out = self.conv1(mesh)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(mesh)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        dilation: int = 1,
        downsample: Optional[MeshPool] = None):
        super().__init__()
        self.conv1 = MeshConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MeshBatchNorm(planes)
        self.conv2 = MeshConv(planes, planes, kernel_size=3, stride=stride,
                              dilation=dilation, bias=False)
        self.bn2 = MeshBatchNorm(planes)
        self.conv3 = MeshConv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MeshBatchNorm(planes * 4)
        self.relu = MeshReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, mesh):
        residual = mesh

        out = self.conv1(mesh)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(mesh)

        out += residual
        out = self.relu(out)

        return out


class MeshResNet(nn.Module):
    def __init__(self, in_channels, block, layers):
        self.inplanes = 64
        super().__init__()
        
        blocks = [1, 2, 4]
        output_stride = 16
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = MeshConv(in_channels, 64, kernel_size=5, stride=2, bias=False)
        self.bn1 = MeshBatchNorm(64)
        self.relu = MeshReLU()
        # self.maxpool = MeshPool('max')

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MeshConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MeshBatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MeshConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MeshBatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def execute(self, mesh):
        x = self.conv1(mesh)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        mid_ref_mesh = x
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat, mid_ref_mesh


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation):
        super().__init__()
        self.atrous_conv = MeshConv(inplanes, planes, kernel_size=kernel_size,
                                    stride=1, dilation=dilation, bias=False)
        self.bn = MeshBatchNorm(planes)
        self.relu = MeshReLU()

    def execute(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, globalpool='mean'):
        super().__init__()
        inplanes = 512
        planes = 128
        dilations = [1, 6, 12, 18]

        self.aspp1 = ASPPModule(inplanes, planes, 1, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, planes, 3, dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, planes, 3, dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, planes, 3, dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(MeshAdaptivePool(globalpool),
                                             nn.Linear(inplanes, planes, bias=False),
                                             nn.BatchNorm(planes),
                                             nn.ReLU())
        self.conv1 = MeshConv(planes * 5, planes, 1, bias=False)
        self.bn1 = MeshBatchNorm(planes)
        self.relu = MeshReLU()
        self.dropout = MeshDropout(0.5)

    def execute(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = x5.broadcast([x.N, 128, x.F], [2])
        x5 = x.updated(x5)
        x = mesh_concat((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class MeshDeeplabDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = 'resnet'
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = MeshConv(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = MeshBatchNorm(48)
        self.relu = MeshReLU()
        self.last_conv = nn.Sequential(MeshConv(48 + 128, 128, kernel_size=3, stride=1, bias=False),
                                       MeshBatchNorm(128),
                                       MeshReLU(),
                                       MeshDropout(0.5),
                                       MeshConv(128, 128, kernel_size=3, stride=1, bias=False),
                                       MeshBatchNorm(128),
                                       MeshReLU(),
                                       MeshDropout(0.1),
                                       MeshConv(128, num_classes, kernel_size=1, stride=1))
        self.unpool = MeshUnpool('bilinear')

    def execute(self, x, low_level_feat, mid_ref_mesh):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = self.unpool(x, ref_mesh=mid_ref_mesh)
        x = self.unpool(x, ref_mesh=low_level_feat)
        x = mesh_concat([x, low_level_feat])
        x = self.last_conv(x)

        return x


class MeshDeepLab(nn.Module):
    def __init__(self, in_channels, out_channels, backbone='resnet18', globalpool='mean'):
        super().__init__()

        if backbone == 'resnet18':
            resnet_layers = [2, 2, 2, 2]
        elif backbone == 'resnet50':
            resnet_layers = [3, 4, 6, 3]
        else:
            raise Exception('Unknown resnet architecture')
        
        self.backbone = MeshResNet(in_channels, BasicBlock, resnet_layers)
        self.aspp = ASPP(globalpool)
        self.decoder = MeshDeeplabDecoder(out_channels)
        self.unpool = MeshUnpool('bilinear')

    def execute(self, mesh: MeshTensor):
        x, low_level_feat, mid_ref_mesh = self.backbone(mesh)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, mid_ref_mesh)
        x = self.unpool(x, ref_mesh=mesh)

        return x.feats
