from typing import List

from .mesh_tensor import MeshTensor

import jittor as jt
import jittor.nn as nn

jt.cudnn.set_max_workspace_ratio(0.01)


class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=True):
        ''' Now, such convolutions patterns are implemented:
            * kernel size = 1, stride = [1, 2]
            * kernel size = 3, dilation = %any%, stride = [1, 2]
            * kernel size = 5, no dilation, stride = [1, 2]
            Note that the valid stride is determined by the subdivision connectivity of the input data (see Section 3.3.4). 
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        assert self.kernel_size % 2 == 1

        if self.kernel_size == 1:
            assert dilation == 1
            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        else:
            kernel_size = 4
            self.conv2d = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=bias)

        assert self.stride in [1, 2]


    def execute(self, mesh_tensor: MeshTensor):
        if self.in_channels != mesh_tensor.C:
            raise Exception(f'feature dimension is {mesh_tensor.C}, but conv kernel dimension is {self.in_channels}')

        if self.kernel_size == 1:           # Simple Convolution
            feats = mesh_tensor.feats
            if self.stride == 2:
                N, C, F = mesh_tensor.shape
                feats = feats.reindex([N, C, F // 4], [
                    'i0', 'i1', 'i2 + @e0(i0) * 3'
                ], extras=[mesh_tensor.Fs // 4])
                mesh_tensor = mesh_tensor.inverse_loop_pool(pooled_feats=feats)
            y = self.conv1d(feats)
        else:                               # General Convolution
            CKP = mesh_tensor.convolution_kernel_pattern(self.kernel_size, self.dilation)
            K = CKP.shape[2]

            conv_feats = mesh_tensor.feats.reindex(
                shape=[mesh_tensor.N, self.in_channels, mesh_tensor.F, K],
                indexes=[
                    'i0',
                    'i1',
                    '@e0(i0, i2, i3)'
                ],
                extras=[CKP, mesh_tensor.Fs],
                overflow_conditions=['i2 >= @e1(i0)'],
                overflow_value=0,
            )                       # [N, C, F, K]

            y0 = mesh_tensor.feats

            if self.stride == 2:
                N, C, F = mesh_tensor.shape
                conv_feats = conv_feats.reindex([N, C, F // 4, K], [
                    'i0', 'i1', 'i2 + @e0(i0) * 3', 'i3'
                ], extras=[mesh_tensor.Fs // 4])
                y0 = y0.reindex([N, C, F // 4], [
                    'i0', 'i1', 'i2 + @e0(i0) * 3'
                ], extras=[mesh_tensor.Fs // 4])
                mesh_tensor = mesh_tensor.inverse_loop_pool(pooled_feats=y0)

            features = []

            # Convolution: see Equation(2) in the corresponding paper
            # 1. w_0 * e_i
            features.append(y0)
            # 2. w_1 * sigma_{e_j}
            features.append(conv_feats.sum(dim=-1))
            # 3. w_2 * sigma_{e_j+1 - e_j}
            features.append(jt.abs(conv_feats[..., [K-1] + list(range(K-1))] - conv_feats).sum(-1))
            # 4. w_3 * sigma_{e_i - e_j}
            features.append(jt.abs(y0.unsqueeze(dim=-1) - conv_feats).sum(dim=-1))
            
            y = jt.stack(features, dim=-1)
            y = self.conv2d(y)[:, :, :, 0]

        return mesh_tensor.updated(y)


class MeshPool(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def execute(self, mesh_tensor: MeshTensor):
        return mesh_tensor.inverse_loop_pool(self.op)


class MeshUnpool(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def execute(self, mesh_tensor, ref_mesh=None):
        if ref_mesh is None:
            return mesh_tensor.loop_unpool(self.mode)
        else:
            return mesh_tensor.loop_unpool(self.mode, ref_mesh.faces, ref_mesh._cache)


class MeshAdaptivePool(nn.Module):
    ''' Adaptive Pool (only support output size of (1,), i.e., global pooling)
    '''
    def __init__(self, op):
        super().__init__()

        if not op in ['max', 'mean']:
            raise Exception('Unsupported pooling method')

        self.op = op

    def execute(self, mesh_tensor: MeshTensor):
        jt_op = 'add' if self.op == 'mean' else 'maximum'

        y = mesh_tensor.feats.reindex_reduce(
            op=jt_op, 
            shape=[mesh_tensor.N, mesh_tensor.C],
            indexes=[
                'i0',
                'i1'
            ],
            overflow_conditions=[
                'i2 >= @e0(i0)'
            ],
            extras=[mesh_tensor.Fs])

        if self.op == 'mean':
            y = y / mesh_tensor.Fs.unsqueeze(dim=-1)

        return y


class MeshBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm(num_features, eps, momentum)

    def execute(self, mesh_tensor: MeshTensor):
        feats = self.bn(mesh_tensor.feats)
        return mesh_tensor.updated(feats)


class MeshReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def execute(self, mesh_tensor: MeshTensor):
        feats = self.relu(mesh_tensor.feats)
        return mesh_tensor.updated(feats)


class MeshDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def execute(self, mesh_tensor):
        feats = self.dropout(mesh_tensor.feats)
        return mesh_tensor.updated(feats)


class MeshLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def execute(self, mesh_tensor: MeshTensor):
        feats = self.conv1d(mesh_tensor.feats)
        return mesh_tensor.updated(feats)


def mesh_concat(meshes: List[MeshTensor]):
    new_feats = jt.concat([mesh.feats for mesh in meshes], dim=1)
    return meshes[0].updated(new_feats)


def mesh_add(mesh_a: MeshTensor, mesh_b: MeshTensor):
    new_feats = mesh_a.feats + mesh_b.feats
    return mesh_a.updated(new_feats)