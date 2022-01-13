import json
import random
from pathlib import Path

from jittor.dataset import Dataset

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def augment_points(pts):
    # scale
    pts = pts * np.random.uniform(0.8, 1.25)

    # translation
    translation = np.random.uniform(-0.1, 0.1)
    pts = pts + translation

    return pts


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh.vertices = rotation.apply(mesh.vertices)
    return mesh


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh


def load_mesh(path, normalize=False, augments=[], request=[]):
    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])
    
    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)

    return mesh.faces, feats, Fs


def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    raw_labels = np.array(segment['raw_labels']) - 1
    sub_labels = np.array(segment['sub_labels']) - 1
    raw_to_sub = np.array(segment['raw_to_sub'])

    return raw_labels, sub_labels, raw_to_sub


class ClassificationDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augment=False, in_memory=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728*2)

        self.batch_size = batch_size
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs']

        self.mesh_paths = []
        self.labels = []
        self.browse_dataroot()

        self.set_attrs(total_len=len(self.mesh_paths))


    def browse_dataroot(self):
        # self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]
        self.shape_classes = sorted([x.name for x in self.dataroot.iterdir() if x.is_dir()])

        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                label = self.shape_classes.index(obj_class.name)
                for obj_path in (obj_class / self.mode).iterdir():
                    if obj_path.is_file():
                        self.mesh_paths.append(obj_path)
                        self.labels.append(label)

        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)

    def __getitem__(self, idx):
        faces, feats, Fs = load_mesh(self.mesh_paths[idx],
                                     normalize=True,
                                     augments=self.augments,
                                     request=self.feats)
        label = self.labels[idx]
        return faces, feats, Fs, label, self.mesh_paths[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, labels, mesh_paths = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)

        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
        
        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        labels = np.array(labels)

        return meshes, labels, mesh_paths


class SegmentationDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augments=None, in_memory=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728)
        self.batch_size = batch_size
        self.in_memory = in_memory
        self.dataroot = dataroot

        self.augments = []
        if train and augments:
            self.augments = augments

        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'center', 'normal']

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()

        self.set_attrs(total_len=len(self.mesh_paths))

    def browse_dataroot(self):
        for dataset in (Path(self.dataroot) / self.mode).iterdir():
            if not dataset.is_dir():
                continue
            for obj_path in dataset.iterdir():
                if obj_path.suffix == '.obj':
                    obj_name = obj_path.stem
                    seg_path = obj_path.parent / (obj_name + '.json')

                    raw_name = obj_name.rsplit('-', 1)[0]
                    raw_path = list(Path(self.dataroot).glob(f'raw/{raw_name}.*'))[0]
                    self.mesh_paths.append(str(obj_path))
                    self.raw_paths.append(str(raw_path))
                    self.seg_paths.append(str(seg_path))
        self.mesh_paths = np.array(self.mesh_paths)
        self.raw_paths = np.array(self.raw_paths)
        self.seg_paths = np.array(self.seg_paths)

    def __getitem__(self, idx):
        faces, feats, Fs = load_mesh(self.mesh_paths[idx], 
                                     normalize=True, 
                                     augments=self.augments,
                                     request=self.feats)
        raw_labels, sub_labels, raw_to_sub = load_segment(self.seg_paths[idx])

        return faces, feats, Fs, raw_labels, sub_labels, raw_to_sub, self.mesh_paths[idx], self.raw_paths[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, raw_labels, sub_labels, raw_to_sub, mesh_paths, raw_paths = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)
        np_sub_labels = np.ones((N, max_f), dtype=np.int32) * -1

        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
            np_sub_labels[i, :Fs[i]] = sub_labels[i]
        
        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        labels = np_sub_labels
        mesh_info = {
            'raw_labels': raw_labels,
            'raw_to_sub': raw_to_sub,
            'mesh_paths': mesh_paths,
            'raw_paths': raw_paths,
        } 
        return meshes, labels, mesh_info
