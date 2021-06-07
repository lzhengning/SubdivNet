import os
import json
from pathlib import Path

import numpy as np
import trimesh

import jittor as jt

from .mesh_tensor import MeshTensor


segment_colors = np.array([
    [0, 114, 189],
    [217, 83, 26],
    [238, 177, 32],
    [126, 47, 142],
    [117, 142, 48],
    [76, 190, 238],
    [162, 19, 48],
    [240, 166, 202],
])


def to_mesh_tensor(meshes):
    return MeshTensor(jt.int32(meshes['faces']), 
                      jt.float32(meshes['feats']), 
                      jt.int32(meshes['Fs']))


def save_results(mesh_infos, preds, labels, name):
    if not os.path.exists('results'):
        os.mkdir('results')

    if isinstance(labels, jt.Var):
        labels = labels.data
        
    results_path = Path('results') / name
    results_path.mkdir(parents=True, exist_ok=True)

    for i in range(preds.shape[0]):
        mesh_path = mesh_infos['mesh_paths'][i]
        mesh_name = Path(mesh_path).stem

        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.visual.face_colors[:, :3] = segment_colors[preds[i, :mesh.faces.shape[0]]]
        mesh.export(results_path / f'pred-{mesh_name}.ply')
        mesh.visual.face_colors[:, :3] = segment_colors[labels[i, :mesh.faces.shape[0]]]
        mesh.export(results_path / f'gt-{mesh_name}.ply')


def update_label_accuracy(preds, labels, acc):
    if isinstance(preds, jt.Var):
        preds = preds.data
    if isinstance(labels, jt.Var):
        labels = labels.data

    for i in range(preds.shape[0]):
        for k in range(len(acc)):
            if (labels[i] == k).sum() > 0:
                acc[k] += ((preds[i] == labels[i]) * (labels[i] == k)).sum() / (labels[i] == k).sum()


def compute_original_accuracy(mesh_infos, preds, labels):
    if isinstance(preds, jt.Var):
        preds = preds.data
    if isinstance(labels, jt.Var):
        labels = labels.data

    accs = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        raw_labels = mesh_infos['raw_labels'][i]
        raw_to_sub = mesh_infos['raw_to_sub'][i]
        accs[i] = np.mean((preds[i])[raw_to_sub] == raw_labels)

    return accs


class ClassificationMajorityVoting:
    def __init__(self, nclass):
        self.votes = {}
        self.nclass = nclass

    def vote(self, mesh_paths, preds, labels):
        if isinstance(preds, jt.Var):
            preds = preds.data
        if isinstance(labels, jt.Var):
            labels = labels.data

        for i in range(preds.shape[0]):
            name = (Path(mesh_paths[i]).stem).split('-')[0]
            if not name in self.votes:
                self.votes[name] = {
                    'polls': np.zeros(self.nclass, dtype=int),
                    'label': labels[i]
                }
            self.votes[name]['polls'][preds[i]] += 1

    def compute_accuracy(self):
        sum_acc = 0
        for name, vote in self.votes.items():
            pred = np.argmax(vote['polls'])
            sum_acc += pred == vote['label']
        return sum_acc / len(self.votes)


class SegmentationMajorityVoting:
    def __init__(self, nclass, name=''):
        self.votes = {}
        self.nclass = nclass
        self.name = name

    def vote(self, mesh_infos, preds, labels):
        if isinstance(preds, jt.Var):
            preds = preds.data
        if isinstance(labels, jt.Var):
            labels = labels.data

        for i in range(preds.shape[0]):
            name = (Path(mesh_infos['mesh_paths'][i]).stem)[:-4]
            nfaces = mesh_infos['raw_labels'][i].shape[0]
            if not name in self.votes:
                self.votes[name] = {
                    'polls': np.zeros((nfaces, self.nclass), dtype=int),
                    'label': mesh_infos['raw_labels'][i],
                    'raw_path': mesh_infos['raw_paths'][i],
                }
            polls = self.votes[name]['polls']
            raw_to_sub = mesh_infos['raw_to_sub'][i]
            raw_pred = (preds[i])[raw_to_sub]
            polls[np.arange(nfaces), raw_pred] += 1
    
    def compute_accuracy(self, save_results=False):
        if save_results:
            if self.name:
                results_path = Path('results') / self.name
            else:
                results_path = Path('results')
            results_path.mkdir(parents=True, exist_ok=True)

        sum_acc = 0
        all_acc = {}
        for name, vote in self.votes.items():
            label = vote['label']
            pred = np.argmax(vote['polls'], axis=1)
            acc = np.mean(pred == label)
            sum_acc += acc
            all_acc[name] = acc

            if save_results:
                mesh_path = vote['raw_path']
                mesh = trimesh.load_mesh(mesh_path, process=False)
                mesh.visual.face_colors[:, :3] = segment_colors[pred[:mesh.faces.shape[0]]]
                mesh.export(results_path / f'pred-{name}.ply')
                mesh.visual.face_colors[:, :3] = segment_colors[label[:mesh.faces.shape[0]]]
                mesh.export(results_path / f'gt-{name}.ply')
        
        if save_results:
            with open(results_path / 'acc.json', 'w') as f:
                json.dump(all_acc, f, indent=4)
        return sum_acc / len(self.votes)
