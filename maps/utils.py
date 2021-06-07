from typing import List
import numpy as np

def check_duplicated(V) -> bool:
    np.unique(V, return_counts=True)
    for i in range(V.shape[0]):
        for j in range(i+1, V.shape[0]):
            if np.abs(V[i] - V[j]).sum() < 1e-7:
                return i, j
    return False

def maximal_independent_set(vids, faces, vertex_faces) -> List:
    mark = {}
    mis = []
    for v in vids:
        if not mark.get(v, False):
            mis.append(v)
            for fid in vertex_faces[v]:
                for u in faces[fid]:
                    mark[u] = True
    return mis
