from maps.utils import maximal_independent_set
import random
from collections import defaultdict
from time import time
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from sortedcollections import ValueSortedDict
from shapely.geometry import LineString
from shapely.geometry import Point
from tqdm import tqdm
from trimesh import PointCloud
from trimesh import Trimesh

from .geometry import face_areas, min_triangle_angles
from .geometry import plane_from_points
from .geometry import to_barycentric, from_barycenteric
from .geometry import CDT, MVT
from .geometry import one_ring_neighbor_uv


class Mesh:
    def __init__(self, vertices, faces):
        mesh = Trimesh(vertices, faces, process=False, maintain_order=True)
        self.verts = vertices.copy()
        self.faces = faces.copy()
        self.vertex_faces = [set(F) for F in mesh.vertex_faces]
        for fs in self.vertex_faces:
            if -1 in fs:
                fs.remove(-1)

        self.V = self.verts.shape[0]
        self.F = self.faces.shape[0]
        self.vmask = np.ones(self.V, dtype=bool)
        self.fmask = np.ones(self.F, dtype=bool)

    def neighbors(self, i: int) -> Set[int]:
        N = set()
        for f in self.vertex_faces[i]:
            N.add(self.faces[f, 0])
            N.add(self.faces[f, 1])
            N.add(self.faces[f, 2])
        N.remove(i)
        return N

    def one_ring_neighbors(self, i: int) -> List[int]:
        G = nx.Graph()
        for f in self.vertex_faces[i]:
            G.add_edge(self.faces[f, 0], self.faces[f, 1])
            G.add_edge(self.faces[f, 1], self.faces[f, 2])
            G.add_edge(self.faces[f, 2], self.faces[f, 0])
        cycle = nx.cycle_basis(G.subgraph(G[i]))[0]

        u, v = cycle[0], cycle[1]
        for f in self.vertex_faces[i]:
            if u in self.faces[f] and v in self.faces[f]:
                clockwise = (
                    (u == self.faces[f, 0] and v == self.faces[f, 1])
                    or (u == self.faces[f, 1] and v == self.faces[f, 2])
                    or (u == self.faces[f, 2] and v == self.faces[f, 0])
                )
                if not clockwise:
                    cycle = cycle[::-1]
                break
        else:
            raise Exception("Impossible")
        return cycle

    def add_vertex(self, vertex):
        if self.V + 1 > self.verts.shape[0]:
            self.vmask = np.append(self.vmask, np.zeros_like(self.vmask))
            self.verts = np.append(self.verts, np.zeros_like(self.verts), axis=0)
        self.verts[self.V] = vertex
        self.vmask[self.V] = True
        self.vertex_faces.append(set())
        self.V += 1

    def remove_face(self, fid):
        self.fmask[fid] = False
        for v in self.faces[fid]:
            self.vertex_faces[v].remove(fid)

    def remove_faces(self, fids):
        self.fmask[fids] = False
        for fid in fids:
            for v in self.faces[fid]:
                self.vertex_faces[v].remove(fid)

    def add_faces(self, new_faces):
        for v0, v1, v2 in new_faces:
            assert v0 != v1 and v1 != v2 and v2 != v0

        if self.F + len(new_faces) > self.faces.shape[0]:
            self.fmask = np.append(self.fmask, np.zeros_like(self.fmask))
            self.faces = np.append(self.faces, np.zeros_like(self.faces), axis=0)
        self.faces[self.F : self.F + len(new_faces)] = new_faces
        self.fmask[self.F : self.F + len(new_faces)] = True

        for fid, face in enumerate(new_faces):
            for v in face:
                self.vertex_faces[v].add(fid + self.F)

        self.F += len(new_faces)

    def remove_vertex(self, i, new_faces, neighbors=None):
        old_faces = self.vertex_faces[i]

        # delete the vertex and related faces
        self.remove_faces(list(self.vertex_faces[i]))
        self.add_faces(new_faces)

        # update vertex_faces
        for k in neighbors:
            for f in old_faces:
                if f in self.vertex_faces[k]:
                    self.vertex_faces[k].remove(f)
            for j, face in enumerate(new_faces):
                if k in face:
                    self.vertex_faces[k].add(j + self.F - len(new_faces))


class BaseMesh(Mesh):
    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)

        self.face_distortion = {i:1 for i in range(self.F)}

    def assign_initial_vertex_weights(self):
        Q = {}
        for i in range(self.V):
            Q[i] = np.zeros([4, 4])
            for fid in self.vertex_faces[i]:
                plane = plane_from_points(self.verts[self.faces[fid]])
                plane = plane.reshape([1, 4])
                Q[i] += plane.T @ plane

        vertex_weights = ValueSortedDict()
        for i in range(self.V):
            vertex_weights[i] = self.compute_vertex_weights(i, Q)
        return vertex_weights, Q

    def compute_vertex_weights(self, i, Q: Dict):
        weight = 0
        coord = np.ones([1, 4])
        for v in self.neighbors(i):
            coord[0, :3] = self.verts[v]
            weight += coord @ Q[i] @ coord.T
        return weight

    def is_validate_removal(self, i: int, neighbors, new_faces, new_edges):
        if not self.is_manifold(new_edges, neighbors):
            return False
        return True

    def is_manifold(self, new_edges, neighbors):
        """check if there is a triangle connected by the neighbhors of i"""
        for v in neighbors:
            for f in self.vertex_faces[v]:
                for a, b in new_edges:
                    if a in self.faces[f] and b in self.faces[f]:
                        return False
        return True


class ParamMesh(Mesh):
    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)

        self.xyz = vertices.copy()
        self.baries = defaultdict(dict)
        self.on_edge = defaultdict(dict)

    def add_xyz(self, points_uv, uv, edge):
        if self.xyz.shape != self.verts.shape:
            self.xyz = np.append(self.xyz, np.zeros_like(self.xyz), axis=0)

        a = np.array(points_uv[edge[1]]) - points_uv[edge[0]]
        b = np.array(uv) - points_uv[edge[0]]
        t = np.linalg.norm(b) / np.linalg.norm(a)
        self.xyz[self.V - 1] = t * self.xyz[edge[1]] + (1 - t) * self.xyz[edge[0]]

    def is_watertight(self) -> bool:
        verts = self.verts[self.vmask]
        faces = self.faces[self.fmask]
        mesh = Trimesh(verts, faces, process=False)
        return mesh.is_watertight and mesh.is_winding_consistent

    def split_triangles_on_segments(self, points_uv: Dict, points_on_ring: Dict, lines):
        for line in lines:
            edges = defaultdict(set)
            for v in points_uv:
                if not v in points_on_ring:
                    for fid in self.vertex_faces[v]:
                        if all([u in points_uv for u in self.faces[fid]]):
                            v0, v1, v2 = self.faces[fid]
                            edges[tuple(sorted([v0, v1]))].add(fid)
                            edges[tuple(sorted([v1, v2]))].add(fid)
                            edges[tuple(sorted([v2, v0]))].add(fid)

            intersections = defaultdict(list)
            for edge, fs in edges.items():
                ret = self.intersect(points_uv, edge, line)
                if ret is not None:
                    if isinstance(ret[0], tuple):
                        self.add_vertex([0, 0, 0])
                        self.add_xyz(points_uv, ret[0], edge)
                        ret[1] = self.V - 1
                        points_uv[self.V - 1] = ret[0]
                        self.on_edge[self.V - 1] = line
                    ret += (edge,)
                    for fid in fs:
                        intersections[fid].append(ret)

            for fid, its in intersections.items():
                if len(its) == 3:
                    u0 = [x[1] for x in its if x[0] is None]
                    u = [x[1] for x in its if x[0] is not None]
                    assert len(u0) == 2, "len(u0) != 2"
                    assert u0[0] == u0[1], "u0[0] != u0[1]"
                    self.split_into_two_triangle(fid, u0[0], u[0])
                elif len(its) == 2:
                    if its[0][0] is not None and its[1][0] is not None:
                        self.split_into_tri_trap(fid, its[0][1:], its[1][1:], points_uv)
                    elif (its[0][0] is None) ^ (its[1][0] is None):
                        raise Exception("[Impossible]")
                elif len(its) == 1:
                    raise Exception("[Impossible]")

    def split_into_two_triangle(self, fid, u0, u):
        v0, v1, v2 = self.faces[fid]
        self.remove_face(fid)
        if v1 == u0:
            v0, v1, v2 = v1, v2, v0
        elif v2 == u0:
            v0, v1, v2 = v2, v0, v1
        assert v0 == u0, "v0 != u0"
        self.add_faces([[v0, v1, u], [v0, u, v2]])

    def split_into_tri_trap(self, fid, edge0, edge1, points_uv):
        v0, v1, v2 = self.faces[fid]
        self.remove_face(fid)
        u0, (e0v0, e0v1) = edge0
        u1, (e1v0, e1v1) = edge1

        def on_edge(x0, x1):
            if sorted([x0, x1]) == sorted([e0v0, e0v1]):
                return u0
            if sorted([x0, x1]) == sorted([e1v0, e1v1]):
                return u1
            return None

        e0 = on_edge(v0, v1)
        e1 = on_edge(v1, v2)
        e2 = on_edge(v2, v0)

        #       v0
        #      /  \
        #    e0    e2
        #    /      \
        #   v1 -e1- v2
        #

        if e0 is None:
            choice_a = [
                [v2, e2, e1],
                [v0, v1, e1],
                [v0, e1, e2],
            ]
            choice_b = [
                [v2, e2, e1],
                [v1, e1, e2],
                [v1, e2, v0],
            ]
        elif e1 is None:
            choice_a = [
                [v0, e0, e2],
                [v1, e2, e0],
                [v1, v2, e2],
            ]
            choice_b = [
                [v0, e0, e2],
                [v2, e2, e0],
                [v2, e0, v1],
            ]
        elif e2 is None:
            choice_a = [
                [v1, e1, e0],
                [v2, e0, e1],
                [v2, v0, e0],
            ]
            choice_b = [
                [v1, e1, e0],
                [v0, e1, v2],
                [v0, e0, e1],
            ]
        else:
            raise Exception("[Impossible]")

        def make_triangle(vids):
            return np.array(
                [
                    points_uv[vids[0]],
                    points_uv[vids[1]],
                    points_uv[vids[2]],
                ]
            )

        min_a = min(
            [
                min_triangle_angles(make_triangle(choice_a[1])),
                min_triangle_angles(make_triangle(choice_a[2])),
            ]
        )
        min_b = min(
            [
                min_triangle_angles(make_triangle(choice_b[1])),
                min_triangle_angles(make_triangle(choice_b[2])),
            ]
        )

        if min_a > min_b:
            return self.add_faces(choice_a)
        else:
            return self.add_faces(choice_b)

    def intersect(self, points_uv, edge, line):
        line = sorted(line)
        edge = sorted(edge)

        if line == edge:
            return None
        if line[0] in edge:
            return [None, line[0]]
        if line[1] in edge:
            return [None, line[1]]

        line = LineString([points_uv[line[0]], points_uv[line[1]]])
        edge = LineString([points_uv[edge[0]], points_uv[edge[1]]])

        ret = edge.intersection(line)
        if isinstance(ret, Point):
            return [(ret.x, ret.y), -1]
        else:
            return None


class MAPS:
    def __init__(self, vertices, faces, base_size, timeout=None, verbose=False):
        self.mesh = Trimesh(vertices, faces, process=False, maintain_order=True)

        self.base = BaseMesh(vertices, faces)
        self.param = ParamMesh(vertices, faces)

        self.base_size = base_size
        self.verbose = verbose
        self.timeout = timeout

        self.param_tri_verts = defaultdict(list)

        self.decimate()
        self.base_size = self.base.fmask.sum()

    def decimate(self):
        start_time = time()
        vertex_weights = ValueSortedDict({i: 0 for i in range(self.base.V)})

        with tqdm(total=self.base.F - self.base_size, disable=not self.verbose) as pbar:
            while self.base.fmask.sum() > self.base_size:
                vw = list(vertex_weights.keys())
                copy = vw[: len(vw) // 4]
                random.shuffle(copy)
                vw[: len(vw) // 4] = copy
                mis = maximal_independent_set(
                    vw, self.base.faces, self.base.vertex_faces
                )
                for i in mis:
                    if self.timeout is not None and time() - start_time > self.timeout:
                        return
                    neighbors = self.base.one_ring_neighbors(i)
                    if self.try_decimate_base_vertex(i):
                        self.base.vmask[i] = 0
                        vertex_weights.pop(i)

                        for k in neighbors:
                            total = 0
                            for fid in self.base.vertex_faces[k]:
                                total += len(self.param_tri_verts[fid])
                            vertex_weights[k] = total

                        pbar.update(2)
                        if self.base.fmask.sum() <= self.base_size:
                            return

    def compute_vertex_weight(self, i: int):
        neighbors = self.base.one_ring_neighbors(i)
        neighbors_uv = one_ring_neighbor_uv(neighbors, self.base.verts, i)

        # Try constrained denauly triangulate
        new_faces, new_edges = CDT(neighbors, neighbors_uv)

        # Check mesh
        if not self.base.is_validate_removal(i, neighbors, new_faces, new_edges):
            neighbors_uv = np.array([uv / np.linalg.norm(uv) for uv in neighbors_uv])
            for v in neighbors:
                new_faces, new_edges = MVT(v, neighbors)
            else:
                return 0

        old_faces = list(self.base.vertex_faces[i])
        old_areas = face_areas(self.base.verts, self.base.faces[old_faces]).sum()
        new_areas = face_areas(self.base.verts, new_faces).sum()

        fd = min(self.base.face_distortion[fid] for fid in old_faces)
        return fd * new_areas / old_areas
    

    def try_decimate_base_vertex(self, i: int) -> bool:
        neighbors = self.base.one_ring_neighbors(i)
        neighbors_uv = one_ring_neighbor_uv(neighbors, self.base.verts, i)

        # Try constrained denauly triangulate
        new_faces, new_edges = CDT(neighbors, neighbors_uv)

        # Check mesh
        if not self.base.is_validate_removal(i, neighbors, new_faces, new_edges):
            neighbors_uv = np.array([uv / np.linalg.norm(uv) for uv in neighbors_uv])
            for v in neighbors:
                new_faces, new_edges = MVT(v, neighbors)
                if self.base.is_validate_removal(i, neighbors, new_faces, new_edges):
                    break
            else:
                return False

        ring_uv = {n: neighbors_uv[k] for k, n in enumerate(neighbors)}
        ring_uv[i] = [0, 0]

        self.reparameterize(i, ring_uv, new_faces, new_edges)
        self.base.remove_vertex(i, new_faces, neighbors)

        return True

    def reparameterize(self, i: int, ring_uv: Dict, new_faces, new_edges):
        points_uv = ring_uv.copy()
        neighbors = set([k for k in ring_uv.keys() if k != i])
        points_on_ring = neighbors.copy()

        for fid in self.base.vertex_faces[i]:
            face = self.base.faces[fid]
            face_uv = [ring_uv[face[0]], ring_uv[face[1]], ring_uv[face[2]]]
            for v in self.param_tri_verts[fid]:
                if not v in points_uv:
                    points_uv[v] = from_barycenteric(face_uv, self.param.baries[v][fid])
                    if v in self.param.on_edge:
                        edge = self.param.on_edge[v]
                        if edge[0] in neighbors and edge[1] in neighbors:
                            points_on_ring.add(v)
                del self.param.baries[v][fid]

        self.param.split_triangles_on_segments(points_uv, points_on_ring, new_edges)

        for v, uv in points_uv.items():
            self.uv_to_xyz_tri(v, uv, ring_uv, new_faces)

        return True

    def uv_to_xyz_tri(self, v: int, uv, verts_uv: Dict, faces: List):
        def in_triangle(point, triangle):
            max_s = np.abs(triangle).max()
            point = point / max_s
            triangle = triangle / max_s
            n1 = np.cross(point - triangle[0], triangle[1] - triangle[0])
            n2 = np.cross(point - triangle[1], triangle[2] - triangle[1])
            n3 = np.cross(point - triangle[2], triangle[0] - triangle[2])
            n1 = 0 if abs(n1) < 1e-10 else n1
            n2 = 0 if abs(n2) < 1e-10 else n2
            n3 = 0 if abs(n3) < 1e-10 else n3
            return ((n1 >= 0) and (n2 >= 0) and (n3 >= 0)) or (
                (n1 <= 0) and (n2 <= 0) and (n3 <= 0)
            )

        found = False
        for f, face in enumerate(faces):
            triangle_uv = [
                verts_uv[face[0]],
                verts_uv[face[1]],
                verts_uv[face[2]],
            ]
            if in_triangle(uv, triangle_uv):
                point_bary = to_barycentric(uv, triangle_uv)
                assert np.abs(point_bary).sum() <= 2
                point_xyz = from_barycenteric(self.base.verts[face], point_bary)
                tri = f + self.base.F
                self.param_tri_verts[tri].append(v)
                self.param.baries[v][tri] = point_bary
                self.param.verts[v] = point_xyz
                found = True

        assert found

    def mesh_upsampling(self, depth) -> Trimesh:
        sub_verts, sub_faces = self.subdivide(depth)

        sub_verts = self.parameterize(sub_verts)

        return Trimesh(sub_verts, sub_faces, process=False, maintain_order=True)

    def subdivide(self, depth):
        verts = self.base.verts[self.base.vmask]
        vmaps = np.cumsum(self.base.vmask) - 1
        faces = self.base.faces[self.base.fmask]
        faces = np.vectorize(lambda f: vmaps[f])(faces)

        for _ in range(depth):
            nV = verts.shape[0]
            nF = faces.shape[0]
            edges_d = np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
            )
            edges_d = np.sort(edges_d, axis=1)
            edges_u, F2E = np.unique(edges_d, axis=0, return_inverse=True)
            new_verts = (verts[edges_u[:, 0]] + verts[edges_u[:, 1]]) / 2
            verts = np.concatenate([verts, new_verts], axis=0)

            E2 = F2E[:nF] + nV
            E0 = F2E[nF : nF * 2] + nV
            E1 = F2E[nF * 2 :] + nV
            faces = np.concatenate(
                [
                    np.stack([faces[:, 0], E2, E1], axis=-1),
                    np.stack([faces[:, 1], E0, E2], axis=-1),
                    np.stack([faces[:, 2], E1, E0], axis=-1),
                    np.stack([E0, E1, E2], axis=-1),
                ],
                axis=0,
            )

        return verts, faces

    def parameterize(self, points):
        param_verts = self.param.verts[: self.param.V]
        param_faces = self.param.faces[self.param.fmask]
        param_mesh = Trimesh(param_verts, param_faces, process=False)

        closest_points, _, triangle_id = param_mesh.nearest.on_surface(points)

        for i, point in enumerate(closest_points):
            face = param_faces[triangle_id[i]]
            triangle = self.param.verts[face]
            xyz = self.param.xyz[face]
            try:
                bary = to_barycentric(point, triangle)
                points[i] = from_barycenteric(xyz, bary)
            except np.linalg.LinAlgError as e:
                points[i] = xyz.mean(axis=0)

        # points, _, _ = self.mesh.nearest.on_surface(points)

        return points

    def save_decimation(self, v):
        faces = [self.base.faces[i] for i, m in enumerate(self.base.fmask) if m]
        mesh = Trimesh(self.base.verts, faces)
        mesh.export("dec.obj")

        verts = [self.param.verts[i] for i, m in enumerate(self.param.vmask) if m]
        pc = PointCloud(verts)
        pc.visual.vertex_colors = np.array([[102, 102, 102, 255]] * self.param.V)
        pc.visual.vertex_colors[v] = [255, 0, 0, 255]
        pc.export("param.ply")

        faces = [self.param.faces[i] for i, m in enumerate(self.param.fmask) if m]
        mesh = Trimesh(verts, faces)
        mesh.export("param.obj")

        verts = self.param.xyz[self.param.vmask]
        mesh = Trimesh(verts, faces)
        mesh.export("rec.obj")
