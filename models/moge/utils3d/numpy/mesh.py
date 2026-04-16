import numpy as np
from typing import *
from ._helpers import batched


__all__ = [
    'triangulate',
    'compute_face_normal',
    'compute_face_angle',
    'compute_vertex_normal',
    'compute_vertex_normal_weighted',
    'remove_corrupted_faces',
    'merge_duplicate_vertices',
    'remove_unreferenced_vertices',
    'subdivide_mesh_simple',
    'mesh_relations',
    'flatten_mesh_indices'
]


def triangulate(
    faces: np.ndarray,
    vertices: np.ndarray = None,
    backslash: np.ndarray = None
) -> np.ndarray:
    """
    Triangulate a polygonal mesh.

    Args:
        faces (np.ndarray): [L, P] polygonal faces
        vertices (np.ndarray, optional): [N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        backslash (np.ndarray, optional): [L] boolean array indicating
            how to triangulate the quad faces. Defaults to None.

    Returns:
        (np.ndarray): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if vertices is not None:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if backslash is None:
            backslash = np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=-1) < \
                        np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], axis=-1)
    if backslash is None:
        loop_indice = np.stack([
            np.zeros(P - 2, dtype=int),
            np.arange(1, P - 1, 1, dtype=int),
            np.arange(2, P, 1, dtype=int)
        ], axis=1)
        return faces[:, loop_indice].reshape((-1, 3))
    else:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        faces = np.where(
            backslash[:, None],
            faces[:, [0, 1, 2, 0, 2, 3]],
            faces[:, [0, 1, 3, 3, 1, 2]]
        ).reshape((-1, 3))
        return faces


@batched(2, None)
def compute_face_normal(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute face normals of a triangular mesh

    Args:
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices

    Returns:
        normals (np.ndarray): [..., T, 3] face normals
    """
    normal = np.cross(
        vertices[..., faces[:, 1], :] - vertices[..., faces[:, 0], :],
        vertices[..., faces[:, 2], :] - vertices[..., faces[:, 0], :]
    )
    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm == 0] = 1
    normal /= normal_norm
    return normal


@batched(2, None)
def compute_face_angle(
        vertices: np.ndarray,
        faces: np.ndarray,
        eps: float = 1e-12
    ) -> np.ndarray:
    """
    Compute face angles of a triangular mesh

    Args:
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices

    Returns:
        angles (np.ndarray): [..., T, 3] face angles
    """
    face_angle = np.zeros_like(faces, dtype=vertices.dtype)
    for i in range(3):
        edge1 = vertices[..., faces[:, (i + 1) % 3], :] - vertices[..., faces[:, i], :]
        edge2 = vertices[..., faces[:, (i + 2) % 3], :] - vertices[..., faces[:, i], :]
        face_angle[..., i] = np.arccos(np.sum(
            edge1 / np.clip(np.linalg.norm(edge1, axis=-1, keepdims=True), eps, None) *
            edge2 / np.clip(np.linalg.norm(edge2, axis=-1, keepdims=True), eps, None),
            axis=-1
        ))
    return face_angle


@batched(2, None, 2)
def compute_vertex_normal(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normal: np.ndarray = None
) -> np.ndarray:
    """
    Compute vertex normals of a triangular mesh by averaging neightboring face normals
    TODO: can be improved.
    
    Args:
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
        face_normal (np.ndarray, optional): [..., T, 3] face normals.
            None to compute face normals from vertices and faces. Defaults to None.

    Returns:
        normals (np.ndarray): [..., N, 3] vertex normals
    """
    if face_normal is None:
        face_normal = compute_face_normal(vertices, faces)
    vertex_normal = np.zeros_like(vertices, dtype=vertices.dtype)
    for n in range(vertices.shape[0]):
        for i in range(3):
            vertex_normal[n, :, 0] += np.bincount(faces[:, i], weights=face_normal[n, :, 0], minlength=vertices.shape[1])
            vertex_normal[n, :, 1] += np.bincount(faces[:, i], weights=face_normal[n, :, 1], minlength=vertices.shape[1])
            vertex_normal[n, :, 2] += np.bincount(faces[:, i], weights=face_normal[n, :, 2], minlength=vertices.shape[1])
    vertex_normal_norm = np.linalg.norm(vertex_normal, axis=-1, keepdims=True)
    vertex_normal_norm[vertex_normal_norm == 0] = 1
    vertex_normal /= vertex_normal_norm
    return vertex_normal


@batched(2, None, 2)
def compute_vertex_normal_weighted(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normal: np.ndarray = None
) -> np.ndarray:
    """
    Compute vertex normals of a triangular mesh by weighted sum of neightboring face normals
    according to the angles

    Args:
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [..., T, 3] triangular face indices
        face_normal (np.ndarray, optional): [..., T, 3] face normals.
            None to compute face normals from vertices and faces. Defaults to None.

    Returns:
        normals (np.ndarray): [..., N, 3] vertex normals
    """
    if face_normal is None:
        face_normal = compute_face_normal(vertices, faces)
    face_angle = compute_face_angle(vertices, faces)
    vertex_normal = np.zeros_like(vertices)
    for n in range(vertices.shape[0]):
        for i in range(3):
            vertex_normal[n, :, 0] += np.bincount(faces[n, :, i], weights=face_normal[n, :, 0] * face_angle[n, :, i], minlength=vertices.shape[1])
            vertex_normal[n, :, 1] += np.bincount(faces[n, :, i], weights=face_normal[n, :, 1] * face_angle[n, :, i], minlength=vertices.shape[1])
            vertex_normal[n, :, 2] += np.bincount(faces[n, :, i], weights=face_normal[n, :, 2] * face_angle[n, :, i], minlength=vertices.shape[1])
    vertex_normal_norm = np.linalg.norm(vertex_normal, axis=-1, keepdims=True)
    vertex_normal_norm[vertex_normal_norm == 0] = 1
    vertex_normal /= vertex_normal_norm
    return vertex_normal
    

def remove_corrupted_faces(
        faces: np.ndarray
    ) -> np.ndarray:
    """
    Remove corrupted faces (faces with duplicated vertices)

    Args:
        faces (np.ndarray): [T, 3] triangular face indices

    Returns:
        np.ndarray: [T_, 3] triangular face indices
    """
    corrupted = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return faces[~corrupted]


def merge_duplicate_vertices(
        vertices: np.ndarray, 
        faces: np.ndarray,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    Args:
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    Returns:
        vertices (np.ndarray): [N_, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
    """
    vertices_round = np.round(vertices / tol)
    _, uni_i, uni_inv = np.unique(vertices_round, return_index=True, return_inverse=True, axis=0)
    vertices = vertices[uni_i]
    faces = uni_inv[faces]
    return vertices, faces


def remove_unreferenced_vertices(
    faces: np.ndarray,
    *vertice_attrs,
    return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Remove unreferenced vertices of a mesh. 
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    Args:
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes

    Returns:
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (np.ndarray, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = np.unique(faces, return_inverse=True)
    faces = inv_map.astype(np.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def subdivide_mesh_simple(
    vertices: np.ndarray,
    faces: np.ndarray, 
    n: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    Args:
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
        n (int, optional): number of subdivisions. Defaults to 1.

    Returns:
        vertices (np.ndarray): [N_, 3] subdivided 3-dimensional vertices
        faces (np.ndarray): [4 * T, 3] subdivided triangular face indices
    """
    for _ in range(n):
        edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=2)
        uni_edges, uni_inv = np.unique(edges.reshape(-1, 2), return_inverse=True, axis=0)
        uni_inv = uni_inv.reshape(3, -1)
        midpoints = (vertices[uni_edges[:, 0]] + vertices[uni_edges[:, 1]]) / 2

        n_vertices = vertices.shape[0]
        vertices = np.concatenate([vertices, midpoints], axis=0)
        faces = np.concatenate([
            np.stack([faces[:, 0], n_vertices + uni_inv[0], n_vertices + uni_inv[2]], axis=1),
            np.stack([faces[:, 1], n_vertices + uni_inv[1], n_vertices + uni_inv[0]], axis=1),
            np.stack([faces[:, 2], n_vertices + uni_inv[2], n_vertices + uni_inv[1]], axis=1),
            np.stack([n_vertices + uni_inv[0], n_vertices + uni_inv[1], n_vertices + uni_inv[2]], axis=1),
        ], axis=0)
    return vertices, faces


def mesh_relations(
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the relation between vertices and faces.
    NOTE: The input mesh must be a manifold triangle mesh.

    Args:
        faces (np.ndarray): [T, 3] triangular face indices

    Returns:
        edges (np.ndarray): [E, 2] edge indices
        edge2face (np.ndarray): [E, 2] edge to face relation. The second column is -1 if the edge is boundary.
        face2edge (np.ndarray): [T, 3] face to edge relation
        face2face (np.ndarray): [T, 3] face to face relation
    """
    T = faces.shape[0]
    edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1).reshape(-1, 2)  # [3T, 2]
    edges = np.sort(edges, axis=1)  # [3T, 2]
    edges, face2edge, occurence = np.unique(edges, axis=0, return_inverse=True, return_counts=True) # [E, 2], [3T], [E]
    E = edges.shape[0]
    assert np.all(occurence <= 2), "The input mesh is not a manifold mesh."

    # Edge to face relation
    padding = np.arange(E, dtype=np.int32)[occurence == 1]
    padded_face2edge = np.concatenate([face2edge, padding], axis=0)  # [2E]
    edge2face = np.argsort(padded_face2edge, kind='stable').reshape(-1, 2) // 3  # [E, 2]
    edge2face_valid = edge2face[:, 1] < T   # [E]
    edge2face[~edge2face_valid, 1] = -1

    # Face to edge relation
    face2edge = face2edge.reshape(-1, 3)  # [T, 3]

    # Face to face relation
    face2face = edge2face[face2edge]  # [T, 3, 2]
    face2face = face2face[face2face != np.arange(T)[:, None, None]].reshape(T, 3)  # [T, 3]
    
    return edges, edge2face, face2edge, face2face


@overload
def flatten_mesh_indices(faces1: np.ndarray, attr1: np.ndarray, *other_faces_attrs_pairs: np.ndarray) -> Tuple[np.ndarray, ...]: 
    """
    Rearrange the indices of a mesh to a flattened version. Vertices will be no longer shared.

    ### Parameters:
    - `faces1`: [T, P] face indices of the first attribute
    - `attr1`: [N1, ...] attributes of the first mesh
    - ...

    ### Returns:
    - `faces`: [T, P] flattened face indices, contigous from 0 to T * P - 1
    - `attr1`: [T * P, ...] attributes of the first mesh, where every P values correspond to a face
    _ ...
    """
def flatten_mesh_indices(*args: np.ndarray) -> Tuple[np.ndarray, ...]:
    assert len(args) % 2 == 0, "The number of arguments must be even."
    T, P = args[0].shape
    assert all(arg.shape[0] == T and arg.shape[1] == P for arg in args[::2]), "The faces must have the same shape."
    attr_flat = []
    for faces_, attr_ in zip(args[::2], args[1::2]):
        attr_flat_ = attr_[faces_].reshape(-1, *attr_.shape[1:])
        attr_flat.append(attr_flat_)
    faces_flat = np.arange(T * P, dtype=np.int32).reshape(T, P)
    return faces_flat, *attr_flat