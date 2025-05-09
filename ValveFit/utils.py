import numpy as np
import pyvista as pv


def export_quad_mesh_vtk(points, filename, tangent_vectors=None):
    n, m, _ = points.shape

    vertices = points.reshape(-1, 3)

    quads = []
    for i in range(n):
        for j in range(m - 1):
            v1 = i * m + j
            v2 = i * m + (j + 1) % m
            v3 = ((i + 1) * m + (j + 1) % m) % (n * m)
            v4 = ((i + 1) * m + j) % (n * m)
            quads.append([v1, v2, v3, v4])

    # Create a pyvista PolyData object from the vertices and quads
    faces = np.hstack([[4] + quad for quad in quads])
    mesh = pv.PolyData(vertices, faces)

    if tangent_vectors is not None:
        tangents_u = tangent_vectors[0].reshape(-1, 3)
        tangents_v = tangent_vectors[1].reshape(-1, 3)
        mesh["tangent_u"] = tangents_u
        mesh["tangent_v"] = tangents_v

    mesh.save(filename)


def save_pc_as_vtk(pc, filename, metadata=None, metadata_label=None):
    """Save a point cloud as a VTK file using PyVista."""
    points = np.asarray(pc)
    poly = pv.PolyData(points)
    if metadata is not None:
        if metadata_label is None:
            metadata_label = "metadata"
        poly[metadata_label] = np.asarray(metadata)
    poly.save(filename)


def save_bspline_smesh(filename, ctrl_pts, knotvectors, degrees, tag="NA"):
    CP_shape = ctrl_pts.shape[0:-1]
    ctrl_pts = np.array(ctrl_pts).reshape(-1, 3)
    filecontent = ["3"]
    filecontent.append(" ".join(map(str, degrees)))
    filecontent.append(f"{CP_shape[0]} {CP_shape[1]}")
    knots2list = [" ".join(map(str, np.array(kv).tolist())) for kv in knotvectors]
    filecontent += knots2list
    cp2list = []
    for row in ctrl_pts.tolist():
        row_str = [str(element) for element in row]
        line = " ".join(row_str)
        cp2list.append(line)

    filecontent += cp2list
    filecontent.append(f"1 {tag}")

    with open(filename, "w") as f:
        f.writelines([line + "\n" for line in filecontent])
