"""
Functions for creating meshes with gmsh.
Originally by Joona Räty, built upon by Mikael Myyrä.
"""

from dataclasses import dataclass
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pydec


@dataclass
class ComplexAndMetadata:
    complex: pydec.SimplicialComplex
    edge_groups: dict[str, list[int]]


def rect_unstructured(
    mesh_width: float, mesh_height: float, tri_radius: float
) -> ComplexAndMetadata:
    """Create a rectangular triangle mesh with nonuniform triangle placement."""

    gmsh.initialize()
    gmsh.model.add("rec")

    gmsh.model.geo.addPoint(0, 0, 0, tri_radius, 1)
    gmsh.model.geo.addPoint(mesh_width, 0, 0, tri_radius, 2)
    gmsh.model.geo.addPoint(mesh_width, mesh_height, 0, tri_radius, 3)
    gmsh.model.geo.addPoint(0, mesh_height, 0, tri_radius, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

    gmsh.model.geo.addPlaneSurface([1], 1)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()
    # gmsh.model.mesh.optimize("Laplace2D")

    return _finalize_mesh_2d()


def square_with_hole(
    outer_extent: float, inner_extent: float, tri_radius: float
) -> ComplexAndMetadata:
    """Create a square mesh with nonuniform triangle placement
    and a hole in the middle."""

    gmsh.initialize()
    gmsh.model.add("rec")

    oe = outer_extent
    ie = inner_extent
    btm_left = gmsh.model.geo.addPoint(-oe, -oe, 0, tri_radius)
    btm_right = gmsh.model.geo.addPoint(oe, -oe, 0, tri_radius)
    top_right = gmsh.model.geo.addPoint(oe, oe, 0, tri_radius)
    top_left = gmsh.model.geo.addPoint(-oe, oe, 0, tri_radius)

    btm = gmsh.model.geo.addLine(btm_left, btm_right)
    right = gmsh.model.geo.addLine(btm_right, top_right)
    top = gmsh.model.geo.addLine(top_right, top_left)
    left = gmsh.model.geo.addLine(top_left, btm_left)

    outer_bounds = [btm, right, top, left]
    gmsh.model.geo.addCurveLoop(outer_bounds)

    btm_left = gmsh.model.geo.addPoint(-ie, -ie, 0, tri_radius)
    btm_right = gmsh.model.geo.addPoint(ie, -ie, 0, tri_radius)
    top_right = gmsh.model.geo.addPoint(ie, ie, 0, tri_radius)
    top_left = gmsh.model.geo.addPoint(-ie, ie, 0, tri_radius)

    btm = gmsh.model.geo.addLine(btm_left, btm_right)
    right = gmsh.model.geo.addLine(btm_right, top_right)
    top = gmsh.model.geo.addLine(top_right, top_left)
    left = gmsh.model.geo.addLine(top_left, btm_left)

    inner_bounds = [btm, right, top, left]
    gmsh.model.geo.addCurveLoop(inner_bounds)
    gmsh.model.geo.addPlaneSurface([1, 2], 1)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, outer_bounds, name="outer boundary")
    gmsh.model.addPhysicalGroup(1, inner_bounds, name="inner boundary")

    gmsh.model.mesh.generate(2)

    gmsh.model.mesh.optimize("Laplace2D")

    return _finalize_mesh_2d()


def rect_uniform(
    mesh_width: float, mesh_height: float, tri_radius: float
) -> pydec.SimplicialMesh:
    """Create a rectangular triangle mesh with uniform triangle size and alignment."""

    nx = int(np.ceil(mesh_width / tri_radius)) + 1
    dx = mesh_width / (nx - 1)
    ny = int(np.round(mesh_height / (0.866 * dx))) + 1
    dy = mesh_height / (ny - 1)
    evenr = np.linspace(0, mesh_width, nx)
    oddr = np.linspace(-dx / 2, mesh_width + dx / 2, nx + 1)
    oddr[0] += dx / 2
    oddr[-1] -= dx / 2
    V = np.empty((0, 2))
    for i in range(ny):
        if i % 2 == 0:
            row = np.vstack((evenr, i * dy * np.ones(nx))).T
        else:
            row = np.vstack((oddr, i * dy * np.ones(nx + 1))).T
        V = np.vstack((V, row))
    T = np.array([[0, nx, nx + 1], [0, nx + 1, 1]])
    RE = np.empty((0, 3))
    for i in range(nx):
        RE = np.vstack((RE, T + i))
    E = np.empty((0, 3))
    for i in range(ny - 1):
        if i % 2 == 0:
            row = RE[:-1] + int(i / 2) * (2 * nx + 1)
        else:
            row = RE[1:] + int((i - 1) / 2) * (2 * nx + 1) + nx
        E = np.vstack((E, row))
    return pydec.SimplicialMesh(V, E.astype("int32"))


def annulus(inner_r: float, outer_r: float, refine_count: int) -> ComplexAndMetadata:
    """Create a 2D unstructured mesh in the shape of an annulus,
    i.e. the area between two concentric circles, or a disk with a hole in the middle.

    Note: I can't figure out how to set a maximum element size for gmsh using circle
    curves, so taking a refinement level as parameter instead for now"""

    gmsh.initialize()
    gmsh.model.add("annulus")

    inner_circ = gmsh.model.occ.addCircle(0, 0, 0, inner_r)
    outer_circ = gmsh.model.occ.addCircle(0, 0, 0, outer_r)
    inner_loop = gmsh.model.occ.addCurveLoop([inner_circ])
    outer_loop = gmsh.model.occ.addCurveLoop([outer_circ])
    gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [outer_loop], name="outer boundary")
    gmsh.model.addPhysicalGroup(1, [inner_loop], name="inner boundary")

    gmsh.model.mesh.generate(dim=2)
    for _ in range(refine_count):
        gmsh.model.mesh.refine()

    gmsh.model.mesh.optimize("Laplace2D")

    return _finalize_mesh_2d()


def _finalize_mesh_2d() -> ComplexAndMetadata:
    """Get the active mesh from gmsh and transform it to a PyDEC 2D mesh
    along with possible edge group information for boundary identification."""

    nodes = gmsh.model.mesh.getNodes()
    # reshape into a list of (x, y, z) vectors
    vertices = np.reshape(nodes[1], (int(len(nodes[1]) / 3), 3))
    # delete the z coordinate since we're in 2D
    vertices = np.delete(vertices, 2, 1)

    edges = np.array(gmsh.model.mesh.getElements(2)[2])
    # reshape into groups of 3 per triangle,
    # subtract 1 because gmsh vertex tags start from 1
    edges = np.reshape(edges, (int(len(edges[0]) / 3), 3)) - 1

    # gather vertices identified with physical groups.
    # as far as I can tell we can only get a list of vertices
    # with no connectivity information from gmsh,
    # so we can't directly produce a list of edges.
    # thus we gather the vertices into a set we can compare edges to
    # once we've built the PyDEC complex
    vert_groups: dict[str, set[int]] = {}
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        name = gmsh.model.getPhysicalName(dim, tag)
        verts = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0]
        vert_groups[name] = set([int(vert) - 1 for vert in verts])

    gmsh.finalize()

    mesh = pydec.SimplicialMesh(vertices, edges.astype("int32"))
    complex = pydec.SimplicialComplex(mesh)

    # now that we have the complex we can iterate over edges
    # and see which edges have vertices belonging to groups
    edge_groups: dict[str, list[int]] = {name: [] for name in vert_groups.keys()}
    for edge_idx, edge in enumerate(complex[1].simplices):
        for name, group in vert_groups.items():
            if edge[0] in group and edge[1] in group:
                edge_groups[name].append(edge_idx)

    return ComplexAndMetadata(
        complex=complex,
        edge_groups=edge_groups,
    )


def _plot_test_cases():
    """Visualize some test cases. Called when this file is run as a standalone script."""

    # triangle mesh with a uniform axis-aligned structure

    plt.figure(figsize=(8, 8), dpi=80)
    mesh = rect_uniform(1, 1, 0.3)
    plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.indices)
    plt.show()

    # unstructured triangle mesh

    mesh = rect_unstructured(np.pi, np.pi, 0.3).complex.mesh
    plt.figure(figsize=(8, 8), dpi=80)
    plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.indices)
    plt.show()

    # more complex case with local refinement,
    # not directly using any of the generator functions

    gmsh.initialize()
    gmsh.model.add("rec")
    gmsh.model.occ.addRectangle(0, 0, 0, np.pi, np.pi)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()
    # gmsh.model.mesh.optimize("Laplace2D")

    b = gmsh.model.mesh.getNodes()[1]
    B = np.reshape(b, (int(len(b) / 3), 3))
    V = np.delete(B, 2, 1)

    c = np.array(gmsh.model.mesh.getElements(2)[2])
    E = np.reshape(c, (int(len(c[0]) / 3), 3)) - 1

    sc = pydec.SimplicialMesh(V, E)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.triplot(sc.vertices[:, 0], sc.vertices[:, 1], sc.indices)
    plt.show()
    # gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    _plot_test_cases()
