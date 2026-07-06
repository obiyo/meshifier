#!/usr/bin/env python3
"""
Test: CadQuery → STEP → Gmsh 0D/1D/2D physical groups → mesh verification.

Uses the same L-shaped extrusion as the main pipeline.  Tags:
  - one vertex  → physical point  "mass_pt"    (0D → point elements)
  - one edge    → physical curve  "stiffener"  (1D → bar elements)
  - one surface → physical surface "top_face"  (2D → quad elements)

Run:
    conda run -n meshcad python test_0d1d.py
"""

import numpy as np
import cadquery as cq
import gmsh
import meshio

STEP_FILE = "test_0d1d.step"
MESH_FILE = "test_0d1d.msh"

# ── 1. Build L-shape and export to STEP ───────────────────────────────────────
pts   = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
solid = cq.Workplane("XY").polyline(pts).close().extrude(5)
cq.exporters.export(solid, STEP_FILE)
print(f"Exported {STEP_FILE}")

# ── 2. Load in Gmsh and inspect topology ──────────────────────────────────────
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)
gmsh.model.add("test_0d1d")
gmsh.model.occ.importShapes(STEP_FILE)
gmsh.model.occ.synchronize()

vertices = gmsh.model.getEntities(dim=0)
edges    = gmsh.model.getEntities(dim=1)
surfaces = gmsh.model.getEntities(dim=2)
volumes  = gmsh.model.getEntities(dim=3)

print(f"\n── Topology ─────────────────────────────────────────────────────────────")
print(f"  Vertices (0D): {len(vertices)}")
print(f"  Edges    (1D): {len(edges)}")
print(f"  Surfaces (2D): {len(surfaces)}")
print(f"  Volumes  (3D): {len(volumes)}")

print(f"\n── Vertices ─────────────────────────────────────────────────────────────")
for _, tag in vertices:
    # getCenterOfMass(0, tag) returns (0,0,0) for all vertices in the OCC kernel.
    # getBoundingBox is the correct way: for a vertex, min == max == position.
    bb = gmsh.model.getBoundingBox(0, tag)
    x, y, z = bb[0], bb[1], bb[2]
    print(f"  v{tag:3d}:  ({x:6.2f}, {y:6.2f}, {z:6.2f})")

print(f"\n── Edges ────────────────────────────────────────────────────────────────")
for _, tag in edges:
    x, y, z  = gmsh.model.occ.getCenterOfMass(1, tag)
    length   = float(gmsh.model.occ.getMass(1, tag))
    print(f"  e{tag:3d}:  center=({x:6.2f}, {y:6.2f}, {z:6.2f})  length={length:.2f}")

# ── 3. Match Gmsh tags to target positions ────────────────────────────────────
def nearest_vertex(target_xyz):
    t = np.array(target_xyz)
    return min(
        [tag for _, tag in vertices],
        key=lambda tag: np.linalg.norm(
            np.array(gmsh.model.getBoundingBox(0, tag)[:3]) - t
        ),
    )

def nearest_edge(target_xyz):
    t = np.array(target_xyz)
    return min(
        [tag for _, tag in edges],
        key=lambda tag: np.linalg.norm(
            np.array(gmsh.model.occ.getCenterOfMass(1, tag)) - t
        ),
    )

def nearest_surface(target_xyz):
    t = np.array(target_xyz)
    return min(
        [tag for _, tag in surfaces],
        key=lambda tag: np.linalg.norm(
            np.array(gmsh.model.occ.getCenterOfMass(2, tag)) - t
        ),
    )

# Origin corner of the L-shape
vert_tag = nearest_vertex((0.0, 0.0, 0.0))
# Bottom edge of the horizontal base (midpoint ≈ (10, 0, 0))
edge_tag = nearest_edge((10.0, 0.0, 0.0))
# Top face of the extrusion (centroid ≈ (8.33, 8.33, 5.0))
surf_tag = nearest_surface((8.33, 8.33, 5.0))

print(f"\n── Matched tags ─────────────────────────────────────────────────────────")
print(f"  mass_pt    → vertex  {vert_tag}")
print(f"  stiffener  → edge    {edge_tag}")
print(f"  top_face   → surface {surf_tag}")

# ── 4. Physical groups ────────────────────────────────────────────────────────
gmsh.model.addPhysicalGroup(0, [vert_tag], name="mass_pt")
gmsh.model.addPhysicalGroup(1, [edge_tag], name="stiffener")
gmsh.model.addPhysicalGroup(2, [surf_tag], name="top_face")

# ── 5. Mesh ───────────────────────────────────────────────────────────────────
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.0)
gmsh.option.setNumber("Mesh.Algorithm",    8)
gmsh.option.setNumber("Mesh.RecombineAll", 0)
gmsh.model.mesh.setRecombine(2, surf_tag)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()
gmsh.write(MESH_FILE)

# ── 6. Verify element counts ──────────────────────────────────────────────────
print(f"\n── Mesh element counts ──────────────────────────────────────────────────")
ETYPE_PT   = 15  # 1-node point
ETYPE_BAR  = 1   # 2-node bar
ETYPE_TRI  = 2   # 3-node triangle
ETYPE_QUAD = 3   # 4-node quadrangle

for (dim, tag), name in [
    ((0, vert_tag), "mass_pt   "),
    ((1, edge_tag), "stiffener "),
    ((2, surf_tag), "top_face  "),
]:
    etypes, etags, _ = gmsh.model.mesh.getElements(dim, tag)
    counts = {et: len(et_) for et, et_ in zip(etypes, etags)}
    n_pt   = counts.get(ETYPE_PT,   0)
    n_bar  = counts.get(ETYPE_BAR,  0)
    n_tri  = counts.get(ETYPE_TRI,  0)
    n_quad = counts.get(ETYPE_QUAD, 0)
    print(f"  {name}  pt={n_pt}  bar={n_bar}  tri={n_tri}  quad={n_quad}")

gmsh.finalize()

# ── 7. Read back with meshio ──────────────────────────────────────────────────
print(f"\n── meshio readback ──────────────────────────────────────────────────────")
mesh = meshio.read(MESH_FILE)
print(f"  Cell block types : {[b.type for b in mesh.cells]}")
print(f"  Physical groups  : {list(mesh.field_data.keys())}")
print(f"  Nodes            : {len(mesh.points)}")
for block, ptags in zip(mesh.cells, mesh.cell_data.get("gmsh:physical", [])):
    if len(block.data):
        print(f"    {block.type:12s}  n={len(block.data):4d}  tags={set(ptags)}")

print("\nPASS" if any(b.type == "vertex"   for b in mesh.cells) and
                  any(b.type == "line"     for b in mesh.cells) and
                  any(b.type in ("quad","triangle") for b in mesh.cells)
     else "FAIL — missing expected cell types")
