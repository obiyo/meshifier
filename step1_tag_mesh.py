#!/usr/bin/env python3
"""
Step 1: CadQuery L-extrusion → STEP → Gmsh physical groups → quad mesh.

Can be used in two modes:

  1. Standalone demo (hardcoded L-shape with 3 centroid-matched faces):
       conda run -n meshcad python step1_tag_mesh.py

  2. Called from face_labeler.py with a JSON label map:
       from step1_tag_mesh import run_mesh
       run_mesh("part.step", face_labels_dict, "part.msh")

Geometry (XY plane, extruded 5 mm in Z):

  y=20 ┌─────┐
       │     │   x∈[0,10], y∈[10,20]  (vertical arm)
  y=10 │     └──────┐
       │            │   x∈[0,20], y∈[0,10]  (horizontal base)
  y=0  └────────────┘
       x=0         x=20
"""

import sys
from pathlib import Path

import numpy as np
import gmsh

ETYPE_TRI  = 2   # 3-node triangle
ETYPE_QUAD = 3   # 4-node quadrangle  (→ CQUAD4 in Nastran)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable meshing function (called by face_labeler.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_mesh(step_file: str, face_labels: dict, mesh_file: str) -> None:
    """
    Import a STEP file, create Gmsh physical groups from face_labels, generate
    a quad-dominant mesh, and write it to mesh_file.

    face_labels format:
        {
          "included": [2, 3, 8],       # Gmsh surface tags to include in mesh
          "groups": {
            "top":        [8],          # named physical groups → PSHELL regions
            "right_wall": [2],
            "step_face":  [3],
          }
        }

    Surfaces in "included" but not in any group are given a default physical
    group named "surf_<tag>" so they still appear in the mesh output.

    Gmsh behaviour: when any physical group is defined, only entities that
    belong to a physical group are exported to the .msh file.  "Excluded"
    surfaces (absent from "included") are meshed internally but not exported.
    """
    included = set(face_labels.get("included", []))
    groups   = face_labels.get("groups", {})

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("part")
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Named physical groups from user assignments
    assigned: set[int] = set()
    for name, tags in groups.items():
        valid = [t for t in tags if t in included]
        if valid:
            gmsh.model.addPhysicalGroup(2, valid, name=name)
            assigned.update(valid)

    # Included-but-unassigned surfaces get a fallback group so they export
    for tag in sorted(included - assigned):
        gmsh.model.addPhysicalGroup(2, [tag], name=f"surf_{tag}")

    # Quad-dominant mesh settings
    gmsh.option.setNumber("Mesh.Algorithm",               8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll",            1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3.0)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.write(mesh_file)

    # Verification
    print(f"\n── Mesh verification ({mesh_file}) ────────────────────────────────")
    all_ok = True
    for name, tags in groups.items():
        for surf_tag in tags:
            if surf_tag not in included:
                continue
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2, surf_tag)
            counts = {et: len(etags) for et, etags in zip(elem_types, elem_tags)}
            n_quad = counts.get(ETYPE_QUAD, 0)
            n_tri  = counts.get(ETYPE_TRI,  0)
            total  = n_quad + n_tri
            pct    = 100.0 * n_quad / total if total else 0.0
            ok     = "✓" if n_quad > 0 else "✗ NO QUADS"
            if n_quad == 0:
                all_ok = False
            print(f"  {name:14s} surf {surf_tag:3d}  quads={n_quad:4d}  tris={n_tri:3d}  quad%={pct:5.1f}%  {ok}")

    gmsh.finalize()

    if not all_ok:
        raise RuntimeError("One or more surfaces produced no quad elements.")
    print("\nPASS: all named surfaces have quad elements.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo: CadQuery → STEP → centroid matching → run_mesh
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cadquery as cq

    STEP_FILE = "part.step"
    MESH_FILE = "part.msh"
    CENTROID_TOL = 0.5

    # Build L-shaped extrusion
    pts = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
    solid = cq.Workplane("XY").polyline(pts).close().extrude(5)

    def cq_centroid(face):
        c = face.Center()
        return np.array([c.x, c.y, c.z])

    all_faces = solid.faces().vals()

    FACE_TARGETS = {
        "top":        (8.33, 8.33, 5.0),
        "right_wall": (20.0,  5.0,  2.5),
        "step_face":  (15.0, 10.0,  2.5),
    }

    tagged_cq = {
        name: min(all_faces, key=lambda f: np.linalg.norm(cq_centroid(f) - np.array(xyz)))
        for name, xyz in FACE_TARGETS.items()
    }
    tag_centroids = {name: cq_centroid(f) for name, f in tagged_cq.items()}

    print("CadQuery tagged-face centroids:")
    for name, c in tag_centroids.items():
        print(f"  {name:12s}  ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

    cq.exporters.export(solid, STEP_FILE)
    print(f"\nExported → {STEP_FILE}")

    # Match CadQuery centroids to Gmsh surface tags
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("match")
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surf_centroids = {
        tag: np.array(gmsh.model.occ.getCenterOfMass(2, tag))
        for _, tag in surfaces
    }
    gmsh.finalize()

    print(f"\nGmsh found {len(surfaces)} surfaces")
    unmatched = set(t for _, t in surfaces)
    groups: dict[str, list[int]] = {}
    included: list[int] = []

    for face_name, ref_c in tag_centroids.items():
        best = min(unmatched, key=lambda t: np.linalg.norm(surf_centroids[t] - ref_c))
        dist = np.linalg.norm(surf_centroids[best] - ref_c)
        if dist > CENTROID_TOL:
            print(f"  WARNING: {face_name} match dist={dist:.4f}", file=sys.stderr)
        groups[face_name] = [best]
        included.append(best)
        unmatched.discard(best)
        print(f"  {face_name:12s} → surf {best}  (dist={dist:.5f})")

    face_labels = {"included": included, "groups": groups}
    run_mesh(STEP_FILE, face_labels, MESH_FILE)
