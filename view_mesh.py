#!/usr/bin/env python3
"""
Visualize part.msh in a browser via PyVista → panel HTML export.

Reads the Gmsh MSH, builds a PyVista UnstructuredGrid colored by physical
group (tagged surface region), and writes a self-contained HTML file that
renders interactively in any browser with WebGL — no display server needed.
"""

import sys
import numpy as np
import meshio
import pyvista as pv
import matplotlib.pyplot as plt

MESH_FILE = "part.msh"
HTML_OUT  = "mesh_view.html"

# ─── 1. Read MSH ─────────────────────────────────────────────────────────────
mesh = meshio.read(MESH_FILE)

tag_to_name = {
    int(v[0]): name
    for name, v in mesh.field_data.items()
    if v[1] == 2
}
print("Physical groups:", tag_to_name)

# ─── 2. Extract quads + region labels ────────────────────────────────────────
phys_tags = mesh.cell_data.get("gmsh:physical", [])
quad_conn, quad_label = [], []

for block, cell_tags in zip(mesh.cells, phys_tags):
    if block.type != "quad":
        continue
    for row, tag in zip(block.data, cell_tags):
        quad_conn.append(row)
        quad_label.append(int(tag))

if not quad_conn:
    sys.exit("No quad elements found.")

quad_conn  = np.array(quad_conn,  dtype=np.int64)
quad_label = np.array(quad_label, dtype=np.int32)
points     = mesh.points[:, :3]

print(f"{len(quad_conn)} quads, {len(points)} nodes")

# ─── 3. Build UnstructuredGrid ───────────────────────────────────────────────
VTK_QUAD   = 9
n          = len(quad_conn)
cells      = np.empty(n * 5, dtype=np.int64)
cells[0::5] = 4
cells[1::5] = quad_conn[:, 0]
cells[2::5] = quad_conn[:, 1]
cells[3::5] = quad_conn[:, 2]
cells[4::5] = quad_conn[:, 3]

grid = pv.UnstructuredGrid(cells, np.full(n, VTK_QUAD, dtype=np.uint8), points)
grid.cell_data["region_id"] = quad_label

# ─── 4. Plot and export HTML ──────────────────────────────────────────────────
# export_html() serializes the scene for VTK.js / WebGL — no display server needed.
pl = pv.Plotter(window_size=(1200, 800))
pl.set_background("white")

palette = plt.get_cmap("tab10")
unique_tags = np.unique(quad_label)

for i, tag in enumerate(unique_tags):
    name   = tag_to_name.get(tag, f"tag_{tag}")
    subset = grid.extract_cells(quad_label == tag)
    color  = [int(c * 255) for c in palette(i / max(len(unique_tags) - 1, 1))[:3]]
    pl.add_mesh(subset, color=color, show_edges=True,
                edge_color="black", line_width=1.5, label=name)

pl.add_legend(bcolor="white", border=True, size=(0.22, 0.18))
pl.add_axes()
pl.camera_position = "iso"

pl.export_html(HTML_OUT)
print(f"Saved → {HTML_OUT}  (open in any browser)")
