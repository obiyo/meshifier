#!/usr/bin/env python3
"""
face_labeler.py — integrated STEP labeler + Gmsh mesher + mesh viewer.

Supports physical groups for all three mesh dimensions:
  0D  vertices  →  point elements  (Nastran: CONM2, CELAS)
  1D  edges     →  bar/line elems  (Nastran: CBAR, CBEAM)
  2D  surfaces  →  shell elements  (Nastran: CQUAD4, CTRIA3)

Usage:
    conda run -n meshcad python face_labeler.py part.step

Phases:
  label  →  select entities by dim tab, assign groups, set mesh size, run mesh
  view   →  inspect mesh colored by group; click Edit Labels to iterate
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import gmsh
import meshio
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v, html
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Integrated STEP labeler + mesher")
parser.add_argument("step_file", help="Path to STEP file")
args = parser.parse_args()

STEP_PATH = Path(args.step_file).resolve()
JSON_PATH = STEP_PATH.with_name(STEP_PATH.stem + "_labels.json")
MESH_FILE = str(STEP_PATH.with_suffix(".msh"))

# ── Colors ────────────────────────────────────────────────────────────────────
C_EXCLUDED = (0.55, 0.55, 0.55)
C_INCLUDED = (0.92, 0.92, 0.92)
C_SELECTED = (1.00, 0.82, 0.00)

_cmap = plt.get_cmap("tab10")


def group_color_rgb(idx: int) -> tuple:
    return tuple(float(c) for c in _cmap(idx % 10)[:3])


def rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ── Load STEP + tessellate via Gmsh ───────────────────────────────────────────
print(f"\nLoading {STEP_PATH.name} ...")
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("labeler")
gmsh.model.occ.importShapes(str(STEP_PATH))
gmsh.model.occ.synchronize()

vertices = gmsh.model.getEntities(dim=0)
edges    = gmsh.model.getEntities(dim=1)
surfaces = gmsh.model.getEntities(dim=2)

surf_tags = [t for _, t in surfaces]
edge_tags = [t for _, t in edges]
vert_tags = [t for _, t in vertices]

bb      = gmsh.model.getBoundingBox(-1, -1)
bb_diag = ((bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2) ** 0.5
mesh_sz = max(bb_diag / 12, 0.5)
sphere_r = bb_diag * 0.012   # vertex sphere radius

# ── Surface info ──────────────────────────────────────────────────────────────
face_info: dict[int, dict] = {}
for _, tag in surfaces:
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, tag)
    try:
        area = float(gmsh.model.occ.getMass(2, tag))
    except Exception:
        area = 0.0
    face_info[tag] = {
        "centroid_xyz": (cx, cy, cz),
        "centroid":     f"({cx:.2f}, {cy:.2f}, {cz:.2f})",
        "area":         round(area, 3),
    }

# ── Edge info ─────────────────────────────────────────────────────────────────
edge_info: dict[int, dict] = {}
for _, tag in edges:
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(1, tag)
    try:
        length = float(gmsh.model.occ.getMass(1, tag))
    except Exception:
        length = 0.0
    edge_info[tag] = {
        "center_xyz": (cx, cy, cz),
        "center":     f"({cx:.2f}, {cy:.2f}, {cz:.2f})",
        "length":     round(length, 3),
    }

# ── Vertex info (getBoundingBox — getCenterOfMass returns 0,0,0 for dim=0) ───
vert_info: dict[int, dict] = {}
for _, tag in vertices:
    bb_v = gmsh.model.getBoundingBox(0, tag)
    x, y, z = bb_v[0], bb_v[1], bb_v[2]
    vert_info[tag] = {
        "pos_xyz": (x, y, z),
        "pos":     f"({x:.2f}, {y:.2f}, {z:.2f})",
    }

# ── Generate coarse tessellation mesh ─────────────────────────────────────────
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_sz)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_sz * 0.1)
gmsh.model.mesh.generate(2)

VTK_TRI  = 5
VTK_LINE = 3

# Surface tessellation grids
face_grids: dict[int, pv.UnstructuredGrid] = {}
for _, surf_tag in surfaces:
    node_tags, coords, _ = gmsh.model.mesh.getNodes(2, surf_tag, includeBoundary=True)
    if len(node_tags) == 0:
        continue
    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2, surf_tag)
    pts_arr = np.array(coords, dtype=float).reshape(-1, 3)
    g2l = {int(nt): i for i, nt in enumerate(node_tags)}
    tris = []
    for et, enodes in zip(elem_types, elem_nodes):
        if et != 2:
            continue
        rows = np.array(enodes, dtype=int).reshape(-1, 3)
        for row in rows:
            try:
                tris.append([g2l[int(r)] for r in row])
            except KeyError:
                pass
    if not tris:
        continue
    tris  = np.array(tris, dtype=np.int64)
    n     = len(tris)
    cells = np.empty(n * 4, dtype=np.int64)
    cells[0::4] = 3
    cells[1::4] = tris[:, 0]
    cells[2::4] = tris[:, 1]
    cells[3::4] = tris[:, 2]
    grid = pv.UnstructuredGrid(cells, np.full(n, VTK_TRI, dtype=np.uint8), pts_arr)
    face_grids[surf_tag] = grid

# Edge tessellation grids
edge_grids: dict[int, pv.UnstructuredGrid] = {}
for _, etag in edges:
    node_tags, coords, _ = gmsh.model.mesh.getNodes(1, etag, includeBoundary=True)
    if len(node_tags) == 0:
        continue
    _, _, elem_nodes = gmsh.model.mesh.getElements(1, etag)
    pts_arr = np.array(coords, dtype=float).reshape(-1, 3)
    g2l = {int(nt): i for i, nt in enumerate(node_tags)}
    lines = []
    for enodes in elem_nodes:
        rows = np.array(enodes, dtype=int).reshape(-1, 2)
        for row in rows:
            try:
                lines.append([g2l[int(r)] for r in row])
            except KeyError:
                pass
    if not lines:
        continue
    lines_arr = np.array(lines, dtype=np.int64)
    n     = len(lines_arr)
    cells = np.empty(n * 3, dtype=np.int64)
    cells[0::3] = 2
    cells[1::3] = lines_arr[:, 0]
    cells[2::3] = lines_arr[:, 1]
    grid = pv.UnstructuredGrid(cells, np.full(n, VTK_LINE, dtype=np.uint8), pts_arr)
    edge_grids[etag] = grid

gmsh.finalize()
print(f"  {len(surf_tags)} surfaces, {len(edge_tags)} edges, {len(vert_tags)} vertices  "
      f"(preview mesh size ≈ {mesh_sz:.1f})")

# ── PyVista plotter ───────────────────────────────────────────────────────────
pv.global_theme.trame.default_mode = "client"
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

pl = pv.Plotter()
pl.set_background("white")

# Surface actors (label phase)
surf_actors: dict[int, pv.Actor] = {}
for surf_tag, grid in face_grids.items():
    actor = pl.add_mesh(grid, color=C_EXCLUDED, show_edges=True,
                        edge_color="dimgray", line_width=0.5,
                        smooth_shading=True, show_scalar_bar=False)
    surf_actors[surf_tag] = actor

# Edge actors (label phase — on top of surfaces)
e_actors: dict[int, pv.Actor] = {}
for etag, grid in edge_grids.items():
    actor = pl.add_mesh(grid, color=C_EXCLUDED, line_width=3,
                        show_scalar_bar=False)
    e_actors[etag] = actor

# Vertex actors (label phase — rendered as spheres)
v_actors: dict[int, pv.Actor] = {}
for vtag, info in vert_info.items():
    sphere = pv.Sphere(radius=sphere_r, center=info["pos_xyz"])
    actor  = pl.add_mesh(sphere, color=C_EXCLUDED, show_scalar_bar=False)
    v_actors[vtag] = actor

# Centroid number labels
centroid_pts  = np.array([face_info[t]["centroid_xyz"] for t in sorted(face_grids)])
centroid_strs = [str(t) for t in sorted(face_grids)]
label_actor   = pl.add_point_labels(
    centroid_pts, centroid_strs,
    font_size=14, point_size=1, bold=True,
    text_color="black", shape_color="white", shape_opacity=0.6,
    always_visible=True,
)

pl.camera_position = "iso"

# Mesh actors (view phase) and per-surface mesh type
mesh_actors: list = []
mesh_types: dict[int, str] = {}   # surf_tag → "quad"|"structured"|"tri"

# ── App state ─────────────────────────────────────────────────────────────────
# Shared
state.app_phase     = "label"   # "label" | "view"
state.active_dim    = "surf"    # "surf" | "edge" | "vert"
state.status_msg    = "Select a dimension tab, then click a chip to select an entity."
state.mesh_size     = round(mesh_sz, 1)
state.mesh_stats    = ""

# Surface
state.selected_surf   = None
state.included        = []
state.groups          = {}
state.group_input     = ""
state.existing_groups = []
state.sel_surf_text   = "—"
state.sel_centroid    = "—"
state.sel_area        = "—"
state.sel_status      = "—"
state.sel_mesh_type   = "quad"
state.has_selection   = False
state.has_groups      = False
state.surf_chips      = []
state.group_summary   = []

# Edge
state.selected_edge      = None
state.edge_included      = []
state.edge_groups        = {}
state.edge_group_input   = ""
state.edge_existing_grps = []
state.sel_edge_text      = "—"
state.sel_edge_center    = "—"
state.sel_edge_length    = "—"
state.sel_edge_status    = "—"
state.has_edge_sel       = False
state.has_edge_groups    = False
state.edge_chips         = []
state.edge_group_summary = []

# Vertex
state.selected_vert      = None
state.vert_included      = []
state.vert_groups        = {}
state.vert_group_input   = ""
state.vert_existing_grps = []
state.sel_vert_text      = "—"
state.sel_vert_pos       = "—"
state.sel_vert_status    = "—"
state.has_vert_sel       = False
state.has_vert_groups    = False
state.vert_chips         = []
state.vert_group_summary = []

_SLIDER_MAX = round(bb_diag / 3, 1)
_SLIDER_MIN = 0.5

# ── Generic helpers ───────────────────────────────────────────────────────────
def _find_in_groups(tag: int, groups: dict) -> str | None:
    for name, tags in groups.items():
        if tag in tags:
            return name
    return None


def _entity_color_rgb(tag: int, included: list, groups: dict,
                      selected: int | None, mesh_types_dict: dict | None = None) -> tuple:
    group_names = list(groups.keys())
    if tag == selected:
        return C_SELECTED
    grp = _find_in_groups(tag, groups)
    if grp is not None:
        return group_color_rgb(group_names.index(grp))
    if tag in included:
        return C_INCLUDED
    return C_EXCLUDED


def _build_entity_chips(tags: list, included: list, groups: dict,
                        selected: int | None, extra_badge: dict | None = None) -> list:
    chips = []
    for tag in sorted(tags):
        rgb = _entity_color_rgb(tag, included, groups, selected)
        hex_color  = rgb_to_hex(*rgb)
        luminance  = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = "#000000" if luminance > 0.55 else "#ffffff"
        grp = _find_in_groups(tag, groups)
        if tag == selected:
            suffix = " ★"
        elif grp:
            suffix = f" [{grp}]"
        elif tag in included:
            suffix = " ✓"
        else:
            suffix = ""
        badge = (extra_badge or {}).get(tag, "")
        chips.append({
            "tag": tag, "label": f"{tag}{suffix}{badge}",
            "color": hex_color, "text_color": text_color,
        })
    return chips

# ── Surface helpers ───────────────────────────────────────────────────────────
def _build_surf_chips() -> list:
    type_badge = {"structured": " (S)", "tri": " (T)"}
    badges = {t: type_badge.get(mesh_types.get(t, "quad"), "") for t in surf_tags}
    return _build_entity_chips(surf_tags, state.included, state.groups,
                                state.selected_surf, badges)


def _refresh_surf_colors():
    for tag, actor in surf_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.included, state.groups, state.selected_surf))
    state.surf_chips = _build_surf_chips()
    ctrl.view_update()


def _sync_surf_groups():
    state.existing_groups = list(state.groups.keys())
    state.group_summary   = [
        f"{n}  →  surf {', '.join(str(t) for t in ts)}"
        for n, ts in state.groups.items()
    ]
    state.has_groups = bool(state.groups)


def _update_surf_display(tag: int):
    info = face_info.get(tag, {})
    grp  = _find_in_groups(tag, state.groups)
    status = f"group: {grp}" if grp else ("included (unassigned)" if tag in state.included else "excluded")
    state.sel_surf_text  = f"surf {tag}"
    state.sel_centroid   = info.get("centroid", "—")
    state.sel_area       = str(info.get("area", "—"))
    state.sel_status     = status
    state.sel_mesh_type  = mesh_types.get(tag, "quad")
    state.has_selection  = True

# ── Edge helpers ──────────────────────────────────────────────────────────────
def _build_edge_chips() -> list:
    return _build_entity_chips(edge_tags, state.edge_included, state.edge_groups,
                                state.selected_edge)


def _refresh_edge_colors():
    for tag, actor in e_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.edge_included, state.edge_groups,
                               state.selected_edge))
    state.edge_chips = _build_edge_chips()
    ctrl.view_update()


def _sync_edge_groups():
    state.edge_existing_grps = list(state.edge_groups.keys())
    state.edge_group_summary = [
        f"{n}  →  edge {', '.join(str(t) for t in ts)}"
        for n, ts in state.edge_groups.items()
    ]
    state.has_edge_groups = bool(state.edge_groups)


def _update_edge_display(tag: int):
    info = edge_info.get(tag, {})
    grp  = _find_in_groups(tag, state.edge_groups)
    status = f"group: {grp}" if grp else ("included" if tag in state.edge_included else "excluded")
    state.sel_edge_text   = f"edge {tag}"
    state.sel_edge_center = info.get("center", "—")
    state.sel_edge_length = str(info.get("length", "—"))
    state.sel_edge_status = status
    state.has_edge_sel    = True

# ── Vertex helpers ────────────────────────────────────────────────────────────
def _build_vert_chips() -> list:
    return _build_entity_chips(vert_tags, state.vert_included, state.vert_groups,
                                state.selected_vert)


def _refresh_vert_colors():
    for tag, actor in v_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.vert_included, state.vert_groups,
                               state.selected_vert))
    state.vert_chips = _build_vert_chips()
    ctrl.view_update()


def _sync_vert_groups():
    state.vert_existing_grps = list(state.vert_groups.keys())
    state.vert_group_summary = [
        f"{n}  →  vert {', '.join(str(t) for t in ts)}"
        for n, ts in state.vert_groups.items()
    ]
    state.has_vert_groups = bool(state.vert_groups)


def _update_vert_display(tag: int):
    info  = vert_info.get(tag, {})
    grp   = _find_in_groups(tag, state.vert_groups)
    status = f"group: {grp}" if grp else ("included" if tag in state.vert_included else "excluded")
    state.sel_vert_text   = f"vert {tag}"
    state.sel_vert_pos    = info.get("pos", "—")
    state.sel_vert_status = status
    state.has_vert_sel    = True

# ── Phase visibility ──────────────────────────────────────────────────────────
def _set_tess_visible(on: bool):
    for actor in surf_actors.values():
        actor.SetVisibility(on)
    for actor in e_actors.values():
        actor.SetVisibility(on)
    for actor in v_actors.values():
        actor.SetVisibility(on)
    try:
        label_actor.SetVisibility(on)
    except Exception:
        pass


def _set_mesh_visible(on: bool):
    for actor in mesh_actors:
        actor.SetVisibility(on)

# ── Mesh-actor builder ────────────────────────────────────────────────────────
def _build_mesh_actors(mesh_file: str, group_orders: dict) -> tuple:
    """
    group_orders: {"surf": [name,...], "edge": [name,...], "vert": [name,...]}
    Returns (actor_list, stats_str).
    """
    mesh = meshio.read(mesh_file)
    tag_to_name = {int(v[0]): name for name, v in mesh.field_data.items() if v[1] in (0,1,2)}
    phys_tags = mesh.cell_data.get("gmsh:physical", [])

    # Collect elements by type
    buckets: dict[str, tuple[list, list]] = {
        "quad": ([], []), "triangle": ([], []),
        "line": ([], []), "vertex":   ([], []),
    }
    for block, cell_tags in zip(mesh.cells, phys_tags):
        if block.type in buckets:
            conn, labels = buckets[block.type]
            for row, tag in zip(block.data, cell_tags):
                conn.append(row)
                labels.append(int(tag))

    VTK_QUAD, VTK_TRI, VTK_LINE, VTK_PT = 9, 5, 3, 1
    vtk_map = {"quad": VTK_QUAD, "triangle": VTK_TRI, "line": VTK_LINE, "vertex": VTK_PT}

    new_actors = []
    counts = {}

    points = mesh.points[:, :3]

    for btype, (conn, labels) in buckets.items():
        if not conn:
            continue
        counts[btype] = len(conn)
        vtk_type = vtk_map[btype]

        # Build variable-length cell array
        cell_parts = []
        for row in conn:
            cell_parts.append(len(row))
            cell_parts.extend(row)
        cells_arr = np.array(cell_parts, dtype=np.int64)
        lab_arr   = np.array(labels, dtype=np.int32)
        types_arr = np.full(len(conn), vtk_type, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells_arr, types_arr, points)
        grid.cell_data["region_id"] = lab_arr

        order = group_orders.get(
            "surf" if btype in ("quad", "triangle") else
            "edge" if btype == "line" else "vert", []
        )

        for tag in np.unique(lab_arr):
            name = tag_to_name.get(int(tag), f"tag_{tag}")
            try:
                idx = order.index(name)
            except ValueError:
                idx = int(tag) % 10
            color_rgb = group_color_rgb(idx)
            color_255 = [int(c * 255) for c in color_rgb]
            subset = grid.extract_cells(lab_arr == tag)
            kw = dict(color=color_255, show_scalar_bar=False)
            if btype in ("quad", "triangle"):
                kw.update(show_edges=True, edge_color="black", line_width=1.0)
            elif btype == "line":
                kw["line_width"] = 4.0
            else:  # vertex
                kw.update(point_size=14, render_points_as_spheres=True)
            actor = pl.add_mesh(subset, **kw)
            actor.SetVisibility(False)
            new_actors.append(actor)

    parts = []
    for t in ("quad", "triangle", "line", "vertex"):
        if t in counts:
            parts.append(f"{counts[t]} {t}s")
    stats = "  ·  ".join(parts) if parts else "no elements"
    return new_actors, stats

# ── Surface callbacks ─────────────────────────────────────────────────────────
def select_surf(tag, **kwargs):
    state.selected_surf = int(tag)
    _update_surf_display(int(tag))
    _refresh_surf_colors()


def include_face():
    tag = state.selected_surf
    if tag is None: return
    if tag not in state.included:
        state.included = state.included + [tag]
    _update_surf_display(tag)
    _refresh_surf_colors()


def exclude_face():
    tag = state.selected_surf
    if tag is None: return
    state.included = [t for t in state.included if t != tag]
    state.groups   = {k: [t for t in v if t != tag] for k, v in state.groups.items()}
    state.groups   = {k: v for k, v in state.groups.items() if v}
    _sync_surf_groups()
    _update_surf_display(tag)
    _refresh_surf_colors()


def assign_group():
    tag  = state.selected_surf
    name = (state.group_input or "").strip()
    if tag is None or not name: return
    if tag not in state.included:
        state.included = state.included + [tag]
    ng = {k: [t for t in v if t != tag] for k, v in state.groups.items()}
    ng = {k: v for k, v in ng.items() if v}
    ng.setdefault(name, [])
    ng[name] = ng[name] + [tag]
    state.groups = ng
    _sync_surf_groups()
    _update_surf_display(tag)
    _refresh_surf_colors()


@state.change("sel_mesh_type")
def _on_mesh_type_change(sel_mesh_type, **kwargs):
    tag = state.selected_surf
    if tag is None: return
    mesh_types[tag] = sel_mesh_type
    state.surf_chips = _build_surf_chips()

# ── Edge callbacks ────────────────────────────────────────────────────────────
def select_edge(tag, **kwargs):
    state.selected_edge = int(tag)
    _update_edge_display(int(tag))
    _refresh_edge_colors()


def include_edge():
    tag = state.selected_edge
    if tag is None: return
    if tag not in state.edge_included:
        state.edge_included = state.edge_included + [tag]
    _update_edge_display(tag)
    _refresh_edge_colors()


def exclude_edge():
    tag = state.selected_edge
    if tag is None: return
    state.edge_included = [t for t in state.edge_included if t != tag]
    state.edge_groups   = {k: [t for t in v if t != tag] for k, v in state.edge_groups.items()}
    state.edge_groups   = {k: v for k, v in state.edge_groups.items() if v}
    _sync_edge_groups()
    _update_edge_display(tag)
    _refresh_edge_colors()


def assign_edge_group():
    tag  = state.selected_edge
    name = (state.edge_group_input or "").strip()
    if tag is None or not name: return
    if tag not in state.edge_included:
        state.edge_included = state.edge_included + [tag]
    ng = {k: [t for t in v if t != tag] for k, v in state.edge_groups.items()}
    ng = {k: v for k, v in ng.items() if v}
    ng.setdefault(name, [])
    ng[name] = ng[name] + [tag]
    state.edge_groups = ng
    _sync_edge_groups()
    _update_edge_display(tag)
    _refresh_edge_colors()

# ── Vertex callbacks ──────────────────────────────────────────────────────────
def select_vert(tag, **kwargs):
    state.selected_vert = int(tag)
    _update_vert_display(int(tag))
    _refresh_vert_colors()


def include_vert():
    tag = state.selected_vert
    if tag is None: return
    if tag not in state.vert_included:
        state.vert_included = state.vert_included + [tag]
    _update_vert_display(tag)
    _refresh_vert_colors()


def exclude_vert():
    tag = state.selected_vert
    if tag is None: return
    state.vert_included = [t for t in state.vert_included if t != tag]
    state.vert_groups   = {k: [t for t in v if t != tag] for k, v in state.vert_groups.items()}
    state.vert_groups   = {k: v for k, v in state.vert_groups.items() if v}
    _sync_vert_groups()
    _update_vert_display(tag)
    _refresh_vert_colors()


def assign_vert_group():
    tag  = state.selected_vert
    name = (state.vert_group_input or "").strip()
    if tag is None or not name: return
    if tag not in state.vert_included:
        state.vert_included = state.vert_included + [tag]
    ng = {k: [t for t in v if t != tag] for k, v in state.vert_groups.items()}
    ng = {k: v for k, v in ng.items() if v}
    ng.setdefault(name, [])
    ng[name] = ng[name] + [tag]
    state.vert_groups = ng
    _sync_vert_groups()
    _update_vert_display(tag)
    _refresh_vert_colors()

# ── JSON export / import ──────────────────────────────────────────────────────
def export_json():
    data = {
        "included":      list(state.included),
        "groups":        {n: list(ts) for n, ts in state.groups.items()},
        "mesh_types":    {str(k): v for k, v in mesh_types.items() if v != "quad"},
        "edge_included": list(state.edge_included),
        "edge_groups":   {n: list(ts) for n, ts in state.edge_groups.items()},
        "vert_included": list(state.vert_included),
        "vert_groups":   {n: list(ts) for n, ts in state.vert_groups.items()},
    }
    with open(str(JSON_PATH), "w") as f:
        json.dump(data, f, indent=2)
    state.status_msg = f"Saved → {JSON_PATH.name}"
    print(f"\nExported: {JSON_PATH}")
    print(json.dumps(data, indent=2))


def _apply_json(data: dict):
    state.included      = data.get("included",      [])
    state.groups        = data.get("groups",        {})
    state.edge_included = data.get("edge_included", [])
    state.edge_groups   = data.get("edge_groups",   {})
    state.vert_included = data.get("vert_included", [])
    state.vert_groups   = data.get("vert_groups",   {})
    mesh_types.clear()
    mesh_types.update({int(k): v for k, v in data.get("mesh_types", {}).items()})
    _sync_surf_groups()
    _sync_edge_groups()
    _sync_vert_groups()
    if state.selected_surf is not None:
        state.sel_mesh_type = mesh_types.get(state.selected_surf, "quad")
    for tag, actor in surf_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.included, state.groups, state.selected_surf))
    for tag, actor in e_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.edge_included, state.edge_groups, state.selected_edge))
    for tag, actor in v_actors.items():
        actor.GetProperty().SetColor(
            *_entity_color_rgb(tag, state.vert_included, state.vert_groups, state.selected_vert))
    state.surf_chips = _build_surf_chips()
    state.edge_chips = _build_edge_chips()
    state.vert_chips = _build_vert_chips()


def load_json():
    if not JSON_PATH.exists():
        state.status_msg = f"Not found: {JSON_PATH.name}"
        return
    with open(JSON_PATH) as f:
        data = json.load(f)
    _apply_json(data)
    try:
        ctrl.view_update()
    except Exception:
        pass
    state.status_msg = f"Loaded {JSON_PATH.name}"

# ── Mesh action ───────────────────────────────────────────────────────────────
def run_mesh_action():
    global mesh_actors
    if not state.groups:
        state.status_msg = "Assign at least one surface group before meshing."
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from step1_tag_mesh import run_mesh

    data = {
        "included":      list(state.included),
        "groups":        {n: list(ts) for n, ts in state.groups.items()},
        "mesh_types":    dict(mesh_types),
        "edge_included": list(state.edge_included),
        "edge_groups":   {n: list(ts) for n, ts in state.edge_groups.items()},
        "vert_included": list(state.vert_included),
        "vert_groups":   {n: list(ts) for n, ts in state.vert_groups.items()},
    }
    group_orders = {
        "surf": list(state.groups.keys()),
        "edge": list(state.edge_groups.keys()),
        "vert": list(state.vert_groups.keys()),
    }
    mesh_size = float(state.mesh_size)
    state.status_msg = "Meshing…"

    try:
        run_mesh(str(STEP_PATH), data, MESH_FILE,
                 mesh_size_max=mesh_size,
                 mesh_size_min=mesh_size * 0.4,
                 mesh_types=data["mesh_types"])
    except Exception as e:
        state.status_msg = f"Mesh failed: {e}"
        import traceback; traceback.print_exc()
        return

    for actor in mesh_actors:
        pl.remove_actor(actor)
    mesh_actors = []

    mesh_actors, stats = _build_mesh_actors(MESH_FILE, group_orders)
    state.mesh_stats = stats

    _set_tess_visible(False)
    _set_mesh_visible(True)
    state.app_phase  = "view"
    state.status_msg = f"Done: {stats}"
    ctrl.view_update()


def edit_labels():
    _set_mesh_visible(False)
    _set_tess_visible(True)
    state.app_phase  = "label"
    state.status_msg = "Edit groups, then Run Mesh again."
    ctrl.view_update()

# ── Auto-load labels JSON if it exists ───────────────────────────────────────
if JSON_PATH.exists():
    try:
        with open(JSON_PATH) as _f:
            _apply_json(json.load(_f))
        print(f"  Auto-loaded labels from {JSON_PATH.name}")
    except Exception as _e:
        print(f"  Could not auto-load {JSON_PATH.name}: {_e}")

# ── Pre-populate chips ────────────────────────────────────────────────────────
state.surf_chips = _build_surf_chips()
state.edge_chips = _build_edge_chips()
state.vert_chips = _build_vert_chips()

# ── trame layout ──────────────────────────────────────────────────────────────
PANEL_BG = "background:#fafafa; border-left:1px solid #e0e0e0; height:100%; overflow-y:auto"
HDR  = "font-weight:600; font-size:0.85em; margin-bottom:6px"
SMALL = "font-size:0.78em; margin:2px 0; color:#444"
MONO  = "font-size:0.80em; padding:2px 0; font-family:monospace; color:#333"


def _dim_panel(dim_key: str,
               chips_state: str,
               select_cb, include_cb, exclude_cb,
               assign_cb,
               group_input_state: str,
               existing_grps_state: str,
               has_sel_state: str,
               has_grps_state: str,
               group_summary_state: str,
               sel_info_lines: list,
               show_mesh_type: bool = False):
    """Reusable right-panel section for one dimension."""
    with html.Div(v_if=f"app_phase === 'label' && active_dim === '{dim_key}'"):

        # Chips
        with html.Div(style="display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px"):
            v.VChip(
                "{{ chip.label }}",
                v_for=f"chip in {chips_state}",
                key=("chip.tag",),
                style=(f"'background:' + chip.color + '; color:' + chip.text_color"
                       " + '; cursor:pointer; font-size:0.78em'",),
                size="small",
                click=(select_cb, "[chip.tag]"),
            )

        v.VDivider(classes="mb-3")

        # Selected entity info
        html.P("Selected", style=HDR)
        with v.VCard(variant="outlined", classes="mb-2"):
            with v.VCardText(classes="pa-2"):
                with html.Div(v_if=f"!{has_sel_state}"):
                    html.P("Click a chip above",
                           style="color:#aaa; font-size:0.82em; margin:0; font-style:italic")
                with html.Div(v_if=has_sel_state):
                    for label, val_expr in sel_info_lines:
                        with html.Div(style="display:flex; gap:4px"):
                            html.Span(label, style="font-size:0.78em; color:#777; min-width:52px")
                            html.Span(f"{{{{ {val_expr} }}}}", style=SMALL)

        # Include / Exclude
        with v.VRow(no_gutters=True, classes="mb-2"):
            with v.VCol(cols=6, classes="pr-1"):
                v.VBtn("Include", block=True, color="primary", variant="outlined",
                       size="small", disabled=(f"!{has_sel_state}",), click=include_cb)
            with v.VCol(cols=6, classes="pl-1"):
                v.VBtn("Exclude", block=True, color="error", variant="outlined",
                       size="small", disabled=(f"!{has_sel_state}",), click=exclude_cb)

        # Mesh type toggle (surfaces only)
        if show_mesh_type:
            with html.Div(v_if=has_sel_state,
                          style="margin-bottom:10px"):
                html.P("Mesh type",
                       style="font-size:0.78em; color:#555; margin:4px 0 4px 0")
                with v.VBtnToggle(
                    v_model=("sel_mesh_type", "quad"),
                    density="compact", variant="outlined",
                    color="primary", mandatory=True,
                    style="width:100%",
                ):
                    v.VBtn("Quad",   value="quad",       size="x-small", style="flex:1")
                    v.VBtn("Struct", value="structured", size="x-small", style="flex:1")
                    v.VBtn("Tri",    value="tri",        size="x-small", style="flex:1")

        v.VDivider(classes="mb-3")

        # Group assignment
        html.P("Assign to group", style=HDR)
        v.VCombobox(
            v_model=(group_input_state, ""),
            label="Group name",
            items=(existing_grps_state, []),
            density="compact", hide_details=True, classes="mb-2",
        )
        v.VBtn("Assign to group", block=True, color="success", variant="tonal",
               size="small", classes="mb-3",
               disabled=(f"!{has_sel_state} || !{group_input_state}",),
               click=assign_cb)

        v.VDivider(classes="mb-3")

        # Group summary
        html.P("Groups", style=HDR)
        with html.Div(v_if=f"!{has_grps_state}"):
            html.P("No groups yet",
                   style="color:#aaa; font-size:0.82em; font-style:italic")
        with html.Div(v_if=has_grps_state):
            with html.Div(v_for=f"line in {group_summary_state}", key="line", style=MONO):
                html.Span("● {{ line }}")


with SinglePageLayout(server) as layout:
    layout.title.set_text("Meshifier")
    layout.icon.hide()

    with layout.toolbar:
        v.VSpacer()
        html.Span(
            f"{STEP_PATH.name}  ·  {len(surf_tags)} surfs  "
            f"{len(edge_tags)} edges  {len(vert_tags)} verts",
            style="font-size:0.9em; color:#666; padding-right:16px",
        )

    with layout.content:
        with v.VContainer(fluid=True, classes="fill-height pa-0"):
            with v.VRow(style="height:100%", no_gutters=True):

                # ── 3D view ────────────────────────────────────────────────
                with v.VCol(cols=9, classes="fill-height"):
                    view = plotter_ui(pl, full_size=True)
                    ctrl.view_update = view.update

                # ── Right panel ────────────────────────────────────────────
                with v.VCol(cols=3, classes="pa-3", style=PANEL_BG):

                    # ═══ LABEL PHASE ═══════════════════════════════════════
                    with html.Div(v_if="app_phase === 'label'"):

                        # Dimension tab selector
                        html.P("Entity type", style=HDR)
                        with v.VBtnToggle(
                            v_model=("active_dim", "surf"),
                            density="compact", variant="outlined",
                            color="primary", mandatory=True,
                            style="width:100%; margin-bottom:12px",
                        ):
                            v.VBtn("Surfaces", value="surf", size="x-small", style="flex:1")
                            v.VBtn("Edges",    value="edge", size="x-small", style="flex:1")
                            v.VBtn("Vertices", value="vert", size="x-small", style="flex:1")

                        # Surface panel
                        _dim_panel(
                            dim_key="surf",
                            chips_state="surf_chips",
                            select_cb=select_surf,
                            include_cb=include_face,
                            exclude_cb=exclude_face,
                            assign_cb=assign_group,
                            group_input_state="group_input",
                            existing_grps_state="existing_groups",
                            has_sel_state="has_selection",
                            has_grps_state="has_groups",
                            group_summary_state="group_summary",
                            sel_info_lines=[
                                ("ID",       "sel_surf_text"),
                                ("Centroid", "sel_centroid"),
                                ("Area",     "sel_area"),
                                ("Status",   "sel_status"),
                            ],
                            show_mesh_type=True,
                        )

                        # Edge panel
                        _dim_panel(
                            dim_key="edge",
                            chips_state="edge_chips",
                            select_cb=select_edge,
                            include_cb=include_edge,
                            exclude_cb=exclude_edge,
                            assign_cb=assign_edge_group,
                            group_input_state="edge_group_input",
                            existing_grps_state="edge_existing_grps",
                            has_sel_state="has_edge_sel",
                            has_grps_state="has_edge_groups",
                            group_summary_state="edge_group_summary",
                            sel_info_lines=[
                                ("ID",     "sel_edge_text"),
                                ("Center", "sel_edge_center"),
                                ("Length", "sel_edge_length"),
                                ("Status", "sel_edge_status"),
                            ],
                        )

                        # Vertex panel
                        _dim_panel(
                            dim_key="vert",
                            chips_state="vert_chips",
                            select_cb=select_vert,
                            include_cb=include_vert,
                            exclude_cb=exclude_vert,
                            assign_cb=assign_vert_group,
                            group_input_state="vert_group_input",
                            existing_grps_state="vert_existing_grps",
                            has_sel_state="has_vert_sel",
                            has_grps_state="has_vert_groups",
                            group_summary_state="vert_group_summary",
                            sel_info_lines=[
                                ("ID",     "sel_vert_text"),
                                ("Pos",    "sel_vert_pos"),
                                ("Status", "sel_vert_status"),
                            ],
                        )

                        # ── Bottom controls (always visible in label phase) ─
                        v.VDivider(classes="my-3")

                        html.P("Element size", style=HDR)
                        v.VSlider(
                            v_model=("mesh_size", round(mesh_sz, 1)),
                            min=_SLIDER_MIN, max=_SLIDER_MAX, step=0.5,
                            thumb_label=True, color="secondary",
                            hide_details=True, classes="mb-3",
                        )

                        with v.VRow(no_gutters=True, classes="mb-2"):
                            with v.VCol(cols=6, classes="pr-1"):
                                v.VBtn("Export JSON", block=True, color="primary",
                                       variant="tonal", size="small",
                                       disabled=("!has_groups",), click=export_json)
                            with v.VCol(cols=6, classes="pl-1"):
                                v.VBtn("Load JSON", block=True, color="default",
                                       variant="outlined", size="small", click=load_json)

                        v.VBtn("Run Mesh", block=True, color="secondary",
                               variant="tonal", classes="mb-2",
                               disabled=("!has_groups",), click=run_mesh_action)

                        html.P("{{ status_msg }}",
                               style="font-size:0.78em; color:#555; margin-top:6px; min-height:1.2em")

                    # ═══ VIEW PHASE ════════════════════════════════════════
                    with html.Div(v_if="app_phase === 'view'"):

                        html.P("Mesh result", style=HDR)
                        with v.VCard(variant="outlined", classes="mb-3"):
                            with v.VCardText(classes="pa-2"):
                                html.P("{{ mesh_stats }}",
                                       style="font-size:0.85em; font-family:monospace; margin:0")

                        for label, state_var in [
                            ("Surfaces", "group_summary"),
                            ("Edges",    "edge_group_summary"),
                            ("Vertices", "vert_group_summary"),
                        ]:
                            html.P(label, style=HDR)
                            with html.Div(v_for=f"line in {state_var}", key="line",
                                          style=MONO + "; margin-bottom:2px"):
                                html.Span("● {{ line }}")

                        v.VDivider(classes="my-3")

                        v.VBtn("Edit Labels", block=True, color="primary",
                               variant="tonal", classes="mb-2", click=edit_labels)
                        v.VBtn("Re-mesh", block=True, color="secondary",
                               variant="outlined", classes="mb-2", click=run_mesh_action)
                        v.VBtn("Export JSON", block=True, color="default",
                               variant="outlined", classes="mb-2", click=export_json)

                        html.P("{{ status_msg }}",
                               style="font-size:0.78em; color:#555; margin-top:6px; min-height:1.2em")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server.start(open_browser=True, port=8080)
