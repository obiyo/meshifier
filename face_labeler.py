#!/usr/bin/env python3
"""
face_labeler.py — browser GUI for STEP face selection and group labeling.

Usage:
    conda run -n meshcad python face_labeler.py part.step

Then open http://localhost:8080 in your browser.

Workflow:
  1. Each surface is labeled with its Gmsh surface tag (number) in the 3D view.
  2. Click the matching chip in the right panel to select a surface.
     The selected surface highlights yellow in the 3D view.
  3. Click Include to add it to the mesh, Exclude to remove it.
  4. Type a group name and click Assign — the surface joins that group.
     Group names become Gmsh physical surfaces → Nastran PSHELL regions.
  5. Repeat for all surfaces of interest.
  6. Click Export JSON to save <stem>_labels.json.
  7. Click Run Mesh to generate the quad mesh via Gmsh.

Color legend:
  Grey   = excluded (no mesh elements in output)
  White  = included, not yet assigned to a group
  Yellow = currently selected
  Color  = assigned to a named group (distinct color per group)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import gmsh
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v, html
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive STEP face labeler")
parser.add_argument("step_file", help="Path to STEP file")
args = parser.parse_args()

STEP_PATH = Path(args.step_file).resolve()
JSON_PATH = STEP_PATH.with_name(STEP_PATH.stem + "_labels.json")

# ── Colors ────────────────────────────────────────────────────────────────────
C_EXCLUDED = (0.55, 0.55, 0.55)
C_INCLUDED = (0.92, 0.92, 0.92)
C_SELECTED = (1.00, 0.82, 0.00)

_cmap = plt.get_cmap("tab10")

def group_color_rgb(idx: int) -> tuple:
    return tuple(float(c) for c in _cmap(idx % 10)[:3])

def rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

C_EXCLUDED_HEX = rgb_to_hex(*C_EXCLUDED)
C_INCLUDED_HEX = rgb_to_hex(*C_INCLUDED)
C_SELECTED_HEX = rgb_to_hex(*C_SELECTED)

# ── Load STEP + tessellate via Gmsh ───────────────────────────────────────────
print(f"\nLoading {STEP_PATH.name} ...")
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("labeler")
gmsh.model.occ.importShapes(str(STEP_PATH))
gmsh.model.occ.synchronize()

surfaces  = gmsh.model.getEntities(dim=2)
surf_tags = [t for _, t in surfaces]

bb      = gmsh.model.getBoundingBox(-1, -1)
bb_diag = ((bb[3]-bb[0])**2 + (bb[4]-bb[1])**2 + (bb[5]-bb[2])**2) ** 0.5
mesh_sz = max(bb_diag / 12, 0.5)

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

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_sz)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_sz * 0.1)
gmsh.model.mesh.generate(2)

VTK_TRI = 5

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

    tris = np.array(tris, dtype=np.int64)
    n    = len(tris)
    cells = np.empty(n * 4, dtype=np.int64)
    cells[0::4] = 3
    cells[1::4] = tris[:, 0]
    cells[2::4] = tris[:, 1]
    cells[3::4] = tris[:, 2]

    grid = pv.UnstructuredGrid(cells, np.full(n, VTK_TRI, dtype=np.uint8), pts_arr)
    grid.cell_data["face_id"] = np.full(n, surf_tag, dtype=np.int32)
    face_grids[surf_tag] = grid

gmsh.finalize()
print(f"  {len(surf_tags)} surfaces, {len(face_grids)} tessellated  "
      f"(preview mesh size ≈ {mesh_sz:.1f})")

# ── PyVista plotter ───────────────────────────────────────────────────────────
pv.global_theme.trame.default_mode = "client"
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

pl = pv.Plotter()
pl.set_background("white")

actors: dict[int, pv.Actor] = {}
for surf_tag, grid in face_grids.items():
    actor = pl.add_mesh(
        grid,
        color=C_EXCLUDED,
        show_edges=True,
        edge_color="dimgray",
        line_width=0.5,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    actors[surf_tag] = actor

# Centroid labels in the 3D view so users can match chips to faces
centroid_pts = np.array(
    [face_info[t]["centroid_xyz"] for t in sorted(face_grids.keys())]
)
centroid_labels = [str(t) for t in sorted(face_grids.keys())]
pl.add_point_labels(
    centroid_pts,
    centroid_labels,
    font_size=14,
    point_size=1,
    bold=True,
    text_color="black",
    shape_color="white",
    shape_opacity=0.6,
    always_visible=True,
)

pl.camera_position = "iso"

# ── App state ─────────────────────────────────────────────────────────────────
state.selected_surf   = None
state.included        = []
state.groups          = {}
state.group_input     = ""
state.existing_groups = []
state.sel_surf_text   = "—"
state.sel_centroid    = "—"
state.sel_area        = "—"
state.sel_status      = "—"
state.group_summary   = []
state.status_msg      = "Click a surface chip to select it"
state.has_selection   = False
state.has_groups      = False
state.surf_chips      = []   # [{tag, label, color_hex, text_color}]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_group(surf_tag: int) -> str | None:
    for name, tags in state.groups.items():
        if surf_tag in tags:
            return name
    return None


def _face_color_rgb(surf_tag: int) -> tuple:
    group_names = list(state.groups.keys())
    if surf_tag == state.selected_surf:
        return C_SELECTED
    if (grp := _find_group(surf_tag)) is not None:
        return group_color_rgb(group_names.index(grp))
    if surf_tag in state.included:
        return C_INCLUDED
    return C_EXCLUDED


def _build_chips() -> list:
    chips = []
    for tag in sorted(face_grids.keys()):
        rgb = _face_color_rgb(tag)
        hex_color = rgb_to_hex(*rgb)
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        text_color = "#000000" if luminance > 0.55 else "#ffffff"
        status = ""
        if tag == state.selected_surf:
            status = " ★"
        elif (grp := _find_group(tag)):
            status = f" [{grp}]"
        elif tag in state.included:
            status = " ✓"
        chips.append({
            "tag": tag, "label": f"Surf {tag}{status}",
            "color": hex_color, "text_color": text_color,
        })
    return chips


def _refresh_colors():
    for surf_tag, actor in actors.items():
        actor.GetProperty().SetColor(*_face_color_rgb(surf_tag))
    state.surf_chips = _build_chips()
    ctrl.view_update()


def _sync_group_state():
    state.existing_groups = list(state.groups.keys())
    state.group_summary   = [
        f"{name}  →  surf {', '.join(str(t) for t in tags)}"
        for name, tags in state.groups.items()
    ]
    state.has_groups = bool(state.groups)


def _update_sel_display(surf_tag: int):
    info = face_info.get(surf_tag, {})
    grp  = _find_group(surf_tag)
    if grp:
        status = f"group: {grp}"
    elif surf_tag in state.included:
        status = "included (unassigned)"
    else:
        status = "excluded"
    state.sel_surf_text = f"surf {surf_tag}"
    state.sel_centroid  = info.get("centroid", "—")
    state.sel_area      = str(info.get("area", "—"))
    state.sel_status    = status
    state.has_selection = True

# ── Surface chip click handler ────────────────────────────────────────────────
def select_surf(tag, **kwargs):
    surf_tag = int(tag)
    state.selected_surf = surf_tag
    _update_sel_display(surf_tag)
    _refresh_colors()

# ── Button handlers ───────────────────────────────────────────────────────────
def include_face():
    tag = state.selected_surf
    if tag is None:
        return
    if tag not in state.included:
        state.included = state.included + [tag]
    _update_sel_display(tag)
    _refresh_colors()


def exclude_face():
    tag = state.selected_surf
    if tag is None:
            return
    state.included = [t for t in state.included if t != tag]
    state.groups = {
        name: [t for t in tags if t != tag]
        for name, tags in state.groups.items()
    }
    state.groups = {k: v for k, v in state.groups.items() if v}
    _sync_group_state()
    _update_sel_display(tag)
    _refresh_colors()


def assign_group():
    tag  = state.selected_surf
    name = (state.group_input or "").strip()
    if tag is None or not name:
        return
    if tag not in state.included:
        state.included = state.included + [tag]
    new_groups = {k: [t for t in v if t != tag] for k, v in state.groups.items()}
    new_groups = {k: v for k, v in new_groups.items() if v}
    new_groups.setdefault(name, [])
    new_groups[name] = new_groups[name] + [tag]
    state.groups = new_groups
    _sync_group_state()
    _update_sel_display(tag)
    _refresh_colors()


def export_json():
    data = {
        "included": list(state.included),
        "groups":   {name: list(tags) for name, tags in state.groups.items()},
    }
    with open(str(JSON_PATH), "w") as f:
        json.dump(data, f, indent=2)
    state.status_msg = f"Saved → {JSON_PATH.name}"
    print(f"\nExported: {JSON_PATH}")
    print(json.dumps(data, indent=2))


def run_mesh_action():
    if not state.groups:
        state.status_msg = "Assign at least one group before meshing."
        return
    sys.path.insert(0, str(Path(__file__).parent))
    from step1_tag_mesh import run_mesh
    data = {
        "included": list(state.included),
        "groups":   {name: list(tags) for name, tags in state.groups.items()},
    }
    mesh_file = str(STEP_PATH.with_suffix(".msh"))
    state.status_msg = "Meshing…"
    try:
        run_mesh(str(STEP_PATH), data, mesh_file)
        state.status_msg = f"Mesh written → {STEP_PATH.stem}.msh"
    except Exception as e:
        state.status_msg = f"Error: {e}"

# ── Pre-populate chips (no view_update yet — view isn't built) ────────────────
state.surf_chips = _build_chips()

# ── trame layout ──────────────────────────────────────────────────────────────
PANEL_BG = "background:#fafafa; border-left:1px solid #e0e0e0; height:100%; overflow-y:auto"

with SinglePageLayout(server) as layout:
    layout.title.set_text("Face Labeler")
    layout.icon.hide()

    with layout.toolbar:
        v.VSpacer()
        html.Span(
            f"{STEP_PATH.name}  ·  {len(surf_tags)} surfaces",
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

                    # Surface chips
                    html.P("Surfaces  (click to select)",
                           style="font-weight:600; font-size:0.85em; margin-bottom:6px")
                    with html.Div(style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px"):
                        v.VChip(
                            "{{ chip.label }}",
                            v_for="chip in surf_chips",
                            key=("chip.tag",),
                            style=("'background:' + chip.color + '; color:' + chip.text_color + "
                                   "'; cursor:pointer; font-size:0.78em'",),
                            size="small",
                            click=(select_surf, "[chip.tag]"),
                        )

                    v.VDivider(classes="mb-3")

                    # Selected face info
                    html.P("Selected face",
                           style="font-weight:600; font-size:0.85em; margin-bottom:6px")
                    with v.VCard(variant="outlined", classes="mb-2"):
                        with v.VCardText(classes="pa-2"):
                            with html.Div(v_if="has_selection"):
                                html.P("{{ sel_surf_text }}",
                                       style="font-weight:bold; margin:0 0 4px 0; font-size:0.9em")
                                html.P("Centroid: {{ sel_centroid }}",
                                       style="font-size:0.78em; margin:2px 0; color:#444")
                                html.P("Area: {{ sel_area }}",
                                       style="font-size:0.78em; margin:2px 0; color:#444")
                                html.P("Status: {{ sel_status }}",
                                       style="font-size:0.78em; margin:2px 0; color:#444; font-style:italic")
                            with html.Div(v_if="!has_selection"):
                                html.P("Click a surface chip above",
                                       style="color:#aaa; font-size:0.82em; margin:0; font-style:italic")

                    # Include / Exclude
                    with v.VRow(no_gutters=True, classes="mb-3"):
                        with v.VCol(cols=6, classes="pr-1"):
                            v.VBtn("Include", block=True, color="primary",
                                   variant="outlined", size="small",
                                   disabled=("!has_selection",),
                                   click=include_face)
                        with v.VCol(cols=6, classes="pl-1"):
                            v.VBtn("Exclude", block=True, color="error",
                                   variant="outlined", size="small",
                                   disabled=("!has_selection",),
                                   click=exclude_face)

                    v.VDivider(classes="mb-3")

                    # Group assignment
                    html.P("Assign to group",
                           style="font-weight:600; font-size:0.85em; margin-bottom:6px")
                    v.VCombobox(
                        v_model=("group_input", ""),
                        label="Group name",
                        items=("existing_groups", []),
                        density="compact",
                        hide_details=True,
                        classes="mb-2",
                    )
                    v.VBtn("Assign to group", block=True, color="success",
                           variant="tonal", size="small", classes="mb-3",
                           disabled=("!has_selection || !group_input",),
                           click=assign_group)

                    v.VDivider(classes="mb-3")

                    # Group summary
                    html.P("Groups",
                           style="font-weight:600; font-size:0.85em; margin-bottom:4px")
                    with html.Div(v_if="!has_groups"):
                        html.P("No groups assigned yet",
                               style="color:#aaa; font-size:0.82em; font-style:italic")
                    with html.Div(v_if="has_groups"):
                        with html.Div(
                            v_for="line in group_summary",
                            key="line",
                            style="font-size:0.80em; padding:2px 0; font-family:monospace; color:#333",
                        ):
                            html.Span("● {{ line }}")

                    v.VDivider(classes="my-3")

                    # Export / Mesh
                    v.VBtn("Export JSON", block=True, color="primary", variant="tonal",
                           classes="mb-2",
                           disabled=("!has_groups",),
                           click=export_json)
                    v.VBtn("Run Mesh", block=True, color="secondary", variant="tonal",
                           classes="mb-2",
                           disabled=("!has_groups",),
                           click=run_mesh_action)

                    html.P("{{ status_msg }}",
                           style="font-size:0.78em; color:#555; margin-top:6px; min-height:1.2em")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server.start(open_browser=True, port=8080)
