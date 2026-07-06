# Meshifier

CadQuery → Gmsh → PyVista meshing pipeline with a browser-based GUI for labeling faces, edges, and vertices into named physical groups for FEM analysis.

---

## What it does

1. **Open** any STEP file in a browser-based 3D viewer
2. **Label** surfaces → named PSHELL regions, edges → bar elements, vertices → point masses
3. **Mesh** with Gmsh (quad-dominant, structured, or tri per surface)
4. **View** the resulting mesh in the same browser window
5. **Export** labels to JSON for round-trip editing; mesh to `.msh` for downstream tools

---

## Setup — Linux (Fedora / Ubuntu)

### 1. Install Miniforge or Miniconda

```bash
# Miniforge (recommended — uses conda-forge by default)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### 2. Create the conda environment

```bash
conda create -n meshcad python=3.11 -y
conda activate meshcad
```

### 3. Install system OpenGL library (required by Gmsh)

```bash
# Gmsh's pip wheel links against libGLU.so.1 — install via conda-forge
conda install -c conda-forge libglu -y
```

> **Why conda-forge?** The pip Gmsh wheel bundles its own `libgmsh.so` but needs `libGLU.so.1` at runtime. On minimal/headless installs this library is absent. Installing via conda-forge puts it on `LD_LIBRARY_PATH` automatically when using `conda run`.

### 4. Install Python packages

```bash
pip install cadquery==2.8.0
pip install gmsh==4.15.2
pip install meshio==5.3.5
pip install pyvista==0.48.4
pip install trame==3.13.2 trame-vtk trame-vuetify
pip install pyNastran==1.4.1
pip install panel
```

> **Note:** `pyNastran 1.4.1` downgrades NumPy from 2.x to `<2`. Check that CadQuery still imports after this step (`python -c "import cadquery"`).

### 5. Verify

```bash
conda run -n meshcad python step1_tag_mesh.py
# Expected: "PASS: all named surfaces have mesh elements."
```

---

## Setup — Windows

### 1. Install Miniforge or Miniconda

Download and run the Windows installer from:
https://github.com/conda-forge/miniforge/releases/latest

Open **Miniforge Prompt** (or Anaconda Prompt) for all subsequent steps.

### 2. Create the conda environment

```bat
conda create -n meshcad python=3.11 -y
conda activate meshcad
```

### 3. Install Python packages

On Windows, `libGLU` ships with the graphics driver — no extra conda package needed.

```bat
pip install cadquery==2.8.0
pip install gmsh==4.15.2
pip install meshio==5.3.5
pip install pyvista==0.48.4
pip install trame==3.13.2 trame-vtk trame-vuetify
pip install pyNastran==1.4.1
pip install panel
```

### 4. Verify

```bat
conda run -n meshcad python step1_tag_mesh.py
```

---

## Running the app

```bash
conda run -n meshcad python face_labeler.py part.step
```

Open `http://localhost:8080` in your browser. The app loads the STEP file and auto-restores any saved labels from `<stem>_labels.json` if it exists alongside the STEP file.

### Workflow

| Step | Action |
|------|--------|
| **Surfaces tab** | Click a chip to select a surface. Click **Include**, type a group name, click **Assign to group**. Use the Quad / Struct / Tri toggle to set mesh type per surface. |
| **Edges tab** | Select an edge. Chips marked `(bar)` are free bar elements (not bounding any surface). Include and assign them to groups for 1D bar elements. |
| **Vertices tab** | Select a vertex, include it, assign to a group for 0D point mass elements. |
| **Export JSON** | Saves `<stem>_labels.json` alongside the STEP file for round-trip reload. |
| **Run Mesh** | Generates `<stem>.msh`. The view switches to the mesh. Click **Edit Labels** to go back. |

---

## Standalone meshing script

```bash
# Demo: builds L-plate + portal frame bars, matches by centroid, meshes
conda run -n meshcad python step1_tag_mesh.py

# From Python — call run_mesh() directly with a labels dict
from step1_tag_mesh import run_mesh
run_mesh("part.step", face_labels_dict, "part.msh", mesh_size_max=3.0)
```

---

## File overview

| File | Purpose |
|------|---------|
| `face_labeler.py` | Trame browser app — label, mesh, view |
| `step1_tag_mesh.py` | Core meshing function + standalone demo |
| `test_0d1d.py` | Pipeline test for 0D/1D/2D elements |
| `ocp_vscode_demo.py` | OCP-VSCode notebook cells for geometry preview |
| `GOTCHAS.md` | Pitfalls encountered during development |

---

## Known issues / requirements

- **Python 3.11** — no conda-forge OSMesa VTK build exists for 3.12+ as of mid-2026; stick to 3.11
- **`conda run -n meshcad`** — always run scripts via `conda run`, not a raw Python path, so `LD_LIBRARY_PATH` includes the conda env libs (Gmsh requires this on Linux)
- **Headless servers** — `export_html()` works headless; interactive trame requires a browser connection but no display on the server side
- See `GOTCHAS.md` for a full list of installation and API pitfalls
