# Gotchas — CadQuery → Gmsh → PyVista pipeline

Environment: Fedora Linux, conda env `meshcad`, Python 3.11, CadQuery 2.8.0 / OCP 7.9.3.1 / Gmsh 4.15.2 / PyVista 0.48.4 / pyNastran 1.4.1.

---

## Installation

### 1. pip gmsh wheel needs `libGLU.so.1` at load time
The `manylinux_2_24` pip wheel bundles its own `libgmsh.so`, but that SO links against the system `libGLU.so.1` (the Mesa OpenGL Utility library). On a minimal or headless Fedora install it is absent.

**Fix:** `conda install -c conda-forge libglu` — adds `libGLU.so.1` to the conda env's lib directory, which `conda run` puts on `LD_LIBRARY_PATH` automatically.

---

### 2. conda-forge `gmsh` package ≠ Python module
`conda install -c conda-forge gmsh` installs `libgmsh.so`, headers, and CMake files — **not** `gmsh.py`. The Python API only comes from the pip wheel.

**Fix:** Always install both: conda-forge gmsh (if you want the shared library) and pip gmsh (for `gmsh.py`). In practice, just pip gmsh + libglu (gotcha #1) is enough.

---

### 3. conda-forge gmsh's `libgmsh.so` depends on FLTK
If you install the conda-forge `gmsh` first (intending to use its `libgmsh.so`), pip's `gmsh.py` will find the conda-forge library via `LD_LIBRARY_PATH` and fail with `libfltk_images.so.1.3: cannot open shared object file`. The conda-forge build links FLTK for its GUI.

**Fix:** Either install conda-forge's `fltk` package to satisfy the dependency, or — simpler — remove the conda-forge gmsh and use only the pip wheel + libglu (gotcha #1).

---

### 4. pyNastran 1.4.1 downgrades NumPy
`pip install pyNastran` installs pyNastran 1.4.1, which pins `numpy < 2.x`. If your env already has NumPy 2.x it will be downgraded (2.4.6 → 1.26.4 in our case). Check that CadQuery and other packages still work after the downgrade before proceeding.

---

### 5. No conda-forge OSMesa VTK build for Python 3.11
The conda-forge `vtk` package has OSMesa (headless) builds, but only up to Python 3.9 / VTK 9.3.1 as of mid-2026. There is no `vtk=*=osmesa*` for Python 3.11.

**Consequence:** pip's VTK (used by PyVista) cannot do headless GPU rendering on a server without either installing the system `mesa-libOSMesa` RPM or using a display. See gotcha #8.

---

## CadQuery

### 6. `Face.Center()` returns `cq.Vector` with lowercase attributes
In CadQuery 2.x, `face.Center()` returns a `cq.Vector`. Access coordinates as `.x`, `.y`, `.z` (lowercase). Raw OCC `gp_Pnt` objects use methods like `.X()`, `.Y()`, `.Z()` — mixing these up gives `AttributeError`.

---

## Gmsh

### 7. `occ.getCenterOfMass()` works before meshing
`gmsh.model.occ.getCenterOfMass(dim, tag)` can be called immediately after `gmsh.model.occ.synchronize()`, before any mesh is generated. It operates on the CAD geometry, not mesh nodes.

---

### 8. Centroid matching survives STEP round-trip exactly
Gmsh's OCC kernel imports STEP geometry and preserves face centroids to floating-point precision. Matching CadQuery face centroids against Gmsh surface centroids after STEP export gives **zero distance** for planar faces. No tolerance fudging needed for simple geometry; a tolerance guard (e.g. 0.5 mm) is good practice for robustness.

---

### 9. Quad recipe: `Algorithm 8` + `RecombineAll 1` + explicit `recombine()`
```python
gmsh.option.setNumber("Mesh.Algorithm",    8)  # Frontal-Delaunay for quads (requires Gmsh ≥ 4.6)
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # auto-recombine all surfaces
gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()                    # explicit second clean-up pass
```
On flat rectangular patches this gives 100 % quads. The explicit `recombine()` call is redundant for planar faces but helps on curved or complex surfaces.

---

## PyVista

### 10. `pv.Plotter.export_html()` works headless despite VTK warnings
Even without a display, EGL, or OSMesa, `export_html()` succeeds. VTK prints warnings about failing to initialize OpenGL, but the method serializes mesh geometry to JSON/binary for **VTK.js** to render client-side in the browser — it never actually renders server-side. The warnings are harmless noise.

---

### 11. PyVista 0.48 API changes
- `pl.get_default_theme()` was removed — use `pv.global_theme` instead.
- Valid Jupyter backends: `"static"`, `"client"`, `"server"`, `"trame"`, `"html"`, `"none"`. The string `"panel"` is **not** valid and raises `ValueError`.
- For standalone browser export, use `pl.export_html("out.html")` directly — no Jupyter backend setting needed.

---

### 12. `export_html()` needs `panel` installed
Under the hood `pl.export_html()` imports `panel` to embed the VTK.js viewer. Without it you get `ModuleNotFoundError: No module named 'panel'`.

**Fix:** `pip install panel`

---

## 0D / 1D elements

### 13. `getCenterOfMass(0, tag)` returns (0, 0, 0) for all vertices

`gmsh.model.occ.getCenterOfMass(dim, tag)` works correctly for dim=1, 2, 3 but silently returns the origin for dim=0 (vertices) in the OCC kernel.

**Fix:** Use `gmsh.model.getBoundingBox(0, tag)` — for a vertex the bounding box min and max are identical and equal to the vertex coordinates:
```python
bb = gmsh.model.getBoundingBox(0, tag)
x, y, z = bb[0], bb[1], bb[2]
```

---

### 14. `generate(2)` creates 0D and 1D elements automatically

`gmsh.model.mesh.generate(2)` generates mesh for all dimensions ≤ 2, including point elements at vertices (dim=0) and line elements on curves (dim=1). No separate `generate(0)` or `generate(1)` calls are needed. Physical groups at each dimension control which entities are exported to the `.msh` file.

---

### 16. `importShapes` drops free edges unless `highestDimOnly=False`

`gmsh.model.occ.importShapes(file)` defaults to `highestDimOnly=True`, which imports only the highest-dimensional entities in the file. For a STEP compound containing a solid plus standalone wire/edge elements, this silently discards all the free edges — only the solid appears.

**Fix:** Always pass `highestDimOnly=False` when the STEP file may contain lower-dimensional entities:
```python
gmsh.model.occ.importShapes(step_file, highestDimOnly=False)
```
This has no downside for pure-solid files.

---

### 17. Detecting free bar edges (no bounding surface) via adjacency

After importing with `highestDimOnly=False`, use upward adjacency to distinguish free bar edges from solid-bounding edges:
```python
adj = gmsh.model.getAdjacencies(1, tag)  # returns (upward, downward)
is_free_bar = len(adj[0]) == 0           # adj[0] = bounding surfaces (empty for free bars)
```
Note: `adj[1]` is the downward adjacency (endpoint vertices, always 2 for a line), NOT the surfaces. Easy to confuse.

---

### 15. meshio cell type names for 0D / 1D elements

After reading a `.msh` file with meshio, lower-dimensional elements appear as:
- `"vertex"` — 1-node point element (Gmsh type 15, → Nastran CONM2/CELAS)
- `"line"` — 2-node bar element (Gmsh type 1, → Nastran CBAR/CBEAM)

Both are present in `mesh.cells` only when they belong to a named physical group.
