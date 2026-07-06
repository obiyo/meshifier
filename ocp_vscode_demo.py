# %%

# The markers "# %%" separate code blocks for execution (cells) 
# Press shift-enter to exectute a cell and move to next cell
# Press ctrl-enter to exectute a cell and keep cursor at the position
# For more details, see https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter

# %%

import cadquery as cq
from ocp_vscode import *

# %%

b = cq.Workplane().box(1,2,3).fillet(0.1)

show(b)

# %%

# L-bracket plate with standalone portal frame bar members
# The bars are free edges (not bounding any surface) — they become 1D bar elements.

pts   = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
plate = cq.Workplane("XY").polyline(pts).close().extrude(5)

bars = [
    cq.Edge.makeLine(cq.Vector( 0,  0, 5), cq.Vector( 0,  0, 12)),  # post
    cq.Edge.makeLine(cq.Vector(20,  0, 5), cq.Vector(20,  0, 12)),  # post
    cq.Edge.makeLine(cq.Vector( 0, 20, 5), cq.Vector( 0, 20, 12)),  # post
    cq.Edge.makeLine(cq.Vector( 0,  0, 12), cq.Vector(20,  0, 12)), # top beam
    cq.Edge.makeLine(cq.Vector( 0,  0, 12), cq.Vector( 0, 20, 12)), # top beam
]

show_object(plate, name="plate", options={"alpha": 0.7})
for i, bar in enumerate(bars):
    label = "post" if i < 3 else "beam"
    show_object(bar, name=f"{label}_{i+1}", options={"color": (255, 128, 0), "linewidth": 3})

# %%
