"""The three simulated topologies, drawn from the edge lists in aeon/topology.py (link lengths in km)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aeon.topology import _nsfnet_edges, _usnet_edges, _european_edges
out = sys.argv[1]
INK, EDGE, NODE = "#1f1f1e", "#8a897f", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 600})
# NSFNET: approximate geographic placement of the standard 14-node network
NSF_POS = {0: (0.0, 3.0), 1: (0.6, 2.0), 2: (0.2, 0.6), 3: (1.8, 2.3), 4: (2.7, 1.8), 5: (3.0, 0.4),
           6: (3.8, 2.1), 7: (5.0, 2.4), 8: (6.3, 2.0), 9: (5.6, 0.3), 10: (5.6, 3.3), 11: (7.2, 2.9),
           12: (7.5, 2.2), 13: (7.0, 1.3)}
# PAN-EUROPEAN: fixed node positions for drawing (same edge list as aeon/topology.py)
_EU = {0:(1270,248),1:(1300,67),2:(1315,197),3:(1350,25),4:(1350,107),5:(1340,253),6:(1370,140),7:(1383,205),
       8:(1403,122),9:(1405,97),10:(1410,187),11:(1421,158),12:(1442,95),13:(1447,139),14:(1437,207),15:(1469,169),
       16:(1484,102),17:(1492,58),18:(1472,9),19:(1474,251),20:(1506,178),21:(1527,153),22:(1515,213),23:(1531,24),
       24:(1549,183),25:(1561,103),26:(1570,218),27:(1580,285)}
EU_POS = {k: (x, -y) for k, (x, y) in _EU.items()}
def us_pos(edges):
    G = nx.Graph(); G.add_weighted_edges_from(edges, weight="km")
    ok, _ = nx.check_planarity(G)
    p0 = nx.planar_layout(G) if ok else None
    return nx.kamada_kawai_layout(G, pos=p0, weight=None)
panels = [("(a) NSFNET (14 nodes, 21 links)", _nsfnet_edges(), NSF_POS),
          ("(b) USNET (24 nodes, 43 links)", _usnet_edges(), us_pos(_usnet_edges())),
          ("(c) PAN-EUROPEAN (28 nodes, 41 links)", _european_edges(), EU_POS)]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1.15, 1, 1]})
for ax, (title, edges, pos) in zip(axs, panels):
    G = nx.Graph(); G.add_weighted_edges_from(edges, weight="km")
    if pos is None:
        pos = nx.kamada_kawai_layout(G, weight="km")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="white", edgecolors=NODE, linewidths=1.0, node_size=110)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: str(n) for n in G}, font_size=5.2, font_color=INK)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={(u, v): d["km"] for u, v, d in G.edges(data=True)},
                                 font_size=4.0, font_color="#5a5952", rotate=False,
                                 bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.8))
    ax.set_title(title, fontsize=7, color=INK); ax.axis("off")
fig.tight_layout(w_pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig1_topologies.{ext}"), bbox_inches="tight")
print("ok")
