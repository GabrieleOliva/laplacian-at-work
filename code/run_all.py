"""Regenerate every CSV file consumed by the TikZ figures of the paper."""
import json
import os

import fig1_examples
import fig2_consensus
import fig3_drawing
import fig4_clustering
import fig5_ahp
from common import DATA

if __name__ == "__main__":
    p = os.path.join(DATA, "results.json")
    if os.path.exists(p):
        os.remove(p)
    for name, mod in [("Figure 1  the four examples", fig1_examples),
                      ("Figure 2  consensus (sensor network)", fig2_consensus),
                      ("Figure 3  drawing (Delaunay)", fig3_drawing),
                      ("Figure 4  clustering (haemoglobin)", fig4_clustering),
                      ("Figure 5  sparse AHP (small world)", fig5_ahp)]:
        print(f"\n=== {name} " + "=" * (52 - len(name)))
        mod.main()
    with open(p) as f:
        print("\n=== results.json " + "=" * 44)
        print(json.dumps(json.load(f), indent=2, sort_keys=True))
