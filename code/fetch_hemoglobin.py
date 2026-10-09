"""Build the haemoglobin protein contact network used in Section 5.

Downloads the crystal structure of human deoxyhaemoglobin (PDB 2HHB) from
RCSB and writes the C-alpha contact network to assets/, following the standard
protein-contact-network convention of Di Paola et al. (Chem. Rev. 2013): two
residues are in contact when their C-alpha atoms lie between 4 and 8 angstrom
apart.

The derived files are committed to the repository, so this script only needs to
be run if you want to rebuild them or change the cut-off; the figures never
download anything.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from common import ASSETS

PDB_ID = "2HHB"
URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
D_MIN, D_MAX = 4.0, 8.0          # angstrom


def parse_ca(text):
    """(chain, residue number, x, y, z) for every C-alpha, altloc A or blank."""
    out = []
    for line in text.splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA " and line[16] in " A":
            out.append((line[21], int(line[22:26]),
                        float(line[30:38]), float(line[38:46]),
                        float(line[46:54])))
    return out


def main():
    os.makedirs(ASSETS, exist_ok=True)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as fh:
        text = fh.read().decode()

    ca = parse_ca(text)
    X = np.array([[x, y, z] for _, _, x, y, z in ca])
    n = len(ca)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    contact = (D >= D_MIN) & (D <= D_MAX)
    np.fill_diagonal(contact, False)

    with open(os.path.join(ASSETS, "hemoglobin_nodes.csv"), "w") as f:
        f.write("id,chain,resid,x,y,z\n")
        for i, (ch, rn, x, y, z) in enumerate(ca):
            f.write(f"{i},{ch},{rn},{x:.3f},{y:.3f},{z:.3f}\n")

    ii, jj = np.nonzero(np.triu(contact))
    with open(os.path.join(ASSETS, "hemoglobin_edges.csv"), "w") as f:
        f.write("i,j\n")
        for a, b in zip(ii, jj):
            f.write(f"{a},{b}\n")

    print(f"  {PDB_ID}: {n} residues, {len(ii)} contacts "
          f"({D_MIN}-{D_MAX} A), chains "
          f"{sorted(set(c for c, *_ in ca))}")


if __name__ == "__main__":
    main()
