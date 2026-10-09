# The Graph Laplacian at Work

Companion repository of the tutorial paper

> G. Oliva, *The Graph Laplacian at Work: One Least-Squares Problem Behind
> Consensus, Graph Drawing, Clustering and Decision Making*, SICE Journal of
> Control, Measurement, and System Integration (submitted).

It contains the code that generates every number and every figure of the
paper, one Jupyter notebook per section, and a set of exercises. It is
self-contained: nothing outside this folder is needed.

## The idea in one line

Four applications of the graph Laplacian that are usually presented in
isolation — distributed average consensus, Laplacian graph drawing, spectral
clustering, and ranking from sparse pairwise comparisons — are governed by a
single least-squares problem

```
minimize  || Bᵀ x − b ||²_W       over x ∈ Rⁿ,      normal equations  L x = B W b
```

where `B` is the incidence matrix of the graph, the unknown `x` lives on the
nodes, and the data `b` live on the edges. With an affine constraint (none, or
anchored nodes) consensus, Tutte's drawing and sparse AHP share the whole
problem and the same distributed iteration; with a normalization, Hall's
layout and spectral clustering share its quadratic form and become eigenvalue
problems.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebooks.txt
jupyter lab notebooks/
```

Open `00_one_problem.ipynb` and run the cells from top to bottom
(Shift+Enter). In VS Code, open this folder, open a notebook, and select the
`.venv` interpreter as the kernel.

## Notebooks

One per section of the paper, to be read alongside it. They import the
functions in `code/`, so they must be run from inside this repository.

| notebook | paper | what it does |
|---|---|---|
| `00_one_problem` | Sec. 2 | Example 2.4 by hand; one solver for consensus, Tutte, sparse AHP and formation control |
| `01_consensus` | Sec. 3 | continuous and discrete consensus on a sensor network; step size and spectral ratio |
| `02_drawing` | Sec. 4 | Hall, degree-normalized and Tutte layouts of a Delaunay mesh; how many leaders |
| `03_clustering` | Sec. 5 | spectral clustering of the haemoglobin contact network; eigengap |
| `04_sparse_ahp` | Sec. 6 | sparse logarithmic least squares, centralized and distributed |

## Exercises

`exercises/exercises.pdf`: ten exercises, from proofs to small numerical
experiments, keyed to the sections of the paper. Exercises marked **[code]**
build on the notebooks. Solutions are available to instructors from the
author.

## The four examples

| Section | graph | n | m | λ₂(L) | why this one |
|---|---|---|---|---|---|
| 3 · consensus | random geometric | 50 | 160 | 0.181 | the standard sensor-network model |
| 4 · graph drawing | Delaunay triangulation | 30 | 77 | 0.651 | planar **and** 3-connected: Tutte's hypotheses hold |
| 5 · clustering | haemoglobin contacts (PDB 2HHB) | 574 | 2251 | 0.098 | a tetramer, so the right answer is known in advance |
| 6 · sparse AHP | Watts–Strogatz small world | 40 | 80 | 0.522 | a sparse elicitation pattern |

* On the **Delaunay** mesh the anchored (Tutte) embedding is provably
  crossing-free, and the code checks that it is — while Hall's layout of the
  *same planar graph* has four crossings: minimizing an energy promises
  nothing about planarity.
* On **haemoglobin** the eigengap heuristic returns k\* = 4 on its own, and the
  resulting clusters agree with the four chains on 546 of 574 residues
  (95.1 %). The algorithm is given a list of 2251 contacts and returns the
  quaternary structure.

## What is in here

```
.
├── code/                    one Python script per figure of the paper
│   ├── common.py            the four graphs, Laplacian, incidence, IO helpers
│   ├── fetch_hemoglobin.py  rebuilds the contact network from RCSB (rarely needed)
│   ├── fig1_examples.py     the four examples and their spectra
│   ├── fig2_consensus.py    continuous-time average consensus
│   ├── fig3_drawing.py      Hall, degree-normalized Hall, anchored (Tutte)
│   ├── fig4_clustering.py   normalized cut, eigengap, k-means
│   ├── fig5_ahp.py          sparse AHP via Sparse Logarithmic Least Squares
│   └── run_all.py           regenerates every CSV and results.json
├── assets/                  the haemoglobin contact network (derived from PDB 2HHB)
├── data/                    generated CSV files + results.json
├── figures/                 TikZ/pgfplots sources of the paper's figures
│   └── pgfstyles.tex        shared styles and colour palette
├── notebooks/               one notebook per section
└── exercises/               exercises.tex and exercises.pdf
```

No figure of the paper is a bitmap and none is produced by Python: the
scripts only write CSV files, and every figure is drawn by pgfplots from
those files when the paper is compiled. `data/results.json` collects every
number quoted in the paper — algebraic connectivities, step sizes,
normalized cuts, chain agreements, Kendall correlations — so the manuscript
can be checked against the code.

Three of the four graphs are generated from a fixed seed. The fourth, the
haemoglobin contact network, is committed under `assets/` as a derived edge
list; `code/fetch_hemoglobin.py` rebuilds it from the deposited structure, so
nothing here needs a network connection.

## Make targets

```bash
make data        # regenerate data/*.csv and data/results.json
make exercises   # rebuild exercises/exercises.pdf (needs LaTeX)
make notebooks   # run every notebook headless, as a test (needs Jupyter)
```

The only dependencies of `code/` are `numpy`, `scipy` and `networkx`
(`requirements.txt`); the notebooks also need `matplotlib` and Jupyter
(`requirements-notebooks.txt`).

## License

Code released under the MIT license (see `LICENSE`). The structure 2HHB is
from the RCSB Protein Data Bank.
