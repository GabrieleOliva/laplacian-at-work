# Companion repository of "The Graph Laplacian at Work".
PY ?= python3
JUPYTER ?= jupyter

.PHONY: all data figures exercises notebooks clean

all: data exercises

# regenerate every CSV file and data/results.json
data:
	$(PY) code/run_all.py

# the five figures of the paper, drawn from data/ (needs LaTeX with pgfplots)
figures:
	cd figures && latexmk -pdf -interaction=nonstopmode figures.tex \
	    && latexmk -c figures.tex

# the exercise sheet (problems only)
exercises:
	cd exercises && latexmk -pdf -interaction=nonstopmode exercises.tex \
	    && latexmk -c exercises.tex

# execute every notebook from top to bottom, without saving the outputs
notebooks:
	@for nb in notebooks/*.ipynb; do \
	    echo "running $$nb"; \
	    $(JUPYTER) nbconvert --to notebook --execute --stdout "$$nb" > /dev/null || exit 1; \
	done

clean:
	rm -rf code/__pycache__ notebooks/.ipynb_checkpoints
	cd exercises && latexmk -c exercises.tex
	cd figures && latexmk -c figures.tex
