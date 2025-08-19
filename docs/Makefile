# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Additional custom targets for EvoAug2 documentation

# Build HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html/"

# Build PDF documentation (requires LaTeX)
pdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Build finished. The LaTeX files are in $(BUILDDIR)/latex/"

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*

# Install documentation dependencies
install-deps:
	pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

# Auto-generate API documentation
api:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "API documentation generated in $(BUILDDIR)/html/"

# Serve documentation locally (requires Python 3.7+)
serve:
	cd $(BUILDDIR)/html && python -m http.server 8000
	@echo "Documentation served at http://localhost:8000"

# Check for broken links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check finished. Look for any errors in the above output."

# Build and open in browser (macOS)
open: html
	open $(BUILDDIR)/html/index.html

# Build and open in browser (Linux)
open-linux: html
	xdg-open $(BUILDDIR)/html/index.html

# Build and open in browser (Windows)
open-windows: html
	start $(BUILDDIR)/html/index.html

# Quick build for development
dev: html
	@echo "Quick build completed. Open $(BUILDDIR)/html/index.html in your browser."

# Full build with all formats
all: clean html pdf
	@echo "Full build completed with HTML and PDF formats." 