# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html loose clean linkcheck serve

help:
	@echo "html       build the site into $(BUILDDIR)/html (warnings are errors)"
	@echo "loose      build without promoting warnings to errors"
	@echo "serve      build, then serve on http://localhost:8000"
	@echo "linkcheck  verify external links"
	@echo "clean      remove $(BUILDDIR)"

# Warnings are errors. The headers are clean, and the point of a docs build
# that fails is to keep them that way -- see docs/api/index.md for the four
# rules a comment has to follow.
html:
	$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Same build without the -W, for finding out how bad it is before fixing it.
loose:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

serve: html
	@cd "$(BUILDDIR)/html" && python3 -m http.server 8000

clean:
	rm -rf "$(BUILDDIR)"
