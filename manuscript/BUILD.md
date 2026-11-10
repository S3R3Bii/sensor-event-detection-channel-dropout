# Build the technical manuscript

This directory contains the complete TeX source, BibTeX bibliography, and four ready-to-use figure images. Build from this directory; no model fitting or figure regeneration is required.

With Tectonic 0.15.0:

~~~sh
tectonic --reruns 2 --keep-logs main.tex
~~~

Or with TeX Live and latexmk:

~~~sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

The PDF appears as `main.pdf`. The checked output is `Technical_Manuscript.pdf`. Tectonic may download its standard TeX bundle on first use. To regenerate the four figure images from the original scores, run `python analysis/make_figures.py` from the source package root after installing the pinned dependencies. It uses saved scores, the frozen config, and the deterministic generator to reconstruct two traces. No training or threshold selection occurs.
