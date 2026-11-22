# Configuration file for the Sphinx documentation of molib.
# Build locally with:  python -m sphinx -b html docs docs/_build/html
import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "molib"
author = "Stephan Helfrich"
copyright = "2024-2026, Stephan Helfrich"
try:
    release = package_version("molib")
except PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints", "auto_examples/*.ipynb", "auto_examples/*.py",
                    "auto_examples/*.zip", "auto_examples/*.json", "auto_examples/*.md5"]

# MyST (Markdown) with $...$ and $$...$$ math
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# autodoc / numpydoc
autodoc_mock_imports = ["gurobipy"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
numpydoc_show_class_members = False     # avoids duplicate member tables with autodoc
numpydoc_validation_checks = set()      # no docstring validation for now
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# sphinx-gallery: examples/plot_*.py are executed during the build
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
}

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "molib"
html_theme_options = {
    "github_url": "https://github.com/StephanHelfrich/molib",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}