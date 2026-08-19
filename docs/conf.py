"""Sphinx configuration for the CLensPy documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import clenspy  # noqa: E402

# -- Project information ----------------------------------------------------

project = "CLensPy"
copyright = "2026, Johnny H. Esteves"
author = "Johnny H. Esteves"
version = clenspy.__version__
release = clenspy.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_wagtail_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Reference material kept locally under docs/, not part of the site.
    "Slice_*.md",
]

# Autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon (the codebase uses NumPy-style docstrings throughout)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_wagtail_theme"

html_theme_options = dict(
    project_name="CLensPy",
    logo="img/logo.png",
    logo_alt="CLensPy - cluster gravitational lensing",
    logo_height=48,
    logo_url="/",
    logo_width=48,
    github_url="https://github.com/estevesjh/clenspy/blob/main/docs/",
    footer_links=",".join(
        [
            "Source|https://github.com/estevesjh/clenspy",
            "Issues|https://github.com/estevesjh/clenspy/issues",
            "PyPI|https://pypi.org/project/clenspy/",
        ]
    ),
)

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/img/logo.png"

html_show_sphinx = False
