import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

project = "strained_apfc"
copyright = "2023, maromei"
author = "maromei"

######################
### GENERAL CONFIG ###
######################

extensions = [
    "myst_parser",  # use markdown
    "breathe",  # for doxygen doc
    "sphinx.ext.mathjax",  # display math in docstrings
    "sphinx.ext.napoleon",  # Numpy docstrings
    "sphinxcontrib.bibtex",  # Bibliography
    "sphinx.ext.todo",  # Todo
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = ["dollarmath", "amsmath"]

##############
### BIBTEX ###
##############

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "super"

#############
### TODOS ###
#############

todo_include_todos = True
todo_link_only = True

#####################
### BREATHE SETUP ###
#####################

breathe_xml_path = Path(__file__).parent.parent.parent
breathe_xml_path = breathe_xml_path / "doxygen/xml"

breathe_projects = {"strained_apfc": str(breathe_xml_path)}
breathe_default_project = "strained_apfc"

###################
### HTML OUTPUT ###
###################

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Strained APFC"
