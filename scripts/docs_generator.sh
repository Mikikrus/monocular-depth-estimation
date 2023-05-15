#!/bin/bash

sphinx-quickstart docs -q -p "Monocular Depth Estimation" -a "Maciej Filanowicz & Mikołaj Kruś" --sep  --ext-autodoc --ext-doctest --ext-githubpages

sphinx-apidoc -f -o docs/source .

python scripts/replace_string.py --path docs/source/conf.py --old "html_theme = 'alabaster'" --new "html_theme = 'sphinx_rtd_theme'"
python scripts/replace_string.py --path docs/source/conf.py --old "exclude_patterns = []" --new "exclude_patterns = ['venv', 'data','*.egg-info', 'build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']"
python scripts/replace_string.py --path docs/source/index.rst --old ".. toctree::\n   :maxdepth: 2\n   :caption: Contents:" --new ".. toctree::\n   :maxdepth: 2\n   :caption: Contents:\n\n   modules"

echo "import os" >> docs/source/conf.py
echo "import sys" >> docs/source/conf.py
echo "sys.path.insert(0, os.path.abspath('../..'))" >> docs/source/conf.py

# Step 5: Generate the HTML files
sphinx-build -b html docs/source docs/build

