#!/bin/bash
cd /home/h621l/projects/science-article
.venv/bin/python3 -c "import cartopy; print('cartopy:', cartopy.__file__)"
echo "---"
.venv/bin/python3 -c "import matplotlib; print('matplotlib:', matplotlib.__file__)"
