# mx2-multi-messenger

Spatially-resolved spectroscopy (reflection, Raman, SHG) of Yuzhou's MoS2-WS2 lateral heterostructure.


## build instructions
- This folder stores all raw data and the manuscript.  The dData workflow must be run locally.
- all processing steps are included in the `build.py` script.  Run this script to compute the full data workflow from composing raw data to post-processing to figure generation.
  - if no arguments are given, build.py will perform all workflow steps
  - arguments `data`, `fits`, `figures` can be used to specify individual steps to run
- This build was developed on Windows OS, but _should_ be robust to Linux and iOS operations.

## folder structure

### data folder


### figures folder
- `figures.py` generates all figures.  
  - Make sure to set `save=True` to generate pngs
- Individual scripts are made for each figure.  You can run these scripts to generate specific figures. 
  - e.g. on Windows: `python raman_si.py`
  - For interactive figures (e.g. pyqt), run file with argument 0:
    `python raman_si.py 0`

### fitting folder

