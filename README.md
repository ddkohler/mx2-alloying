# mx2-alloying

This repository hosts the code used for composing the manuscript 
"Spatially-resolved spectroscopy of MoS2-WS2 lateral heterostructures" (working title).

## build instructions
- You will need the following packages
  - WrightTools >= 4.3.6
  - osfclient == 0.0.5
- all processing steps are included in the `build.py` script.  Run this script e.g. on Windows:
  ```
  python build.py <args>
  ```
  args [optional]
  - `fetch`: download and extract the [raw data](https://osf.io/6gxsn)
  - `data`: perform all data processing steps and store as [wt5](http://wright.tools/en/stable/wt5.html) files
  - `figures`: generate manuscript figures
  - if no arguments are given, build.py will perform all workflow steps (in order)
- This workflow was developed on Windows OS, but _should_ be robust to Linux and iOS.

## folder structure

### data folder

### figures folder
- `figures.py` generates all figures.  
  - Make sure to set `save=True` to generate pngs
- Individual scripts are made for each figure.  You can run these scripts to generate specific figures. 
  - e.g. on Windows: `python raman_si.py`
  - For interactive figures (e.g. pyqt), run file with argument 0:
    `python raman_si.py 0`

