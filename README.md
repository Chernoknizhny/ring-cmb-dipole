# Angular excess of galaxies at 25.5° from the CMB dipole

This repository contains the data analysis code, the input catalogues, and the manuscript for the paper:

**"Angular excess of galaxies at 25.5° from the CMB dipole: a superposition of the Virgo cluster periphery, the Hydra–Antlia wall, and the southern Local Supercluster extension"**

*Author:* Evgeny V. Chernoknizhny (Independent researcher)  
*ORCID:* [0009-0007-2558-9172](https://orcid.org/0009-0007-2558-9172)  
*DOI (v2.0):* [10.5281/zenodo.23457373](https://doi.org/10.5281/zenodo.23457373)  
*Concept DOI (all versions):* [10.5281/zenodo.23441334](https://doi.org/10.5281/zenodo.23441334)

## Project Overview

The paper reports the detection of a statistically significant angular excess of galaxy surface density at an angular radius of 25.5° around the CMB dipole direction (RA = 167.9°, Dec = −7.2°). 

**Key result (v2.0):** The excess is not a ring in the geometric sense. Azimuthal and depth analysis shows that it is a projection of three known local volume structures:
1.  **Arc 210–240°** — the Virgo cluster periphery (median depth \(cz \approx 1230-1450\) km/s).
2.  **Arc 30–60°** — the near part of the Hydra–Antlia wall (median \(cz \approx 2400-2650\) km/s).
3.  **Arc 180–210°** — the southern extension of the Virgo supercluster (median \(cz \approx 1240-1600\) km/s).

After masking the Virgo cone, the ratio drops from ~1.9 to ~1.4, but the residual splits into two independent arcs, consistently reproduced in both catalogues (2MRS and CF4).

## Repository Contents

*   `2mrs.csv`, `cf4.csv` — Input catalogues (columns: RA, DEC, CZ).
*   `convert_to_csv.py` — Script to convert original FITS files to CSV format.
*   `ring_mask.py` — Code for the main ring ratio analysis, mock catalogues, and Virgo masking test.
*   `ring_final_test.py` — Code for the final decomposition: sector analysis, arc cross-match, and cz-profiles.
*   `paper_v2.tex` — LaTeX source of the manuscript (v2.0).
*   `CMB_dipole_excess_25.5deg_v2.pdf` — Compiled PDF of the manuscript.

## How to Run the Analysis

1.  **Install dependencies:**
    ```bash
    pip install numpy pandas scipy matplotlib astropy
