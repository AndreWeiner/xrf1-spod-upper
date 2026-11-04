<p align="center">
  <a href="https://www.for2895.uni-stuttgart.de/">
    <img src="assets/for2895_logo.png" alt="FOR 2895 research unit" width="480">
  </a>
</p>

# XRF1 upper-wing surface analysis

**Reproducible workflows accompanying a collaborative research article.**

This repository provides the data preparation and analysis workflows for the XRF1 upper-wing surface study within [FOR 2895](https://www.for2895.uni-stuttgart.de/), a research unit investigating unsteady flow and interaction phenomena at high-speed stall conditions.

The article is a collaborative effort between the [**Institute of Aerodynamics and Gas Dynamics**](https://www.iag.uni-stuttgart.de/en/), University of Stuttgart, and the [**Institute of Fluid Mechanics**](https://tu-dresden.de/ing/maschinenwesen/ism), TU Dresden.

[Reproducibility](#reproducibility) · [Publications and archive](#publications-and-archive) · [Acknowledgements](#acknowledgements)

## Reproducibility

### Setup

Use Python 3.14.6 and LaTeX. Run from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
cp .env.example .env
```

Set `PLT_DATA_DIR`, `PSP_DATA_DIR` and `OUTPUT_DIR` in `.env`. Run each section in order, starting with an empty output directory.

### Numerical data

Surface conversion:

```bash
python scripts/convert_surface.py --zones wing-upper --fields cp cfx cfy cfz
```

SVD and statistics plots:

```bash
python scripts/svd_study.py full
python scripts/review_svd.py
```

SPOD and spectral comparisons:

```bash
python scripts/surface_spod.py full
python scripts/plot_taper_comparison.py
python scripts/compare_cf_spectra.py
```

### PSP data

Statistics and masking comparison:

```bash
python scripts/psp_study.py full
python scripts/review_psp.py
python scripts/psp_global_statistics.py
python scripts/psp_default_crop_statistics.py
python scripts/psp_automatic_crop5_statistics.py
python scripts/plot_psp_global_statistics.py
```

Spatial-mean histories and spectra:

```bash
python scripts/plot_psp_means.py
python scripts/review_psp_trends.py
python scripts/compare_psp_mean_spectra.py
python scripts/update_strouhal_coordinates.py
```

SVD and rank plots:

```bash
python scripts/psp_svd.py full
python scripts/review_psp_svd.py
```

## License

The source code is licensed under the [GNU General Public License v3.0](LICENSE) (GPL-3.0-only).

## Publications and archive

Archive and publication links will be added.

## Acknowledgements

The authors gratefully acknowledge funding by the **Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)** within the research unit [FOR 2895](https://www.for2895.uni-stuttgart.de/).
