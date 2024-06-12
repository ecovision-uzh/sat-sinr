# Sat-SINR: High-Resolution Species Distribution Models through Satellite Imagery
This repository contains the code for replicating the experiments from **Sat-SINR: High-Resolution Species Distribution Models through Satellite Imagery**.
The work extends the [SINR](https://www.github.com/elijahcole/sinr) model with satellite imagery on the basis of the [GeoLifeClef 2023 challenge](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/overview). The code uses the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework with [Hydra](https://hydra.cc/docs/intro/).
# How to Run
To replicate the experiments, you will need to setup the Python environment, download the additional data, and pre-process two files.
It is important that the paths in the config local.yaml point to the proper files.
## Setup
Clone the repository:
```bash
git clone https://github.com/ecovision-uzh/sat-sinr.git
```
Create a new [Python](https://www.python.org) environment and install the requirements (in Unix):
```bash
python3 -m venv .sat-sinr-venv
source .sat-sinr-venv/bin/activate
pip install -r requirements.txt
```
## Additional Data Downloads
For the experiments, you are required to download series of additional files.
### Sentinel-2 Images
Source: [GeoLifeClef 2023 challenge](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/data), extracted from the [Ecodatacube platform](https://stac.ecodatacube.eu/)

Volume: ~5 million RGB & NIR Sentinel-2 images of size 128x128
### Bioclimatic & Elevation Rasters
Source: [WorldClim](https://www.worldclim.org/data/worldclim21.html)

Volume: 19 global bioclimatic rasters & 1 elevation raster at a resolution of 30 arcseconds (~1km)
### GBIF PO Occurrence Data
Source: [GeoLifeClef 2023 challenge](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/data), extracted from [GBIF](https://www.gbif.org/)

Volume: ~5 million occurrences in a .csv
### PA Surveys
Source: [GeoLifeClef 2023 challenge](https://www.kaggle.com/competitions/geolifeclef-2023-lifeclef-2023-x-fgvc10/data), collected from multiple sources

Volume: ~5.000 surveys in a .csv
## Data Pre-Processing Scripts
To replicate the experiments from the work, you need to pre-process the environmental and PO data.
### Bioclimatic & Elevation Rasters
The Jupyter notebook [crop and scale bioclim](https://github.com/ecovision-uzh/sat-sinr/blob/master/scripts/crop%20and%20scale%20bioclim.ipynb) loads the 20 TIFF-rasters and
turns them into a single numpy array, normalizes it and crops it to the European bounds.
### GBIF PO Occurrence Data
The Jupyter notebook [reduce classes](https://github.com/ecovision-uzh/sat-sinr/blob/master/scripts/crop%20and%20scale%20bioclim.ipynb) loads the 5 million PO occurences,
reduces sample-number per class to 1000 and removes all classes with less than 10 samples.
# Experiments
Ensure that the paths in the config local.yaml point to the proper files.

Train late fusion Sat-SINR with location, bioclimatic and Sentinel-2 images as predictor:
```bash
python3 main.py model=sat_sinr_lf dataset.predictors=loc_env_sent2
```
Train SINR with location as predictor:
```bash
python3 main.py model=sinr dataset.predictors=loc
```
The model options are: "sat_sinr_ef", "sat_sinr_mf", "sat_sinr_lf", "sinr" and "log_reg".

The dataset.predictors options are any combination of "loc", "env" and "sent2".
## Hyperparameters & Configs
The base_config.yaml contains the model and training parameters.

The dataset.yaml config contains dataset parameters.

The local.yaml config contains the paths pointing to the various files.
# Citation
The paper was presented at ISPRS TCII Symposium in June 2024:

Dollinger, J., Brun, P., Sainte Fare Garnot, V., and Wegner, J. D.: Sat-SINR: High-Resolution Species Distribution Models Through Satellite Imagery, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., X-2-2024, 41–48, https://doi.org/10.5194/isprs-annals-X-2-2024-41-2024, 2024
```
@Article{SatSINR_ISPRS2024,
AUTHOR = {Dollinger, J. and Brun, P. and Garnot, V. S. F. and Wegner, J. D.},
TITLE = {High-Resolution Species Distribution Models through Satellite Imagery},
JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {X-2-2024},
YEAR = {2024},
PAGES = {41-48},
URL = {https://doi.org/10.5194/isprs-annals-X-2-2024-41-2024},
DOI = {}
}
```
