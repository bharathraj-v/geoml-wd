# Geospatial Machine Learning Working Directory

This is the working directory of the research internship work that aims to apply machine learning to geospatial data.

## Technologies

Geospatial Packages: [Conda geospatial package](https://geospatial.gishub.org/), Machine Learning: [Tensorflow](tensorflow.org) and/or [Pytorch](pytorch.org). Every package and library used is listed in the [requirements.txt](requirements.txt) file.

## Objectives

### Exploring Solar Panel Semantic Segmentation:

**Semantic Segmentation**:
Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category.

Dataset chosen for Solar panel Segmentation: https://zenodo.org/record/5171712

**Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery**: A photovoltaic (PV) dataset from satellite and aerial imagery. The dataset includes three groups of PV samples collected at the spatial resolution of 0.8m, 0.3m and 0.1m, namely PV08 from Gaofen-2 and Beijing-2 imagery, PV03 from aerial photography, and PV01 from UAV orthophotos. PV08 contains rooftop and ground PV samples. Ground samples in PV03 are divided into five categories according to their background land use type: shrub land, grassland, cropland, saline-alkali, and water surface. Rooftop samples in PV01 are divided into three categories according to their background roof type: flat concrete, steel tile, and brick. Data document can refer to the preprint https://essd.copernicus.org/preprints/essd-2021-270/


## Directory Structure

- `data/`: contains raw and processed data files.
- `notebooks/`: contains Jupyter notebooks with code and analysis.
- `models/`: contains trained machine learning models.
- `reports/`: contains weekly reports and visualizations generated.
