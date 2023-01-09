# Geospatial Machine Learning Working Directory

This directory contains the research internship work for Dr. Prakash P S, a researcher at ICHEC, that aims to apply geospatial machine learning techniques to analyze and model geospatial data. Geospatial machine learning involves using machine learning algorithms to analyze and understand patterns in data that has a geographic component, and it has many potential applications in many fields.

## Technologies

The following technologies are used in this project:

- Geospatial Packages: [Conda geospatial package](https://geospatial.gishub.org/)
- Machine Learning: [Tensorflow](https://tensorflow.org) and/or [Pytorch](https://pytorch.org).

All packages and libraries used are listed in the [requirements.txt](requirements.txt) file.

## Objectives

### Exploring Solar Panel Semantic Segmentation

The first objective is to explore the application of semantic segmentation to solar panel data. Semantic segmentation, or image segmentation, is the task of clustering parts of an image together that belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a specific category.

For this, the dataset chosen for solar panel segmentation is [Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery](https://zenodo.org/record/5171712). This dataset includes three groups of PV samples collected at different spatial resolutions (0.8m, 0.3m, and 0.1m) and includes samples from satellite and aerial imagery. The dataset is divided into different categories based on background land use type or roof type. More information about the dataset can be found in the [preprint](https://essd.copernicus.org/preprints/essd-2021-270/).

## Directory Structure

- `data/`: contains raw and processed data files.
- `notebooks/`: contains Jupyter notebooks with code and analysis.
- `models/`: contains trained machine learning models.
- `reports/`: contains weekly reports and visualizations generated during the project.
