### Tasks:

- Installed QGIS and explored the software’s functionality of shapes and lines.
- Configured GeoML Virtual Environment for Local Development. - Contains all the packages and libraries that would be required for Geospatial Machine Learning. https://github.com/bharathraj-v/geoml-wd/blob/main/requirements.txt
- Created a notebook file that explores Raster data from sample data from the chosen dataset. https://github.com/bharathraj-v/geoml-wd/blob/main/usage.ipynb
- Explored a lot of datasets for Solar Panel Segmentation. The notable ones are
	- https://www.kaggle.com/datasets/tunguz/deep-solar-dataset - Labelled Data
	- https://www.nearmap.com/au/en - Unlabelled High-Resolution Data
	- https://zenodo.org/record/5171712 - Labelled High-Quality Data
- Selected the “Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery” dataset (https://zenodo.org/record/5171712) for the task for the following reasons.

	- Labelled data with detailed masks and 3-channel raster data.
	- Covers diverse locations such as Rooftops, Ground, Shrubwood, Grassland, Cropland, Saline Alkali, Water Surface, etc. hence having the potential for powering a model that works in any geographical location globally.
	- Considerably high-resolution images and masks - the majority of them being 1024x1024 and the overall size - 7GB
	- More about the dataset at https://essd.copernicus.org/articles/13/5389/2021/essd-13-5389-2021.pdf
- Overall, Explored Raster and Vector data using QGIS and Python and explored datasets, & finalized a dataset containing raster bitmaps for the training of the ML model.