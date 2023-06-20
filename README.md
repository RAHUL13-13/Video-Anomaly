<div align="center">
<h1>
<b>
Anomaly Detection Project
</b>
</h1>
<h4>

</h4>
</div align="center">
This repository contains the code for an anomaly detection project. The project focuses on detecting anomalies in a given dataset using various techniques and models. The code is organized into different directories, each serving a specific purpose. Below is a description of the contents of this repository:

<h1>
<b>
Directory Structure
</b>
</h1>

Auxilliary_files: This directory contains auxiliary files that are used in the project.

- aux.py: This file provides auxiliary functions and utilities for the anomaly detection process.

- text_divide.py: This file is responsible for dividing text data into smaller units for analysis.

Model: This directory contains files related to the models used for anomaly detection.

- discriminator.py: This file contains the implementation of the discriminator model.
- generator.py: This file contains the implementation of the generator model.
- ResNext 3D-101: This directory contains files related to the ResNext 3D-101 model used for feature extraction and processing.
- ResNext 3d-101 feature_extractor.py: This file implements the feature extraction process using the ResNext 3D-101 model.
- frame_interpolation.py: This file performs frame interpolation for video data preprocessing.
- model.py: This file contains the implementation of the ResNext 3D-101 model.
- pre-process.py: This file handles the pre-processing steps for the input data.
- preprocessing.py: This file contains functions for data preprocessing.
- random_sequence_shuffler.py: This file shuffles the input sequences randomly.
- video_loader.py: This file provides functionality to load video data.

Visualisation: This directory contains files related to the visualization of the anomaly detection results.

- PCA.py: This file performs Principal Component Analysis (PCA) for data visualization.
- t-SNE.py: This file performs t-Distributed Stochastic Neighbor Embedding (t-SNE) for data visualization.
