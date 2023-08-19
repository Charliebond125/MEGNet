This repository contains code and resources for the research project titled "MEGNet: A MEG-Based Deep Learning Model for Cognitive and Motor Imagery Classification" The project aims to develop a robust pipeline for accurately classifying motor and cognitive imagery using magnetoencephalography (MEG) data. The proposed study includes meticulous preprocessing, model configuration, and training using deep learning architectures.


# Project Overview
The research project is structured into three main components:

* Data Preprocessing Notebook: This Jupyter Notebook (data_preprocessing.ipynb) outlines the comprehensive preprocessing pipeline designed to enhance the quality and reliability of MEG data. It covers resampling, bad channel interpolation, filtering, and artifact removal.

* Data Preparation and Model Training Notebook: In the Jupyter Notebook (data_prep_and_model_training.ipynb), the preprocessed data is further prepared for model training. Different deep learning models, including MEGNet, ShallowConvNet, and DeepConvNet, are trained, evaluated, and compared using cross-validation techniques.

* Model Implementations: The Python file (MEGmodels.py) contains implementations of the MEGNet, ShallowConvNet, and DeepConvNet architectures, each tailored for analyzing MEG data. These implementations are utilized in the training process.

# How to Use
* Open the Jupyter Notebook data_preprocessing.ipynb. Follow the step-by-step instructions to preprocess the MEG data. Adjust the preprocessing parameters as needed for your specific dataset.

* Move on to the Jupyter Notebook data_prep_and_model_training.ipynb. This notebook demonstrates how to prepare the preprocessed data for model training. It covers the configuration of different deep learning architectures and the hyperparameter tuning process. Training, validation, and evaluation procedures are explained in detail.

* If you wish to explore the model architectures themselves, refer to the models.py file. This file contains the implementations of MEGNet, ShallowConvNet, and DeepConvNet architectures, which can be further customized and utilized for various applications.

# Dependencies
The code in this repository depends on several Python packages:

- numpy
- pandas
- mne
- scikit-learn
- tensorflow (for deep learning models)
Please ensure you have these packages installed before running the code.

For any error or bugs contact us at 'minervasarma@gmail.com' and 'charliebond125@gmail.com'
