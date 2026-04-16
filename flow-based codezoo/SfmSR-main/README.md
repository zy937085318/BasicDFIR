# SfmSR: Semantic-Guided Flow Matching for Fast and Accurate Remote Sensing Image Super-Resolution

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**SfmSR** is a novel deep learning framework for remote sensing image super-resolution (SR), leveraging **semantic-guided flow matching** to achieve fast and accurate high-resolution image reconstruction. This repository provides the official implementation of our method, along with weights and evaluation scripts.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Training](#training)
   - [Testing](#testing)
5. [Weight](#pre-trained-models)
6. [Datasets](#datasets)

---

## Introduction
Remote sensing image super-resolution is a critical task for enhancing the spatial resolution of low-resolution satellite or aerial images. Traditional methods often suffer from slow inference speeds or artifacts in complex scenes. **SfmSR** addresses these challenges by introducing a **semantic-guided flow matching** mechanism, which:
- Accelerates the sampling process by reducing the number of required iterations.
- Improves stability and accuracy by incorporating semantic information to guide the reconstruction process.
- Reduces hallucination artifacts, ensuring reliable results in practical applications.

This repository provides the code and models to reproduce our results and facilitate further research.

---

## Key Features
- **Semantic-Guided Flow Matching**: Utilizes semantic maps to guide the flow matching process, ensuring accurate and stable super-resolution.
- **Fast Inference**: Optimized sampling process for real-time or near-real-time applications.
- **High-Quality Results**: Produces high-resolution images with minimal artifacts, even in complex scenes.
- **Modular Design**: Easy to extend and adapt for other image restoration tasks.

---

## Installation
To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/SfmSR.git
cd SfmSR
pip install -r requirements.txt
```
---

## Usage
### Training
To train the SfmSR model on your dataset, run the following command:

```bash
python train.py --confdataset_config_name configs/train_config.json
```
- Modify the train_config.yml file to specify dataset paths, hyperparameters, and training settings.
- Training logs and model checkpoints will be saved in the logs/ directory.
### Eval
```bash
python eval.py 
```

- Update the test_config.json file to specify the test dataset path and model checkpoint.

- Results, including super-resolved images and metrics (e.g., PSNR, LPILS), will be print.

### Weight
model weights are available for download:

- **SfmSR Model Weights**: [Download Link](https://drive.google.com/file/d/1NVDbDvDTOaob1-Vvz174DynGLpDVR7e2/view?usp=sharing)

Download the weights and place them in the `wegiht/` directory for use in inference or fine-tuning.

---

## Datasets
### Validation Dataset
The validation dataset used in this repository is available in the [FastDiffSR repository](https://github.com/Meng-333/FastDiffSR). You can download the dataset from there and run prepare_data_mfe_dm.py 

### Pretraining Dataset
For pretraining, we used the **fMoW (Functional Map of the World) dataset**. You can download and prepare the dataset using the official repository:
- **fMoW Dataset GitHub**: [fMoW Dataset](https://github.com/fMoW/dataset)

Follow the instructions in the official repository to download and preprocess the dataset.