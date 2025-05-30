# NONA (Nearness of Neighbors Attention)

This repository contains the anonymized implementation for the paper:

> **Title:** Nearness of Neighbors Attention for Regression in Supervised Fine-Tuning  
> **Submitted to:** Neurips 2025

## Overview

NONA is a differentiable neighbor-based regression method. By directly supervising predictions through a mean squared error loss computed from true labels of neighboring points, NONA naturally aligns embedding geometry with the regression task.

This repository provides the full experimental pipeline, including model training and evaluation.

## Directory Structure  
nona_anon/  
├── data/ # Experimental datasets.  
│ └── dataset/  
│  └── ...  
│ └── dataset_classes.py # For building dataset specific dataloading  
├── models.py/ # NONA model components.  
├── similarity.py/ # Similarity matrix for NONA and SoftStep learned attention masking.  
├── utils.py/ # Shared helper functions.  
├── finetune.py # Perofrming supervised fine-tuning with NONA configs and dense benchmark.  
├── synthetic_regression_surfaces.ipynb # For exploring comparative ability of models to fit continuously labeled data.  
├── requirements.txt # Python dependencies.  
└── README.md # This file.  
