# LPA: Location Prediction with Activity Semantics

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![star](https://img.shields.io/github/stars/nehSgnaiL/LPA)](https://github.com/nehSgnaiL/LPA/stargazers)

## Overview

This repository contains the code and sample data for the LPA model from our paper: ***Improving Next Location Prediction with Inferred Activity Semantics in Mobile Phone Data***.

We propose a **semantics-enhanced next location prediction framework** that infers and integrates user activities into an LSTM architecture with attention mechanisms and multimodal embeddings. 
The findings **highlight the value of enriching trajectory data with activity-level context**, which enables models to better capture the behavioural motivations behind movement. 

![Framework](./img/research-framework.png)

## Table of contents

- [Overview](#Overview)
- [Structure](#Structure)
- [Quick-start](#Quick-start)

## Structure

The repository is developed using the following libraries: [Pytorch](https://github.com/pytorch/pytorch) and [LibCity](https://github.com/LibCity/Bigscity-LibCity).

Directory and file descriptions:

+ `data`: Contains sample data for running the framework (data format requirements align with LibCity). 

+ `downstream` & `embed`: Define the LPA model for next-location prediction.

+ `utils`: Supplementary code for running the framework.

+ `args.py`: Default configuration settings for the framework.

+ `dataset.py`: Functions for reading and loading data.

+ `evaluator.py`: Evaluation functions to assess framework performance.

+ `executor.py`: Defines the training process of the framework.

+ `pipeline.py`: Pipeline for executing the framework.

+ `main.py`: Main script to run the framework.

## Quick-start
Step 1. Set Configuration in `args.py`

> [!TIP] 
> The key parameter `activity_type` in `args.py` determines the activity type integrated into the model: 
> + **None**: The model uses only location data.
> + **A3**: The model integrates location data and primary activities (Home, Work, Non-mandatory).
> + **A6**: The model integrates location data, primary activities, and non-mandatory activities (Home, Work, Shopping, Leisure, Eat out, Personal affairs).

Step 2. Run the Framework

> [!NOTE] 
> Execute `main.py` to start the framework. Note the following:
> + The configurations in `args.py` are default settings.
> + Any configurations set directly in `main.py` will overwrite the default config.

Step 3. Check the Results

> [!NOTE] 
> Upon successful completion, a `cache` directory will be generated containing:
> + Training logs
> + Cached datasets
> + Model checkpoints

## Citation

If you consider it useful for your research or development, please consider citing our [paper]().

```
to be added.
```
