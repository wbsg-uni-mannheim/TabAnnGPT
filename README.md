# Table Annotation using ChatGPT
This repository contains the code for the experiments run in the paper <a href="https://arxiv.org/abs/2306.00745">"Column Type Annotation using ChatGPT"</a>.

## Installation

The conda environment used for the experiments can be found in the file "llm.yml". To build the enviroment, navigate to the folder containing the yml file and run ```conda env create -f llm.yml```

## Data

The data used for the experiments is a sample from the WDC Schema.org Table Annotation Benchmark and is stored in the folder *data*. The folder contains a training set and a test set.

## Running the experiments

There are two types of experiments conducted: prompt-based (text completion) and message-based (chat completion). We design the prompts in three different formats: text, column and table. For example, if you would like to re-run the experiments for column format using prompts, you would need to run the notebook *Prompt-column-experiments.ipynb*.