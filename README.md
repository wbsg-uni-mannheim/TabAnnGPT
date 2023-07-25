# Table Annotation using ChatGPT
This repository contains the code for the experiments run in the paper <a href="https://arxiv.org/abs/2306.00745">"Column Type Annotation using ChatGPT"</a>.

## Installation

The conda environment used for the experiments can be found in the file "llm.yml". To build the enviroment, navigate to the folder containing the yml file and run ```conda env create -f llm.yml```

## Data

The data used for the experiments is a sample from the WDC Schema.org Table Annotation Benchmark (<a href="http://webdatacommons.org/structureddata/sotab/">SOTAB</a>) and is stored in the folder *data*. The folder contains a training set and a test set.

## Running the experiments

There are two types of experiments conducted: prompt-based (text completion) and message-based (chat completion). We design the prompts in three different formats: text, column and table. For example, if you would like to re-run the experiments for column format using prompts, you would need to run the notebook *Prompt-column-experiments.ipynb*.

## Zero-shot comparison of ChatGPT-0301, GPT4-0613 and Falcon40B-instruct

|   | ChatGPT-0301<br>F1|GPT4-0613<br>F1|Falcon40B-instruct<br>F1|
|---|---|---|---|
|column+instructions|78.61|92.36|11.67|
|text+instructions|74.15|91.56|10.38|
|table+instructions|85.25|95.14|2.7|
|Two-step pipeline|89.47|94.95|-|
