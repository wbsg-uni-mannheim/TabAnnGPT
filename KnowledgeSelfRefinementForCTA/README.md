# Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation
This folder contains the code to reproduce the experiments in the paper <b>"Knowledge Generation and Self-refinement strategies for LLM-based CTA"</b>.

## Environement Setup

The packages required for running our code can be found in the `unsloth_env.yml` file. To build the enviroment, navigate to the folder containing the yml file and run:

 `conda env create -f unsloth_env.yml`.

## Data
We use three datasets is our experiments: a subset of SOTAB V2 CTA, a subset of WikiTURL and the Limaye dataset. The traning, validation and test splits usedin our pper can be found in the `data` directory.

The vocabulary of each dataset can be found in the respective "labels" folder in the "data" directory. For example, the label space for the WikiTURL dataset can be found in the folder `data/wikitables-2-labels/wikitables-2_all_labels.txt`.

## Baselines
To run the baselines for the <b>OpenAI</b> models, the `GPT-baseline.py` file is used. The `datasets` parameter can be changed to the dataset(s) being tested and the number of demonstrations selected for the prompts can be set to 0 or 5 according to the paper.

To run the baselines for the <b>open-sourced Llama</b> models, the `run-llama-baselines.sh` file can be used, by setting the parameter `--dataset` to the desired dataset name and the `--nr` (number of shots) to the desired number of demonstrations (0 or 5).

## Knowledge Generation Prompting

Pre-requisites:
1. Generate the label definitions: The ``generate_definitions.py`` file is used to generate the <i>initial</i> and <i>demonstration</i> definitions.
2. Create embeddings for the train, validation and test set using the ` embed_datasets.py` file.

We provide the generated definitions in the respective labels folder of each dataset in the data directory. Initial definitions file names follow the following pattern:

`[dataset_name]-[model_name(used to generate the definitions)]_initial_definitions.json`

The demonstration definitions follow the same pattern, an endng with demonstration_definitions. For our non-fine-tuning experiments, the model used to generate the definitions is always the model gpt-4o-2024-05-13.

To use the definitions in knowledge promping, the `GPT-definitions.py` file can be used for testing OpenAI models or `run-llama-knowledge-prompting.sh` for open-sourced models.

## Error-based self-refinement of definitions

To self-refine demonstration definitions, these steps should be followed:
1. Run classification of the validation set using knowledge prompting with demonstration definitions: The file `GPT-definitions.py` can be used with the needed dataset, the demonstration definitions and the parameter `run_val` set to True.
2. Refine the demonstration definitions by using the `refine_definitions.py` file.

After these steps, the refined definitions are created. We provide the refined definitions used in our paper in the respective labels folder. The refined definitions name follow the following pattern: 

`[dataset_name]-[previous_definitions_name]-[model_name(whose errors are used for refining the definitions)]_refined_definitions.json`

The refined definitions used in our paper are made available in the labels folder in the `data` directory of the respecitive dataset.

## Fine-tune open-source Llama models:
To create the four different sets for fine-tuning the Llama models, the `create-ft-sets.py` script is used. To create the data for fine-tuning OpenAI models, the `create-openai-sets.py` is used.

To fine-tune the models use the `run-fine-tuning.sh`, choose the name of the model to fine-tune, the dataset to use, the set to use and other parameters such as learning rate, epochs and batch size.

To use the fine-tuned model, the `run-fine-tuned-model-testing.sh`, and enter in the model name the name of the folder where the model was saved.

## Self-correction pipeline:
To use the self-correction pipeline, the following steps should be followed:

1. Run one-step CTA using zero or knowledge prompting with demonstration and comparative definitions with `GPT-baselines.py`/`run-llama-baselines.sh` or `GPT-definitions.py`/`run-llama-knowledge-prompting.sh` respectively. If these files have already been run for the above strategies, there is no need to run them again.
2. To correct the classification of the first step, run `GPT-reviewer.py`/`run-llama-reviewer.sh`/`.

## Model predictions, prompts and evaluation files
The predictions, evaluation and messages (prompts fed to the models) are organized under the respective model name in the `output` directory in three folders: predictions, evaluation and messages. For example, to check the predictions of the gpt-4o-mini model, the file can be found in the directory `output/sotabv2-subsetu/gpt-4o-mini/predictions/`.