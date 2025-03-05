# SOTAB V2 CTA subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20" --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20" --run="-self-cons-0.7"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20" --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20" --run="-self-cons-0.7"

# WikiTURL subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --runs 3 --dataset_version="-random-20" --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --runs 3 --dataset_version="-random-20" --run="-self-cons-0.7"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --suff -hier --runs 3  --dataset_version="-random-20" --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --suff -hier --runs 3  --dataset_version="-random-20" --run="-self-cons-0.7"

# Limaye dataset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3 --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3 --run="-self-cons-0.7"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3 --run="-self-cons-0.5"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3 --run="-self-cons-0.7"
