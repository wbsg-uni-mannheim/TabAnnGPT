# SOTAB V2 CTA subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 5 --dataset_version -random-20 --runs 3  --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 0 --runs 3  --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="sotabv2-subsetu" --shot 5 --dataset_version -random-20 --runs 3  --dataset_version="-random-20"

# WikiTURL subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --runs 3 --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 5 --dataset_version -random-20 --suff -hier --runs 3  --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 0 --suff -hier --runs 3  --dataset_version="-random-20"
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="wikitables-2" --shot 5 --dataset_version -random-20 --suff -hier --runs 3  --dataset_version="-random-20"

# Limaye dataset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --dataset="limayeu" --shot 5 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="limayeu" --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.py --model_id="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" --dataset="limayeu" --shot 5 --runs 3