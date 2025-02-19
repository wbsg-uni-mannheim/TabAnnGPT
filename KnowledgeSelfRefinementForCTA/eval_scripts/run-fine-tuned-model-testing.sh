# Simple fine-tuning
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_sotabv2-subsetu-random-20_0.0001_16_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_wikitables-2-random-20_0.0001_16_10_5020_r=32_la=32_ld=0.1 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_sotabv2-subsetu-random-20_0.0001_8_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_wikitables-2-random-20_0.0001_16_10_5020_r=32_la=32_ld=0.1 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

# Fine-tuning with definitions
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-with-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-with-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.0 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-with-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_wikitables-2-random-20-with-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.05 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-with-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-with-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-with-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_wikitables-2-random-20-with-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.05 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

# Multi-task
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-multi-task-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-multi-task-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-multi-task-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_wikitables-2-random-20-multi-task-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.05 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-multi-task-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-multi-task-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.0 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-multi-task-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_wikitables-2-random-20-multi-task-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.1 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

# Multi-task-3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit_wikitables-2-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration_0.0001_16_10_5020_r=32_la=32_ld=0.05 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id sotabv2-subsetu-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_sotabv2-subsetu-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.1 --dataset sotabv2-subsetu --shot 0 --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id wikitables-2-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration/FT-unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit_wikitables-2-random-20-multi-task-3-gpt-4o-2024-05-13_demonstration_0.0001_8_10_5020_r=32_la=32_ld=0.1 --dataset wikitables-2 --shot 0 --suff -hier --runs 3

