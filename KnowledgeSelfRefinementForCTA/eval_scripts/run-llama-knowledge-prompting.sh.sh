# SOTAB V2 CTA subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-similar_refined --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-comparative --suff -similar --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-similar_refined --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --run -unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-comparative --suff -similar --runs 3

# WikiTURL subset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-comparative --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-similar-hier_refined --suff -similar-hier --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-comparative --suff -similar-hier --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-similar-hier_refined --suff -similar-hier --runs 3

# Limaye dataset
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-similar_refined --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -unsloth-Meta-Llama-3.1-8B-Instruct-bnb-4bit-comparative --suff -similar --runs 3

CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_demonstration --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_initial --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -gpt-4o-2024-05-13_demonstration_unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-similar_refined --suff -similar --runs 3
CUDA_VISIBLE_DEVICES="0" python llm-test-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --run -unsloth-Meta-Llama-3.1-70B-Instruct-bnb-4bit-comparative --suff -similar --runs 3