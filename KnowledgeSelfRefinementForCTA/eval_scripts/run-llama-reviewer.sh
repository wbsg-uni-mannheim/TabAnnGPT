# Possible methods
# "-reviewer" (zero-shot setup)
# "-reviewer-explanation" (zero-shot setup with textual explanation)
# "-reviewer-errors" (selective comparative definitions setup)
# "-reviewer-errors-explanation" (selective comparative definitions setup with textual explanation)

# SOTAB V2 CTA subset
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer-errors

# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset sotabv2-subsetu --shot 0 --method -reviewer-errors

# Limaye dataset
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer-errors

# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset limayeu --shot 0 --method -reviewer-errors

# WikiTURL subset
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer-errors

# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer --run -gpt-4o-2024-05-13_demonstration --suff -similar
# CUDA_VISIBLE_DEVICES="0" python reviewer.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --dataset wikitables-2 --shot 0 --method -reviewer-errors
