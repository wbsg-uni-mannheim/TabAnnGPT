# simple fine-tuning
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20 --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20 --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1

CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20 --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20 --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1

# fine-tuning with definitions
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-with-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-with-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1

CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-with-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-with-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1

# multi-task
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-multi-task-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-multi-task-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1

CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-multi-task-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-multi-task-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1

# multi-task-3
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-multi-task-3-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-multi-task-3-gpt-4o-2024-05-13_demonstration --batch_size 16 --lr 1e-4 --r 32 --la 32 --ld 0.1

CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set sotabv2-subsetu --subset -random-20-multi-task-3-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1
CUDA_VISIBLE_DEVICES="0" python llm-ft-unsloth.sh --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --set wikitables-2 --subset -random-20-multi-task-3-gpt-4o-2024-05-13_demonstration --batch_size 8 --lr 1e-4 --r 32 --la 32 --ld 0.1
