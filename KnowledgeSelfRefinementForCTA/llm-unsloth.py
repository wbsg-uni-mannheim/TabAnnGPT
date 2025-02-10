import argparse
import torch
from transformers import  TrainingArguments, EarlyStoppingCallback
from utils import save_pickle_file
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login
import logging
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import pdb

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    login(token="HF_token") # Specify HF token if needed
    
    parser.add_argument("--model_id", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--r", default=8, type=int, help="LoRA r.")
    parser.add_argument("--la", default=8, type=int, help="LoRA alpha.")
    parser.add_argument("--ld", default=0.0, type=float, help="LoRA dropout.")
    parser.add_argument( "--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--set", type=str, default="sotabv2-subsetu")
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--suff", type=str, default="")

    args = parser.parse_args()
    
    print(args)

    max_seq_length = 5020
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
        cache_dir="hf_cache/"
    )

    run_val = True # default always run validation

    # Fine-tuning Sets:
    # Simple set
    # Fine-tuning with definitions
    # Multi-task fine-tuning
    # Multi-task-with demonstrations
    # Validation suffix
    val_suffix = ""
    if "definitions" not in args.subset:
        # Validation set for simple, multi-task sets
        val_suffix = ""
    else:
        val_suffix += "-with-gpt-4o-2024-05-13_demonstration_definitions"

    if "-instr" in args.subset:
      val_suffix += "-instr"
    if "-hier" in args.subset:
       val_suffix += "-hier"

    # Load datasets for fine-tuning
    if run_val:
        data_files = {
            "train": f"data/sets/{args.set}-cta-train{args.subset}.json",
            "val": f"data/sets/{args.set}-cta-val{val_suffix}.json",
        }
    else:
       data_files = {
            "train": f"data/sets/{args.set}-cta-train{args.subset}.json",
        }
    print("Loading following datasets:")
    print(f"data/sets/{args.set}-cta-train{args.subset}.json")
    print(f"data/sets/{args.set}-cta-val{val_suffix}.json")

    # Load datasets
    datasets = load_dataset("json", data_files=data_files, cache_dir='hf_cache/', field="data")
    
    def format_chat_template(row):
        row_json = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row
    
    # Tokenize datasets
    datasets = datasets.map(format_chat_template,num_proc=4)

    callback = EarlyStoppingCallback(early_stopping_patience=3)
    
    for run in range(0,args.runs):
        output_path = f"ft-models/{args.set}{args.subset}/FT-{args.model_id.replace('/','-')}_{args.set}{args.subset}_{args.lr}_{args.batch_size}_{args.epochs}_{max_seq_length}_r={args.r}_la={args.la}_ld={args.ld}{args.suff}/{run}"
        
        if not os.path.exists(f"{output_path}/result"):
            if "FT" not in args.model_id:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=args.r,
                    lora_alpha=args.la,
                    lora_dropout=args.ld,
                    bias = "none",
                    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    use_rslora=True,
                    use_gradient_checkpointing="unsloth",
                    random_state = run,
                    loftq_config = None,
                )
            
            print(model.print_trainable_parameters())

            training_args = TrainingArguments(
                output_dir=output_path,
                logging_strategy = "epoch",
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=int(args.batch_size/2),
                num_train_epochs=args.epochs,
                eval_strategy="epoch" if run_val else "no",
                save_strategy="epoch",
                save_total_limit = 2,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                gradient_checkpointing=True,
                logging_first_step=True,
                load_best_model_at_end=True if "call" in args.suff else False
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=datasets['train'],
                eval_dataset=datasets["val"] if run_val else None,
                dataset_text_field="text",
                args=training_args,
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=max_seq_length,
                callbacks = [callback] if "call" in args.suff else None
            )
            
            trainer_stats = trainer.train() #resume_from_checkpoint=checkpoint
            trainer.model.save_pretrained(f"{output_path}/result/")
            save_pickle_file(f"{output_path}/result/trainer_history.pkl", trainer.state.log_history)