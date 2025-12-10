import json
import pandas as pd
from datasets import Dataset
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import torch
import shutil

# Configuration
MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = "models/mt5-transliteration"
MAX_LENGTH = 128
EPOCHS = 15 # More epochs for small data
BATCH_SIZE = 4

# Prefix map
PREFIX_MAP = {
    "hin": "transliterate to hindi: ",
    "tam": "transliterate to tamil: ",
    "ben": "transliterate to bengali: "
}

def load_data(path):
    df = pd.read_json(path, lines=True)
    return Dataset.from_pandas(df)

def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    print("Loading data...")
    # These files are now hardcoded valid transliteration data
    train_ds = load_data("data/train.json")
    val_ds = load_data("data/val.json")
    
    def preprocess_function(examples):
        inputs = [PREFIX_MAP[lang] + src for lang, src in zip(examples["lang"], examples["src"])]
        targets = examples["tgt"]
        
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing data...")
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["src", "tgt", "lang"])
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=["src", "tgt", "lang"])
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=5e-4, # Higher LR for small data/T5
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=5
    )
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        # Save base model anyway for pipeline continuity
        print("Saving base model as fallback...")
        
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete!")

if __name__ == "__main__":
    # Clean up old model dir if exists
    # shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    main()
