import json
import pandas as pd
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import torch
import shutil
import os

# Configuration
# User requested "preloaded" mBART. Using local path if it appears valid, else huggingface hub.
# Assuming 'models/base_mbart' might contain the model, otherwise we download/cache.
# Given the user's specific request "delete mt5" and "use only mBart which is preloaded", 
# we will try to use the local 'models/base_mbart' if it exists, or the standard hub name.
MODEL_NAME = "models/base_mbart" if os.path.exists("models/base_mbart") else "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "models/mbart-transliteration"
MAX_LENGTH = 128
EPOCHS = 10 
BATCH_SIZE = 4

# Language Code Mapping
# mBART-50 uses specific locale codes.
# Source is always Roman English (en_XX)
# Targets are Hindi (hi_IN), Tamil (ta_IN), Bengali (bn_IN)
LANG_MAP = {
    "hin": "hi_IN",
    "tam": "ta_IN",
    "ben": "bn_IN"
}

def load_data(path):
    df = pd.read_json(path, lines=True)
    return Dataset.from_pandas(df)

def main():
    model_name_or_path = "models/base_mbart" if os.path.exists("models/base_mbart") else "facebook/mbart-large-50-many-to-many-mmt"
    print(f"Attempting to load tokenizer and model from: {model_name_or_path}")
    
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
        model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Failed to load from {model_name_or_path}: {e}")
        if model_name_or_path != "facebook/mbart-large-50-many-to-many-mmt":
            print("Falling back to Hugging Face Hub model...")
            model_name_or_path = "facebook/mbart-large-50-many-to-many-mmt"
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
            model = MBartForConditionalGeneration.from_pretrained(model_name_or_path)
        else:
            raise e
            
    # Update MODEL_NAME for reference if needed, though we use objects now
    
    print("Loading data...")
    train_ds = load_data("data/train.json")
    val_ds = load_data("data/val.json")
    
    def preprocess_function(examples):
        # mBART multiligual translation/transliteration
        # We need to set the source language (en_XX) and force the target language token.
        
        # Inputs: Roman English
        inputs = examples["src"]
        targets = examples["tgt"]
        langs = examples["lang"]
        
        # Tokenize inputs (Source: English)
        tokenizer.src_lang = "en_XX"
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        
        # Tokenize targets
        # For mBART, we need to handle mixed languages in a batch if the trainer supports it,
        # OR we can just tokenize cleanly. 
        # Standard mBART usage:
        # labels = tokenizer(text_target=targets, ...)
        
        # However, we have different target languages in the same batch potentially?
        # The simple way is to process one by one or group them, BUT:
        # We can just tokenize the targets normally.
        # The TRICK for mBART during INFERENCE is forcing the BOS token.
        # During TRAINING, the decoder_input_ids should start with the lang token.
        # transformers library handles this usually if we use `text_target`.
        
        # IMPORTANT: To make sure the model learns to generate the correct language,
        # we usually create a separate column for forced_bos_token_id or we rely on the tokenizer
        # to add the language code at the start of labels.
        
        # Let's tokenize targets carefully.
        # Since we have mixed targets in the batch (maybe), passing `src_lang` to tokenizer sets it for INPUT.
        # For targets, we should ideally tokenized them with their respective languages.
        
        # Actually, let's look at how we iterate.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        # We also need to tell the model which language to generate for each sample?
        # mBART expects `forced_bos_token_id` to be passed to generate().
        # During training, the teacher forcing includes the lang token at the start of decoder_input_ids.
        
        # To handle mixed languages efficiently in one preprocess call:
        # We probably need to update the labels to prepend the language token manually if the tokenizer didn't?
        # MBart50TokenizerFast usually prepends the lang token if set.
        
        # A robust way for mixed batches:
        # Iterate and tokenize? Or assume the batch is mixed and handle it?
        # The dataset has 'lang' column.
        
        # Let's do it manually to be safe for mixed batch training.
        
        final_labels = []
        for tgt, lang in zip(targets, langs):
            mapped_lang = LANG_MAP[lang]
            tokenizer.tgt_lang = mapped_lang
            # Tokenize single target
            tokenized_tgt = tokenizer(tgt, max_length=MAX_LENGTH, truncation=True)
            final_labels.append(tokenized_tgt["input_ids"])
            
        model_inputs["labels"] = final_labels
        
        return model_inputs

    print("Preprocessing data...")
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["src", "tgt", "lang"])
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=["src", "tgt", "lang"])
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=5e-5, # Standard for mBART fine-tuning
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
        
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete!")

if __name__ == "__main__":
    # Clean up old model dir if exists
    # shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    main()
