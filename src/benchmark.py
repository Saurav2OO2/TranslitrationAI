import time
import os
import json
import torch
import ctranslate2
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration
import pandas as pd
from datasets import load_dataset

MODEL_DIR = "models/mt5-transliteration"
CT2_DIR = "models/mt5-ct2"
TEST_FILE = "data/test.json"

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # MB

def benchmark():
    # Load test data
    df = pd.read_json(TEST_FILE, lines=True)
    samples = df.to_dict('records')
    # Limit to 10 for speed
    samples = samples[:10]
    
    print(f"Benchmarking on {len(samples)} samples...")
    
    # 1. HF Model
    print("Loading HF Model...")
    model_to_load = MODEL_DIR if os.path.exists(MODEL_DIR) else "google/mt5-small"
    print(f"Using {model_to_load}")
    tokenizer = MT5TokenizerFast.from_pretrained(model_to_load)
    hf_model = MT5ForConditionalGeneration.from_pretrained(model_to_load)
    if torch.cuda.is_available():
        hf_model = hf_model.cuda()
    
    start_time = time.time()
    for item in samples:
        src = item['src']
        # lang mapping for prefix
        lang_map = {"hin": "transliterate to hindi: ", "tam": "transliterate to tamil: ", "ben": "transliterate to bengali: "}
        prefix = lang_map[item['lang']]
        print(f"DEBUG: prefix={type(prefix)}, src={type(src)}, val={src}")
        
        inputs = tokenizer(str(prefix) + str(src), return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        generated_tokens = hf_model.generate(**inputs, max_length=128)
        _ = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    hf_time = time.time() - start_time
    
    # 2. CTranslate2 Model
    print("Loading CT2 Model logic...")
    # CT2 Translator for Encoder-Decoder
    # If using CT2, we need to load it if exists, else skip
    if os.path.exists(CT2_DIR):
        translator = ctranslate2.Translator(CT2_DIR)
        
        start_time = time.time()
        for item in samples:
            src = item['src']
            lang_map = {"hin": "transliterate to hindi: ", "tam": "transliterate to tamil: ", "ben": "transliterate to bengali: "}
            prefix = lang_map[item['lang']]
            
            # Tokenize
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(prefix + src))
            
            # Translate
            results = translator.translate_batch([source])
            # Decode handled implicitly in speed check but technically need decoding
        ct2_time = time.time() - start_time
    else:
        print("CT2 Model not found, skipping speed test.")
        ct2_time = -1
    
    # Sizes
    hf_size = get_dir_size(MODEL_DIR) if os.path.exists(MODEL_DIR) else 0 # Approximate base model size is large
    ct2_size = get_dir_size(CT2_DIR)
    
    print("\nResults:")
    print(f"HF Time: {hf_time:.4f}s")
    print(f"CT2 Time: {ct2_time:.4f}s")
    if ct2_time > 0:
        print(f"Speedup: {hf_time/ct2_time:.2f}x")
    if ct2_size > 0:
        print(f"HF Size (Local): {hf_size:.2f} MB")
        print(f"CT2 Size: {ct2_size:.2f} MB")

if __name__ == "__main__":
    benchmark()
