import time
import os
import json
import torch
import ctranslate2
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import pandas as pd
from datasets import load_dataset

MODEL_DIR = "models/mbart-transliteration"
CT2_DIR = "models/mbart-ct2"
TEST_FILE = "data/test.json"
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Language Code Mapping
LANG_MAP = {
    "hin": "hi_IN",
    "tam": "ta_IN",
    "ben": "bn_IN"
}

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    # Return in GB if large, but MB is standard here
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
    model_to_load = MODEL_DIR if os.path.exists(MODEL_DIR) else BASE_MODEL
    print(f"Using {model_to_load}")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_to_load)
    hf_model = MBartForConditionalGeneration.from_pretrained(model_to_load)
    if torch.cuda.is_available():
        hf_model = hf_model.cuda()
    
    start_time = time.time()
    for item in samples:
        src = item['src']
        target_lang_code = LANG_MAP[item['lang']]
        
        # mBART Inference
        tokenizer.src_lang = "en_XX"
        encoded = tokenizer(src, return_tensors="pt")
        if torch.cuda.is_available():
            encoded = encoded.to("cuda")
            
        generated_tokens = hf_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code],
            max_length=128
        )
        _ = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
    hf_time = time.time() - start_time
    
    # 2. CTranslate2 Model
    print("Loading CT2 Model logic...")
    # CT2 Translator for Encoder-Decoder
    if os.path.exists(CT2_DIR):
        translator = ctranslate2.Translator(CT2_DIR)
        
        start_time = time.time()
        for item in samples:
            src = item['src']
            target_lang_code = LANG_MAP[item['lang']]
            
            # Tokenize source
            tokenizer.src_lang = "en_XX"
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(src))
            
            # Translate
            # CT2 requires target prefix or forced decoding.
            # API: translator.translate_batch(source, target_prefix=[[lang_token]])
            target_prefix = [[target_lang_code]]
            
            results = translator.translate_batch([source], target_prefix=target_prefix)
            # Decode not timed, but showing API usage
            
        ct2_time = time.time() - start_time
    else:
        print("CT2 Model not found, skipping speed test.")
        ct2_time = -1
    
    # Sizes
    hf_size = get_dir_size(MODEL_DIR) if os.path.exists(MODEL_DIR) else 0 
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
