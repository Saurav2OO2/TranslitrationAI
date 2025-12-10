import os
import shutil
import ctranslate2
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration

# Configuration
BASE_MODEL = "google/mt5-small"
MODEL_DIR = "models/mt5-transliteration"
OUTPUT_DIR = "models/mt5-ct2"

def optimize_model():
    print(f"Optimizing model from {MODEL_DIR} to {OUTPUT_DIR}...")
    
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    # Convert using ctranslate2 command line tool equivalent
    # ctranslate2.converters.TransformersConverter(
    #     model_name_or_path, activation_scales=None, copy_files=None, load_as_float16=False, 
    #     low_cpu_mem_usage=False, trust_remote_code=False
    # ).convert(output_dir, vmap=None, quantization=None, force=False)
    
    # If local model doesn't exist (training failed) or is invalid, use base model
    model_path = MODEL_DIR
    # Check if directory exists AND contains config.json
    is_valid = os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    
    if not is_valid:
        print(f"Local model {MODEL_DIR} not found or invalid. Downloading base model {BASE_MODEL}...")
        model_path = "models/base_mbart"
        tokenizer = MT5TokenizerFast.from_pretrained(BASE_MODEL)
        model = MT5ForConditionalGeneration.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=model_path,
    )
    
    print("Converting with int8 quantization...")
    converter.convert(
        output_dir=OUTPUT_DIR,
        quantization="int8",
        force=True
    )
    
    # Copy tokenizer files
    tokenizer = MT5TokenizerFast.from_pretrained(model_path)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Optimization complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    optimize_model()
