import os
import shutil
import ctranslate2
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Configuration
BASE_MODEL = "models/base_mbart" if os.path.exists("models/base_mbart") else "facebook/mbart-large-50-many-to-many-mmt"
MODEL_DIR = "models/mbart-transliteration"
OUTPUT_DIR = "models/mbart-ct2"

def optimize_model():
    print(f"Optimizing model from {MODEL_DIR} to {OUTPUT_DIR}...")
    
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    # If local model doesn't exist (training failed) or is invalid, use base model
    model_path = MODEL_DIR
    # Check if directory exists AND contains config.json
    is_valid = os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    
    if not is_valid:
        print(f"Local trained model {MODEL_DIR} not found or invalid. Falling back to base model...")
        
        # Determine base model path
        base_model_candidate = "models/base_mbart"
        use_hub = True
        
        if os.path.exists(base_model_candidate):
            # Check if it's loadable
            try:
                temp_tokenizer = MBart50TokenizerFast.from_pretrained(base_model_candidate)
                # If successful, use it
                base_model_path = base_model_candidate
                use_hub = False
            except Exception:
                print(f"Local base model {base_model_candidate} is corrupted. using Hub model.")
                
        if use_hub:
           base_model_path = "facebook/mbart-large-50-many-to-many-mmt"

        print(f"Using base model from: {base_model_path}")
        model_path = "models/base_mbart_local_copy"
        
        # Download/Save to local path for CTranslate2 to consume
        try:
            tokenizer = MBart50TokenizerFast.from_pretrained(base_model_path)
            model = MBartForConditionalGeneration.from_pretrained(base_model_path)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
        except Exception as e:
            print(f"Failed to prepare base model: {e}")
            return
        
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
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Optimization complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    optimize_model()
