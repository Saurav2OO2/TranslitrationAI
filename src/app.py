import gradio as gr
import ctranslate2
from transformers import MBart50TokenizerFast
import os

MODEL_PATH = "models/mbart-ct2"
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Fallback to non-optimized if not exists
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/mbart-transliteration"
    IS_CT2 = False
    from transformers import MBartForConditionalGeneration
else:
    IS_CT2 = True

model_to_load = MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL
if model_to_load == BASE_MODEL: 
    IS_CT2 = False
    from transformers import MBartForConditionalGeneration

print(f"Loading model from {model_to_load} (CT2={IS_CT2})...")

try:
    if IS_CT2:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_to_load)
        translator = ctranslate2.Translator(model_to_load)
    else:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_to_load)
        model = MBartForConditionalGeneration.from_pretrained(model_to_load)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to base Hugging Face model...")
    IS_CT2 = False
    from transformers import MBartForConditionalGeneration
    model_to_load = BASE_MODEL
    tokenizer = MBart50TokenizerFast.from_pretrained(model_to_load)
    model = MBartForConditionalGeneration.from_pretrained(model_to_load)

# Language Code Mapping
LANG_MAP = {
    "Hindi": "hi_IN",
    "Tamil": "ta_IN",
    "Bengali": "bn_IN"
}

def transliterate(text, target_lang_name):
    target_lang_code = LANG_MAP[target_lang_name]
    
    # Tokenize
    if IS_CT2:
        tokenizer.src_lang = "en_XX"
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        
        target_prefix = [[target_lang_code]]
        results = translator.translate_batch(
            [source],
            target_prefix=target_prefix
        )
        target_tokens = results[0].hypotheses[0]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
    else:
        tokenizer.src_lang = "en_XX"
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code],
            max_length=128
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

demo = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(label="Input Text (English/Roman)", placeholder="e.g. namaste"),
        gr.Dropdown(choices=["Hindi", "Tamil", "Bengali"], label="Target Language", value="Hindi")
    ],
    outputs="text",
    title="Multilingual Transliteration (mBART)",
    description="Transliterate English text to Hindi, Tamil, or Bengali using mBART."
)

if __name__ == "__main__":
    demo.launch()
