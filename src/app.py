import gradio as gr
import ctranslate2
from transformers import MT5TokenizerFast
import os

MODEL_PATH = "models/mt5-ct2"
# Fallback to non-optimized if not exists
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/mt5-transliteration"
    IS_CT2 = False
    from transformers import MT5ForConditionalGeneration
else:
    IS_CT2 = True

print(f"Loading model from {MODEL_PATH} (CT2={IS_CT2})...")
tokenizer = MT5TokenizerFast.from_pretrained(MODEL_PATH)

if IS_CT2:
    try:
        translator = ctranslate2.Translator(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load CT2 model: {e}")
        IS_CT2 = False
        from transformers import MT5ForConditionalGeneration
        MODEL_PATH = "models/mt5-transliteration" 
        if not os.path.exists(MODEL_PATH):
             MODEL_PATH = "google/mt5-small"
        print(f"Fallback to HF model: {MODEL_PATH}")
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
else:
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)

PREFIX_MAP = {
    "Hindi": "transliterate to hindi: ",
    "Tamil": "transliterate to tamil: ",
    "Bengali": "transliterate to bengali: "
}

def transliterate(text, target_lang_name):
    prefix = PREFIX_MAP[target_lang_name]
    input_text = prefix + text
    
    # Tokenize
    if IS_CT2:
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
        results = translator.translate_batch(
            [source],
        )
        target_tokens = results[0].hypotheses[0]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
    else:
        inputs = tokenizer(input_text, return_tensors="pt")
        generated = model.generate(**inputs, max_length=128)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

demo = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(label="Input Text (English/Roman)", placeholder="e.g. namaste"),
        gr.Dropdown(choices=["Hindi", "Tamil", "Bengali"], label="Target Language", value="Hindi")
    ],
    outputs="text",
    title="Multilingual Transliteration (mT5 + CTranslate2)",
    description="Transliterate English text to Hindi, Tamil, or Bengali."
)

if __name__ == "__main__":
    demo.launch()
