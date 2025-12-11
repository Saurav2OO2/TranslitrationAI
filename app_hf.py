import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load model from Hugging Face Hub
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

print(f"Loading model: {MODEL_NAME}")
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Language Code Mapping
LANG_MAP = {
    "Hindi": "hi_IN",
    "Tamil": "ta_IN",
    "Bengali": "bn_IN"
}

def transliterate(text, target_lang_name):
    """Transliterate Roman/English text to target Indic script."""
    if not text or not text.strip():
        return "Please enter some text to transliterate."
    
    target_lang_code = LANG_MAP[target_lang_name]
    
    # Set source language and tokenize
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt")
    
    # Generate with forced target language
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code],
        max_length=128
    )
    
    # Decode and return
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=transliterate,
    inputs=[
        gr.Textbox(
            label="Input Text (Roman/English)", 
            placeholder="e.g., namaste, good, school",
            lines=2
        ),
        gr.Dropdown(
            choices=["Hindi", "Tamil", "Bengali"], 
            label="Target Language", 
            value="Hindi"
        )
    ],
    outputs=gr.Textbox(label="Transliterated Output", lines=2),
    title="🌐 Multilingual Transliteration",
    description="Convert Roman/English text to Hindi, Tamil, or Bengali scripts using mBART-50.",
    examples=[
        ["namaste", "Hindi"],
        ["good", "Hindi"],
        ["school", "Tamil"],
        ["vanakkam", "Tamil"],
        ["namoskar", "Bengali"],
        ["yellow", "Hindi"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
