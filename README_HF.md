# Multilingual Transliteration with mBART

This Space provides transliteration from Roman/English script to Hindi, Tamil, and Bengali using the mBART-50 model.

## Features
- **Roman to Hindi**: Convert English/Roman text to Devanagari script
- **Roman to Tamil**: Convert English/Roman text to Tamil script  
- **Roman to Bengali**: Convert English/Roman text to Bengali script

## Examples

| Input (Roman) | Hindi | Tamil | Bengali |
|---------------|-------|-------|---------|
| namaste | नमस्ते | நமஸ்தே | নমস্তে |
| good | गुड | குட் | গুড |
| school | स्कूल | ஸ்கூல் | স্কুল |

## Model
Uses `facebook/mbart-large-50-many-to-many-mmt` fine-tuned for transliteration tasks.

## Usage
1. Enter your text in Roman/English script
2. Select target language (Hindi, Tamil, or Bengali)
3. Get transliterated output

Built with Gradio and Transformers 🤗
