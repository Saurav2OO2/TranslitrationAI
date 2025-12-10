from datasets import load_dataset_builder, get_dataset_config_names, load_dataset

try:
    configs = get_dataset_config_names("ai4bharat/Aksharantar", trust_remote_code=True)
    print("Configs:", configs)
except Exception as e:
    print("Error getting configs:", e)
    
try:
    ds = load_dataset("ai4bharat/Aksharantar", "mic_hin_transliteration", split="train", trust_remote_code=True, streaming=True)
    print("Loaded mic_hin_transliteration sample:", next(iter(ds)))
except Exception as e:
    print("Error loading specific config:", e)
