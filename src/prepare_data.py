import os
import random
import pandas as pd

# Define constraints
TARGET_LANGS = ["hin", "tam", "ben"]
NUM_SAMPLES = 50

def prepare_data():
    all_data = []
    
    # Hardcoded transliteration data to guarantee correct behavior (Transliteration NOT Translation)
    # English/Roman -> Native Script
    print("Generating hardcoded transliteration data...")
    
    # Format: (English/Roman, Native)
    data_map = {
        "hin": [
            ("namaste", "नमस्ते"), ("aap", "आप"), ("kya", "क्या"), ("kar", "कर"), ("rahe", "रहे"), ("ho", "हो"),
            ("mera", "मेरा"), ("naam", "नाम"), ("saurav", "सौरव"), ("hai", "है"),
            ("good", "गुड"), ("morning", "मॉर्निंग"), ("india", "इंडिया"), ("delhi", "दिल्ली"), ("mumbai", "मुंबई"),
            ("school", "स्कूल"), ("ghar", "घर"), ("pani", "पानी"), ("khana", "खाना"), ("chalo", "चलो"),
            ("kaise", "कैसे"), ("main", "मैं"), ("tum", "तुम"), ("hum", "हम"), ("woh", "वो"),
            ("kitab", "किताब"), ("pen", "पेन"), ("phone", "फ़ोन"), ("laptop", "लैपटॉप"), ("bottle", "बोतल"),
            ("table", "टेबल"), ("kursi", "कुर्सी"), ("fan", "फैन"), ("light", "लाइट"), ("door", "दरवाजा"),
            ("window", "खिड़की"), ("road", "रोड"), ("gadi", "गाड़ी"), ("bus", "बस"), ("train", "ट्रेन"),
            ("station", "स्टेशन"), ("airport", "एयरपोर्ट"), ("market", "मार्केट"), ("shop", "शॉप"), ("money", "मनी"),
            ("bank", "बैंक"), ("office", "ऑफिस"), ("work", "वर्क"), ("job", "जॉब"), ("friend", "फ्रेंड")
        ],
        "tam": [
            ("vanakkam", "வணக்கம்"), ("neengal", "நீங்கள்"), ("epadi", "எப்படி"), ("irukiroorgal", "இருக்கிறீர்கள்"),
            ("en", "என்"), ("peyar", "பெயர்"), ("chennai", "சென்னை"), ("tamily", "தமிழ்"), ("nalla", "நல்ல"),
            ("good", "குட்"), ("morning", "மார்னிங்"), ("night", "நைட்"), ("sappadu", "சாப்பாடு"), ("tanni", "தண்ணீர்"),
            ("veedu", "வீடு"), ("palli", "பள்ளி"), ("kalloori", "கல்லூரி"), ("office", "அலுவலகம்"), ("bus", "பேருந்து"),
            ("train", "ரயில்"), ("car", "கார்"), ("bike", "பைக்"), ("road", "சாலை"), ("kadai", "கடை"),
            ("kasu", "காசு"), ("bank", "வங்கி"), ("friend", "நண்பன்"), ("love", "காதல்"), ("amma", "அம்மா"),
            ("appa", "அப்பா"), ("thambi", "தம்பி"), ("thangai", "தங்கை"), ("annan", "அண்ணன்"), ("akka", "அக்கா"),
            ("school", "ஸ்கூல்"), ("class", "கிளாஸ்"), ("teacher", "டீச்சர்"), ("student", "ஸ்டூடன்ட்"), ("book", "புத்தகம்"),
            ("pen", "பேனா"), ("paper", "பேப்பர்"), ("phone", "போன்"), ("laptop", "லேப்டாப்"), ("computer", "கம்ப்யூட்டர்"),
            ("table", "மேசை"), ("chair", "நாற்காலி"), ("fan", "மின்விசிறி"), ("light", "விளக்கு"), ("door", "கதவு"), ("window", "ஜன்னல்")
        ],
        "ben": [
            ("namoskar", "নমস্কার"), ("apni", "আপনি"), ("kemon", "কেমন"), ("achen", "আছেন"),
            ("amar", "আমার"), ("naam", "নাম"), ("kolkata", "কলকাতা"), ("bangla", "বাংলা"), ("bhalo", "ভালো"),
            ("good", "গুড"), ("morning", "মর্নিং"), ("night", "নাইট"), ("khabar", "খাবার"), ("jol", "জল"),
            ("bari", "বাড়ি"), ("school", "স্কুল"), ("college", "কলেজ"), ("office", "অফিস"), ("bus", "বাস"),
            ("train", "ট্রেন"), ("car", "গাড়ি"), ("bike", "বাইক"), ("road", "রাস্তা"), ("dokan", "দোকান"),
            ("taka", "টাকা"), ("bank", "ব্যাংক"), ("bondhu", "বন্ধু"), ("bhalobasha", "ভালবাসা"), ("ma", "মা"),
            ("baba", "বাবা"), ("bhai", "ভাই"), ("bon", "বোন"), ("dada", "দাদা"), ("didi", "দিদি"),
            ("class", "ক্লাস"), ("teacher", "শিক্ষক"), ("student", "ছাত্র"), ("book", "বই"),
            ("pen", "পেন"), ("paper", "কাগজ"), ("phone", "ফোন"), ("laptop", "ল্যাপটপ"), ("computer", "কম্পিউটার"),
            ("table", "টেবিল"), ("chair", "চেয়ার"), ("fan", "ফ্যান"), ("light", "লাইট"), ("door", "দরজা"), ("window", "জানালা"),
            ("kothay", "কোথায়")
        ]
    }

    for lang in TARGET_LANGS:
        pairs = data_map[lang]
        # Extend to 50 if needed
        while len(pairs) < NUM_SAMPLES:
            pairs.extend(pairs[:NUM_SAMPLES - len(pairs)])
        
        pairs = pairs[:NUM_SAMPLES]
        
        for p in pairs:
            all_data.append({"src": p[0], "tgt": p[1], "lang": lang})

    # Shuffle all combined data
    random.shuffle(all_data)
    
    # Split
    train_data = []
    val_data = []
    test_data = []
    
    # Stratified split roughly
    for lang in TARGET_LANGS:
        lang_items = [x for x in all_data if x['lang'] == lang]
        # Ensure we have data
        if not lang_items:
            continue
        train_data.extend(lang_items[:40])
        val_data.extend(lang_items[40:45])
        test_data.extend(lang_items[45:50])
        
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save to JSONL
    pd.DataFrame(train_data).to_json("data/train.json", orient="records", lines=True)
    pd.DataFrame(val_data).to_json("data/val.json", orient="records", lines=True)
    pd.DataFrame(test_data).to_json("data/test.json", orient="records", lines=True)
    
    print("Data saved to data/ directory.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    prepare_data()
