import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-07-01-preview"
)

# 🔍 Enhanced Linguistic Analysis Engine (7 Languages)
system_prompt = """
You are an advanced linguistic analysis engine specialized in detecting gibberish in:
Spanish (es), Portuguese (pt), Chinese (zh), Japanese (ja), German (de), French (fr), and Hindi (hi).

### Enhanced Detection Framework ###
1. Gibberish Definition:
- Text lacking coherent meaning/structure in target languages
- Random keyboard mashing (e.g., "asdfghjkl")
- Invalid character combinations
- Meaningless symbol strings

2. Valid Content Includes:
- ANY recognizable words in target languages
- Numbers/mathematical expressions with context
- Formal structures (code, formulas, IDs)
- Grammatical patterns (even partial)
- Proper nouns/names
- Common abbreviations

### Language-Specific Validation Rules ###

1. Script Validation:
- Hindi: Valid Devanagari sequences (e.g., "कक्षा" valid, "कखग" invalid)
- Japanese: Valid script mixing (Kanji+Hiragana+Katakana)
- Chinese: Valid character combinations (check radicals)
- European: Proper diacritic usage (e.g., "café" vs "c@fe")

2. Structural Analysis:
- Minimum 1 valid morpheme per language
- Valid word boundaries
- Acceptable punctuation

3. Statistical Checks:
- Keyboard walking patterns (qwerty, azerty)
- Character repetition (aaaa, 1111)
- Impossible n-grams (e.g., "xzq" in Spanish)

### Decision Protocol ###
Respond EXACTLY "Valid" if ANY of:
- ≥1 valid word in target languages
- Numbers with context (e.g., "123 Main St")
- Code/formulas/IDs (e.g., "ID-1234")
- Proper names/places

Respond EXACTLY "<Reason>|<lang_code>" if:
- No valid words (e.g., "asdfg" → "Random chars|es")
- Invalid script mixing (e.g., "漢字abc" → "Script mix|ja")
- Meaningless symbols (e.g., "@#$%" → "Symbols|pt")
- Keyboard patterns (e.g., "qwertyuiop" → "Keyboard walk|de")
- Repetitive nonsense (e.g., "Müll Müll Müll" → "Repetitive nonsense|de")
- Pseudo-words without meaning (e.g., "アプイマプ、スジドゥス" → "Pseudo-words|ja")
- Random syllable combinations (e.g., "ko bo ko bo" → "Random syllables|ja")

### Examples ###
Valid:
- Spanish: "Hola, ¿cómo estás?" 
- Portuguese: "Rua das Flores, 123"
- Chinese: "我的名字是张三"  
- Japanese: "東京スカイツリー"
- German: "Straßenbahn Haltestelle"
- French: "C'est la vie"
- Hindi: "मेरा नाम राज है"

Gibberish:
- Spanish: "asdf ñlkj" → "Random chars|es"
- Portuguese: "zxcv qwer" → "Keyboard walk|pt" 
- Chinese: "随机汉字组合" → "No meaning|zh"
- Japanese: "あかさたなはま" → "No structure|ja"
- German: "qwertz uiopü" → "Keyboard seq|de"
- French: "!@£$%^&*" → "Symbols only|fr"
- Hindi: "कखग घङच" → "No morphemes|hi"
"""

# ✍️ Optimized User Prompt Template
user_prompt = """
Conduct comprehensive gibberish analysis on this text (focus: es/pt/zh/ja/de/fr/hi):

Text to analyze: "{text}"

Analysis Steps:
1. Language Identification
2. Script Validation
3. Structural Check
4. Statistical Analysis
5. Contextual Meaning Analysis


Required Output:
- "Valid" OR
- "<2-word reason in english>|<lang_code>"

Critical Notes:
- Consider partial words as valid only if they form meaningful fragments
- Proper nouns/names must be recognizable as such
- Common abbreviations must be legitimate
- Flag random sequences >3 characters
- Detect repetitive nonsense patterns
- Reject pseudo-words that don't form real vocabulary
"""

# 🌐 Enhanced Language-Specific Error Messages
LANGUAGE_ERRORS = {
    "hi": "अमान्य पाठ: कोई सार्थक शब्द नहीं मिले",
    "es": "Texto no válido: sin palabras reconocibles",
    "pt": "Texto inválido: sem estrutura linguística",
    "zh": "无效文本: 没有有意义的词语",
    "ja": "無効なテキスト: 認識可能な単語なし",
    "de": "Ungültiger Text: Keine sinnvollen Wörter",
    "fr": "Texte invalide : aucun mot reconnaissable"
}

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """
    Enhanced gibberish detection function using LLM analysis
    
    Args:
        text: Input text to analyze
    
    Returns:
        Tuple: (status, error_type, localized_message)
            status: 'T' for valid, 'F' for invalid
            error_type: '' if valid, 'gibberish_error' if invalid
            localized_message: Error message in detected language
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.format(text=text).strip()}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1,  # Lower temperature for more deterministic results
            top_p=0.9,
            max_tokens=20,
            stop=None
        )

        result = response.choices[0].message.content.strip()
        
        if result.lower() == "valid":
            return ("T", "", "")
        else:
            # Parse the error reason and language code
            parts = result.split('|')
            reason = parts[0].strip() if len(parts) > 0 else "Invalid text"
            lang_code = parts[1].strip() if len(parts) > 1 else "es"
            
            return (
                "F", 
                "gibberish_error", 
                f"{reason}: {LANGUAGE_ERRORS.get(lang_code, LANGUAGE_ERRORS['es'])}"
            )
    except Exception as e:
        return ("F", "api_error", f"Analysis failed: {str(e)}")


# === Test Suite ===
def run_tests():
    test_cases = [
        # Valid cases
        ("Hello world", "T"),  # English allowed as it's sometimes acceptable
        ("Hola mundo", "T"),
        ("123 Main Street", "T"),
        ("東京タワー", "T"),
        ("नमस्ते", "T"),
        
        # Gibberish cases
        ("asdfghjkl", "F"),
        ("qwertyuiop", "F"),
        ("!@#$%^&*", "F"),
        ("कखग घङच", "F"),
        ("xzqy wvut", "F"),
        
        # Edge cases
        ("", "T"),  # Empty string
        ("A", "T"),  # Single character
        ("ID-1234", "T"),  # Code/ID format
        ("漢字+ひらがな+カタカナ", "T")  # Valid mixed script
    ]

    print("=== Enhanced Gibberish Detector Tests ===")
    for idx, (text, expected) in enumerate(test_cases, 1):
        status, err_type, message = check_gibberish(text)
        result = "✅ PASS" if status == expected else "❌ FAIL"
        print(f"\nTest {idx}: {result}")
        print(f"Text: '{text}'")
        print(f"Expected: {expected} | Actual: {status}")
        if status == "F":
            print(f"Reason: {message}")
    
    print("\n=== Test Completion ===")

if __name__ == "__main__":
    run_tests()
