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

# 🔍 Enhanced System Prompt (7 Languages)
system_prompt = """
You are a world-class linguistic analysis system specializing in detecting gibberish across:
Spanish (es), Portuguese (pt), Chinese (zh), Japanese (ja), German (de), French (fr), and Hindi (hi).

### Deep Analysis Framework ###
1. Gibberish Indicators:
- Random keyboard sequences (qwerty, azerty)
- Invalid character combinations for script/language
- Meaningless repetition (e.g., "foo foo foo")
- Pseudo-words resembling but not matching real vocabulary
- Nonsensical symbol combinations
- Impossible n-grams for the language
- Mixed scripts without meaning (e.g., 漢字+abc+123)
- Syllable salad (random valid syllables without structure)

2. Validity Criteria (ANY of these should pass):
- ≥1 valid word/morpheme in target language
- Recognizable proper nouns/names
- Grammatical structure (even partial)
- Contextual coherence between words
- Valid numerical/formal expressions
- Common abbreviations/acronyms
- Culturally relevant phrases/idioms
- Common greetings in any language
- Mixed scripts when culturally appropriate
- Short but meaningful expressions

### Language-Specific Rules ###
[Spanish/Portuguese]
- Validate diacritic usage (ñ, ç, á, etc.)
- Check for valid word endings (-ción, -mente)
- Reject keyboard walks (qwerty, asdf)

[Chinese/Japanese]
- Validate character combinations
- Check radical usage in Chinese
- Validate script mixing in Japanese
- Reject random kanji/hanzi combinations

[German/French]
- Validate compound words
- Check for valid prefixes/suffixes
- Validate diacritic usage (ü, ö, ä, é, è)

[Hindi]
- Validate Devanagari sequences
- Check for valid matra combinations
- Reject random akshara combinations

### Decision Protocol ###
When in doubt, err on the side of validity. Respond ONLY:
- "Valid" if text meets ANY validity criteria OR is a:
  * Recognizable proper noun/name
  * Common phrase/expression
  * Partial but meaningful fragment
  * Mixed but valid script combination
- "<Reason>|<lang_code>" ONLY for clear gibberish cases

Example Responses:
1. "Valid"  # For "Hola mundo"
2. "Valid"  # For "123 Main St"
3. "Keyboard walk|es"
4. "Random kanji|ja"
5. "No morphemes|hi"
"""

# ✍️ Optimized User Prompt
user_prompt = """
Analyze this text for gibberish with balanced rigor:

Text: "{text}"
Expected Language: {lang_code}

Analysis Steps:
1. Identify probable language(s)
2. Validate script/character usage
3. Check for meaningful morphemes
4. Evaluate grammatical structure
5. Assess contextual coherence

Special Cases to Accept:
- Common greetings/phrases in any language
- Proper nouns (names, addresses, IDs)
- Mixed scripts when culturally appropriate
- Short but meaningful expressions
- Numbers/formulas/codes

Required Output:
ONLY one of:
- "Valid" OR
- "<Reason>|<lang_code>"
"""

# 🌐 Comprehensive Error Messages
LANGUAGE_ERRORS = {
    "hi": "अमान्य पाठ: कोई सार्थक भाषाई संरचना नहीं",
    "es": "Texto no válido: sin estructura lingüística",
    "pt": "Texto inválido: sem estrutura reconhecível", 
    "zh": "无效文本: 无有意义的语言结构",
    "ja": "無効なテキスト: 意味のある構造なし",
    "de": "Ungültiger Text: Keine sinnvolle Struktur",
    "fr": "Texte invalide : aucune structure significative"
}

def check_gibberish(text: str, lang_code: str = "es") -> Tuple[str, str, str]:
    """
    Enhanced gibberish detector with better valid text recognition
    
    Args:
        text: Input text to analyze
        lang_code: Expected language code for analysis
    
    Returns:
        Tuple: (status, error_type, message)
            status: 'T' (valid) or 'F' (invalid)
            error_type: '' or 'gibberish_error'
            message: Localized error if invalid
    """
    try:
        # Skip empty string check
        if not text.strip():
            return ("T", "", "")
            
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.format(text=text, lang_code=lang_code).strip()}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.0,  # More deterministic
            top_p=0.9,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()
        
        if result.lower() == "valid":
            return ("T", "", "")
        else:
            # Parse response for reason and detected language
            parts = result.split('|')
            reason = parts[0].strip() if len(parts) > 0 else "Invalid text"
            detected_lang = parts[1].strip() if len(parts) > 1 else lang_code
            
            return (
                "F",
                "gibberish_error",
                f"{reason}: {LANGUAGE_ERRORS.get(detected_lang.lower(), LANGUAGE_ERRORS['es'])}"
            )
            
    except Exception as e:
        return ("F", "api_error", f"Analysis failed: {str(e)}")


# === Enhanced Test Suite ===
def run_tests():
    test_cases = [
        # Valid Texts (should pass)
        ("Hola mundo", "es", "T"),
        ("Olá mundo", "pt", "T"),
        ("你好世界", "zh", "T"), 
        ("こんにちは世界", "ja", "T"),
        ("Hallo Welt", "de", "T"),
        ("Bonjour le monde", "fr", "T"),
        ("नमस्ते दुनिया", "hi", "T"),
        ("123 Main St", "en", "T"),
        ("ID-4567-XY", "es", "T"),
        ("Paris", "fr", "T"),  # Single proper noun
        ("東京", "ja", "T"),  # Short but valid
        ("漢字+ひらがな", "ja", "T"),  # Mixed script
        ("A", "fr", "T"),  # Single character
        ("", "es", "T"),  # Empty string
        
        # Gibberish Cases (should fail)
        ("asdfghjkl", "es", "F"),
        ("qwertyuiop", "pt", "F"),
        ("随机汉字", "zh", "F"),  # Random Chinese characters
        ("あかさたなは", "ja", "F"),  # Random hiragana
        ("Müll Müll", "de", "F"),  # Meaningless repetition
        ("blah blah", "fr", "F"),
        ("कखग घङच", "hi", "F"),  # Random Devanagari
        ("!@#$%^&*", "es", "F"),
        ("xzqy wvut", "pt", "F"),
        ("漢字+abc+123", "ja", "F")  # Nonsense mixing
    ]

    print("=== Enhanced Gibberish Detector Test Results ===")
    passed = 0
    for idx, (text, lang, expected) in enumerate(test_cases, 1):
        status, err_type, message = check_gibberish(text, lang)
        result = "✅ PASS" if status == expected else "❌ FAIL"
        if result == "✅ PASS":
            passed += 1
        print(f"\nTest {idx}: {result}")
        print(f"Text: '{text}'")
        print(f"Language: {lang} | Expected: {expected} | Actual: {status}")
        if status == "F":
            print(f"Reason: {message}")
    
    print(f"\n=== Testing Complete: {passed}/{len(test_cases)} passed ===")

if __name__ == "__main__":
    run_tests()
