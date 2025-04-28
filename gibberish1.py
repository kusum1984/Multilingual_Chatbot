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

# 🔍 Ultra-Detailed System Prompt (7 Languages)
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

2. Validity Criteria:
- ≥1 valid word/morpheme in target language
- Recognizable proper nouns/names
- Grammatical structure (even partial)
- Contextual coherence between words
- Valid numerical/formal expressions
- Common abbreviations/acronyms
- Culturally relevant phrases/idioms

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
Respond ONLY:
- "Valid" if text meets ANY validity criteria
- "<Reason>|<lang_code>" if gibberish, where:
  - Reason: 2-3 word description
  - lang_code: 2-letter language code

Example Responses:
1. "Valid"
2. "Keyboard walk|es"
3. "Random kanji|ja"
4. "No morphemes|hi"
5. "Symbol salad|pt"
"""

# ✍️ Optimized User Prompt
user_prompt = """
Analyze this text for gibberish with extreme rigor:

Text: "{text}"

Analysis Steps:
1. Identify probable language(s)
2. Validate script/character usage
3. Check for meaningful morphemes
4. Evaluate grammatical structure
5. Assess contextual coherence

Required Output:
ONLY one of:
- "Valid" OR
- "<Reason>|<lang_code>"

Critical Rules:
- Numbers/formulas/codes are valid
- Proper nouns are valid if recognizable
- Partial words valid if meaningful fragments
- Reject repetitive nonsense
- Flag pseudo-words
- Detect keyboard patterns
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
    Ultimate gibberish detector using advanced prompt engineering
    
    Args:
        text: Input text to analyze
        lang_code: Default language code for error messages
    
    Returns:
        Tuple: (status, error_type, message)
            status: 'T' (valid) or 'F' (invalid)
            error_type: '' or 'gibberish_error'
            message: Localized error if invalid
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.format(text=text).strip()}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1,
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


# === Comprehensive Test Suite ===
def run_tests():
    test_cases = [
        # Valid Texts
        ("Hola mundo", "es", "T"),
        ("Olá mundo", "pt", "T"),
        ("你好世界", "zh", "T"), 
        ("こんにちは世界", "ja", "T"),
        ("Hallo Welt", "de", "T"),
        ("Bonjour le monde", "fr", "T"),
        ("नमस्ते दुनिया", "hi", "T"),
        ("123 Main St", "en", "T"),
        ("ID-4567-XY", "es", "T"),
        
        # Gibberish Cases
        ("asdfghjkl", "es", "F"),
        ("qwertyuiop", "pt", "F"),
        ("随机汉字", "zh", "F"),
        ("あかさたなは", "ja", "F"),
        ("Müll Müll", "de", "F"),
        ("blah blah", "fr", "F"),
        ("कखग घङच", "hi", "F"),
        ("!@#$%^&*", "es", "F"),
        ("xzqy wvut", "pt", "F"),
        
        # Edge Cases
        ("", "es", "T"),  # Empty string
        ("A", "fr", "T"),  # Single character
        ("漢字+ひらがな+カタカナ", "ja", "T")  # Valid mixed script
    ]

    print("=== Ultimate Gibberish Detector Test Results ===")
    for idx, (text, lang, expected) in enumerate(test_cases, 1):
        status, err_type, message = check_gibberish(text, lang)
        result = "✅ PASS" if status == expected else "❌ FAIL"
        print(f"\nTest {idx}: {result}")
        print(f"Text: '{text}'")
        print(f"Language: {lang} | Expected: {expected} | Actual: {status}")
        if status == "F":
            print(f"Reason: {message}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    run_tests()
