import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

# Language-specific error messages
ERROR_MESSAGES = {
    'ES': "El texto en español no tiene sentido.",
    'PT': "O texto em português é nonsense.",
    'ZH': "中文文本是乱码。",
    'JA': "日本語のテキストは無意味です。",
    'DE': "Der deutsche Text ist sinnlos.",
    'FR': "Le texte français est un non-sens.",
    'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'DEFAULT': "The text appears to be gibberish."
}

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """
    Advanced gibberish detection for 7 languages with localized error messages
    
    Returns:
        Tuple: (status, error_type, message)
        - status: 'T' (valid) or 'F' (invalid)
        - error_type: '' or 'gibberish_error'
        - message: Localized error message if invalid
    """
    # Enhanced system prompt
    system_prompt = """
    # Advanced Multilingual Gibberish Detector
    
    ## Languages Supported:
    Spanish (ES), Portuguese (PT), Chinese (ZH), 
    Japanese (JA), German (DE), French (FR), Hindi (HI)
    
    ## Validity Criteria (ANY of these make text VALID):
    - Contains ≥1 real dictionary word
    - Recognizable proper nouns/names
    - Valid numbers/addresses/codes
    - Common phrases/expressions
    - Meaningful single characters
    
    ## Gibberish Indicators (ALL must be true to reject):
    - Random keyboard sequences (qwerty, asdf)
    - Impossible character combinations
    - Meaningless repetition
    - Nonsensical symbol mixes
    - Pseudo-words with no meaning
    
    ## Language-Specific Rules:
    [ES/PT] Check diacritics and word endings
    [ZH/JA] Validate character combinations
    [DE/FR] Verify compound words/prefixes
    [HI] Validate Devanagari sequences
    
    ## Response Format:
    - "Valid" OR
    - "Invalid|<lang_code>"
    """

    user_prompt = f"""
    Analyze this text for gibberish: "{text}"
    
    Detection Steps:
    1. Identify probable language(s)
    2. Check character validity for language
    3. Verify word existence
    4. Evaluate structure
    
    Respond ONLY with:
    - "Valid" if text has meaning
    - "Invalid|<lang_code>" if gibberish
    """

    client = get_client()
    
    try:
        # Handle empty string case
        if not text.strip():
            return ("T", "", "")
            
        response = client.chat.completions.create(
            model="gpt-4",  # Replace with your actual model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=20
        )

        result = response.choices[0].message.content.strip()
        
        if result == "Valid":
            return ("T", "", "")
        else:
            # Parse the language code from response
            parts = result.split('|')
            lang_code = parts[1] if len(parts) > 1 else 'DEFAULT'
            error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
            
            return (
                "F",
                "gibberish_error",
                error_msg
            )
            
    except Exception as e:
        return ("F", "api_error", f"Analysis failed: {str(e)}")


# ===== TEST CASES =====
def run_tests():
    test_cases = [
        # Valid Texts (should pass)
        ("Hola mundo", "ES", "T"),
        ("Olá Brasil", "PT", "T"),
        ("你好世界", "ZH", "T"),
        ("こんにちは世界", "JA", "T"),
        ("Hallo Welt", "DE", "T"),
        ("Bonjour le monde", "FR", "T"),
        ("नमस्ते दुनिया", "HI", "T"),
        ("123 Main St", "EN", "T"),
        ("ID-4567-XY", "ES", "T"),
        
        # Gibberish Cases (should fail)
        ("asdfghjkl", "ES", "F"),
        ("qwertyuiop", "PT", "F"),
        ("随机汉字", "ZH", "F"),
        ("あかさたなは", "JA", "F"),
        ("xzqy wvut", "DE", "F"),
        ("blah blah", "FR", "F"),
        ("केाीी", "HI", "F"),
        ("!@#$%^&*", "EN", "F")
    ]

    print("=== Gibberish Detection Test Results ===")
    passed = 0
    total = len(test_cases)
    
    for idx, (text, lang, expected) in enumerate(test_cases, 1):
        status, err_type, msg = check_gibberish(text)
        result = "✅ PASS" if status == expected else "❌ FAIL"
        if status == expected:
            passed += 1
            
        print(f"\nTest {idx}: {result}")
        print(f"Text: '{text}'")
        print(f"Language: {lang} | Expected: {expected} | Actual: {status}")
        if status == "F":
            print(f"Message: {msg}")
    
    print(f"\n=== Results: {passed}/{total} passed ===")

if __name__ == "__main__":
    run_tests()
