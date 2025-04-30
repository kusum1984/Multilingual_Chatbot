import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Load environment variables
load_dotenv()

# Original ERROR_MESSAGES dictionary (unchanged)
ERROR_MESSAGES = {
    'EN': "The given English word is nonsense.",
    'FR': "Le mot français donné est un non-sens.",
    # ... (keep all original language entries)
    'DEFAULT': "The text appears to be gibberish."
}

# Original LANGUAGE_NAMES dictionary (unchanged)
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    # ... (keep all original language mappings)
}

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def get_system_prompt():
    """ORIGINAL SYSTEM PROMPT (unchanged)"""
    return """
    # Ultimate Gibberish Detection System v2.0
    
    ## Your Task:
    Analyze text for meaningful content in ANY language using these guidelines:
    
    === VALID TEXT EXAMPLES ===
    1. Dictionary Words: "apple", "computer", "こんにちは"
    2. Proper Nouns: "New York", "東京タワー"
    3. Technical Codes: "ID-5849-BN"
    
    === GIBBERISH EXAMPLES ===
    1. Random Typing: "asdfjkl;"
    2. Impossible Combinations: "xzqywv"
    3. Meaningless Repetition: "asdf asdf asdf"
    
    === RESPONSE FORMAT ===
    Respond with ONLY:
    - "Valid" OR
    - "Invalid|<reason>|<detected_lang>"
    """

def detect_language(text: str) -> Tuple[str, str]:
    """New helper to detect language for any text"""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Identify the language. Respond with just the 2-letter ISO code."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=10
        )
        lang_code = response.choices[0].message.content.strip().upper()
        lang_name = LANGUAGE_NAMES.get(lang_code.lower(), lang_code)
        return (lang_code, lang_name)
    except Exception:
        return ("XX", "Unknown")

def format_error_response(word: str, lang_code: str) -> str:
    """ORIGINAL ERROR FORMAT (unchanged)"""
    error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
    return f"word-{word} Langcode-{lang_code} expected error -{error_msg}"

def format_valid_response(word: str, lang_code: str, lang_name: str) -> str:
    """New format for valid words"""
    return f"word-{word} Langcode-{lang_code} (Valid {lang_name})"

def check_gibberish(text: str) -> Tuple[str, str, str, str, str]:
    """Enhanced version that maintains original detection but adds language info"""
    if not text.strip():
        return ("T", "XX", "Unknown", "", "")
    
    # First get language info (new)
    lang_code, lang_name = detect_language(text)
    
    # ORIGINAL GIBBERISH DETECTION LOGIC (unchanged)
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Analyze: '{text}'"}
            ],
            temperature=0.0,
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()
        
        if result == "Valid":
            return ("T", lang_code, lang_name, format_valid_response(text, lang_code, lang_name), "")
        else:
            parts = result.split('|')
            if len(parts) >= 3:
                detected_lang = parts[1].upper()
                if detected_lang in ERROR_MESSAGES:
                    lang_code = detected_lang
                    lang_name = LANGUAGE_NAMES.get(detected_lang.lower(), detected_lang)
            return ("F", lang_code, lang_name, "", format_error_response(text, lang_code))
    except Exception:
        return ("F", "XX", "Unknown", "", format_error_response(text, "XX"))

def run_tests():
    """Test function showing both valid and gibberish cases"""
    test_cases = [
        # Valid words with expected language
        ("Hello", "T", "EN"),
        ("Bonjour", "T", "FR"), 
        ("こんにちは", "T", "JA"),
        ("مرحبا", "T", "AR"),
        
        # Gibberish cases
        ("asdfghjkl", "F", None),
        ("केाीी", "F", "HI"),
        ("xzqywv", "F", None),
        ("123 123 123", "F", None)
    ]

    results = []
    for text, expected_status, expected_lang in test_cases:
        status, lang_code, lang_name, valid_msg, error_msg = check_gibberish(text)
        
        results.append({
            'Word': text,
            'Expected Status': expected_status,
            'Actual Status': status,
            'Language Code': lang_code,
            'Language Name': lang_name,
            'Valid Message': valid_msg,
            'Error Message': error_msg
        })
    
    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel("gibberish_test_results.xlsx", index=False)
    
    # Print samples
    print("Sample Valid Output:")
    print(df[df['Actual Status'] == 'T'].head(2)['Valid Message'].values)
    print("\nSample Gibberish Output:")
    print(df[df['Actual Status'] == 'F'].head(2)['Error Message'].values)

if __name__ == "__main__":
    run_tests()
