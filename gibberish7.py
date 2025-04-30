import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Language-specific error messages with the exact required format
ERROR_MESSAGES = {
    'ES': "Langcode-ES expected error - El texto en español no tiene sentido.",
    'PT': "Langcode-PT expected error - O texto em português é nonsense.",
    'ZH': "Langcode-ZH expected error - 中文文本是乱码。",
    'JA': "Langcode-JA expected error - 日本語のテキストは無意味です。",
    'DE': "Langcode-DE expected error - Der deutsche Text ist sinnlos.",
    'FR': "Langcode-FR expected error - Le texte français est un non-sens.",
    'HI': "Langcode-HI expected error - दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'DEFAULT': "Langcode-XX expected error - The text appears to be gibberish."
}

# Language code to name mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'ru': 'Russian',
    'ar': 'Arabic',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'pa': 'Punjabi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'or': 'Odia',
    'ml': 'Malayalam',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'pl': 'Polish',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'fi': 'Finnish',
    'da': 'Danish',
    'no': 'Norwegian',
    'he': 'Hebrew',
    'fa': 'Persian',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fil': 'Filipino',
    'sw': 'Swahili',
    'ha': 'Hausa',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'st': 'Sotho',
    'sn': 'Shona',
    'am': 'Amharic',
    'so': 'Somali',
    'haw': 'Hawaiian',
    'mi': 'Māori',
    'sm': 'Samoan',
    'to': 'Tongan',
    'fj': 'Fijian',
    'el': 'Greek',
    'hu': 'Hungarian',
    'cs': 'Czech',
    'sk': 'Slovak',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sl': 'Slovenian',
    'bg': 'Bulgarian',
    'uk': 'Ukrainian',
    'be': 'Belarusian',
    'kk': 'Kazakh',
    'uz': 'Uzbek',
    'ky': 'Kyrgyz',
    'tg': 'Tajik',
    'mn': 'Mongolian',
    'bo': 'Tibetan',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'km': 'Khmer',
    'lo': 'Lao',
    'my': 'Burmese',
    'ka': 'Georgian',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'tk': 'Turkmen',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'cy': 'Welsh',
    'ga': 'Irish',
    'gd': 'Scottish Gaelic',
    'mt': 'Maltese',
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'af': 'Afrikaans',
    'is': 'Icelandic',
    'fo': 'Faroese',
    'sa': 'Sanskrit',
    'la': 'Latin',
    'eo': 'Esperanto'
}

def get_language_name(lang_code):
    """Convert language code to full name"""
    return LANGUAGE_NAMES.get(lang_code, lang_code) if lang_code else "Gibberish"

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """
    Gibberish detector that returns formatted error messages
    
    Returns:
        Tuple: (status, error_type, message)
        - status: 'T' (valid) or 'F' (invalid)
        - error_type: '' or 'gibberish_error'
        - message: Formatted error message if invalid
    """
    system_prompt = """
    Analyze text for meaningful content in any language. 
    Respond ONLY with:
    - "Valid" OR
    - "Invalid|<lang_code>|<reason>"
    Where lang_code is 2-letter language code (e.g. HI, ES)
    and reason is brief explanation
    """

    client = get_client()
    
    try:
        if not text.strip():
            return ("T", "", "")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze: {text}"}
            ],
            temperature=0.0,
            max_tokens=100
        )

        result = response.choices[0].message.content.strip()
        
        if result == "Valid":
            return ("T", "", "")
        else:
            parts = result.split('|')
            if len(parts) >= 3:
                lang_code = parts[1].upper()
                reason = parts[2]
                error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
                return ("F", "gibberish_error", error_msg)
            return ("F", "gibberish_error", ERROR_MESSAGES['DEFAULT'])
            
    except Exception as e:
        return ("F", "api_error", ERROR_MESSAGES['DEFAULT'])

def run_tests():
    test_cases = [
        # Valid Texts with expected language codes
        ("Hello world", "T", "EN"),
        ("Bonjour le monde", "T", "FR"),
        ("Hola mundo", "T", "ES"),
        ("Привет мир", "T", "RU"),
        ("مرحبا بالعالم", "T", "AR"),
        ("你好世界", "T", "ZH"),
        ("こんにちは世界", "T", "JA"),
        ("안녕하세요 세계", "T", "KO"),
        ("สวัสดีชาวโลก", "T", "TH"),
        ("Xin chào thế giới", "T", "VI"),
        ("Hallo Welt", "T", "DE"),
        ("Ciao mondo", "T", "IT"),
        ("Olá mundo", "T", "PT"),
        ("Witaj świecie", "T", "PL"),
        ("Γειά σου Κόσμε", "T", "EL"),
        ("Merhaba dünya", "T", "TR"),
        ("Hej världen", "T", "SV"),
        ("Pozdrav svijete", "T", "HR"),
        ("Ahoj světe", "T", "CS"),
        ("Helló világ", "T", "HU"),
        ("नमस्ते दुनिया", "T", "HI"),
        ("হ্যালো বিশ্ব", "T", "BN"),
        ("வணக்கம் உலகம்", "T", "TA"),
        ("שלום עולם", "T", "HE"),
        ("ID-5849-BN", "T", None),
        
        # Gibberish Cases - will trigger language-specific errors
        ("asdfghjkl", "F", None),
        ("qwertyuiop", "F", None),
        ("йцукенгшщз", "F", None),
        ("ضصثقضصثق", "F", None),
        ("ㅁㄴㅇㄹㅁㄴㅇ", "F", None),
        ("ฟหกด่าสว", "F", None),
        ("αβγδαβγδ", "F", None),
        ("אבגדאבגד", "F", None),
        ("!@#$%^&*", "F", None),
        ("xjdkl 392 sdk", "F", None),
        ("xzqy wvut", "F", None),
        ("blah blah", "F", None),
        ("केाीी", "F", "HI"),  # Hindi gibberish - will trigger HI error
        ("asdf asdf asdf", "F", None),
        ("123 123 123", "F", None),
        ("foo@bar$", "F", None),
        ("漢字漢字", "F", None),
        ("कखगघ", "F", "HI"),  # Hindi gibberish
        ("zzxxyy", "F", None),
        ("qwopasdf", "F", None),
        ("1a2b3c4d", "F", None),
        ("asdf1234", "F", None),
        ("@#$%^&", "F", None),
        ("zxcvbnm", "F", None),
        ("asdf;lkj", "F", None),
    ]

    print("=== Gibberish Detection Test ===")
    print(f"Running {len(test_cases)} test cases\n")
    
    # Prepare results with specified columns
    results = []
    columns = [
        'Language', 
        'Word', 
        'Expected Status', 
        'Actual Status',
        'Error Message'
    ]
    
    for text, expected_status, expected_lang in test_cases:
        # Run detection
        status, err_type, msg = check_gibberish(text)
        
        # Get language name for display
        lang_name = get_language_name(expected_lang) if expected_lang else "Gibberish"
        
        # Build result row
        results.append([
            lang_name,
            text,
            expected_status,
            status,
            msg if status == "F" else ""
        ])
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=columns)
    
    # Save to Excel
    filename = "gibberish_test_results.xlsx"
    df.to_excel(filename, index=False)
    
    print(f"\n=== Results saved to {filename} ===")
    print("Sample output for Hindi gibberish case:")
    hindi_case = df[df['Word'] == "केाीी"].iloc[0]
    print(f"Word: {hindi_case['Word']}")
    print(f"Error Message: {hindi_case['Error Message']}")

if __name__ == "__main__":
    load_dotenv()
    run_tests()
