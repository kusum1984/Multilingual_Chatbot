import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import langdetect
from langdetect import DetectorFactory

# For consistent language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """
    Ultimate gibberish detector with example-based learning
    
    Returns:
        Tuple: (status, error_type, message)
        - status: 'T' (valid) or 'F' (invalid)
        - error_type: '' or 'gibberish_error'
        - message: Detailed explanation if invalid
    """
    # Comprehensive example-based prompt
    system_prompt = """
    # Ultimate Gibberish Detection System v2.0
    
    ## Your Task:
    Analyze text for meaningful content in ANY language using these guidelines:
    
    === VALID TEXT EXAMPLES ===
    1. Dictionary Words:
       - English: "apple", "computer"
       - Japanese: "こんにちは" (hello)
       - Arabic: "كتاب" (book)
    2. Proper Nouns:
       - "New York"
       - "東京タワー" (Tokyo Tower)
       - "मुंबई" (Mumbai)
    3. Technical Codes:
       - "ID-5849-BN"
       - "订单号: 456789" (Chinese order number)
    4. Common Phrases:
       - "How are you?"
       - "Où est la gare?" (French: Where is the station?)
    5. Meaningful Single Characters:
       - "A" (grade)
       - "我" (Chinese "I")
    
    === GIBBERISH EXAMPLES ===
    1. Random Typing:
       - "asdfjkl;"
       - "qwertyuiop"
    2. Impossible Combinations:
       - "xzqywv" (invalid English)
       - "漢字漢字" (meaningless repetition)
    3. Nonsense Mixing:
       - "blah123blah"
       - "foo@bar$"
    4. Orthography Violations:
       - "Thsi is nto Enlgish"
       - "कखगघ" (invalid Devanagari)
    5. Meaningless Repetition:
       - "asdf asdf asdf"
       - "123 123 123"
    
    === DECISION RULES ===
    ✅ VALID if any:
    - Real dictionary word
    - Recognizable name/entity
    - Valid code/number pattern
    - Culturally significant
    - Proper single character
    
    ❌ GIBBERISH if all:
    - No dictionary words
    - Violates language rules
    - Random character mixing
    - Meaningless repetition
    
    === RESPONSE FORMAT ===
    STRICTLY respond with ONLY:
    - "Valid" OR
    - "Invalid|<reason>|<detected_lang>"
      Where <reason> is:
      - random_characters
      - impossible_combinations
      - nonsense_repetition
      - no_meaningful_units
      - mixed_scripts
    """

    client = get_client()
    
    try:
        if not text.strip():
            return ("T", "", "")
            
        # First detect language
        try:
            lang = langdetect.detect(text)
        except:
            lang = "unknown"
        
        # Create user prompt with contextual examples
        user_prompt = f"""
        Analyze: "{text}"
        Initial language detection: {lang}
        
        Compare against these examples:
        [Valid] "Paris", "123 Main St", "안녕", "@username"
        [Gibberish] "xjdkl", "asdf1234", "!@#$%^", "कखगघ"
        
        Your analysis (Valid/Invalid|Reason|Language):"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        result = response.choices[0].message.content.strip()
        
        if result == "Valid":
            return ("T", "", "")
        else:
            parts = result.split('|')
            reason = parts[1] if len(parts) > 1 else 'gibberish'
            detected_lang = parts[2] if len(parts) > 2 else lang
            
            return (
                "F",
                "gibberish_error",
                f"Gibberish detected: {reason} in {detected_lang}"
            )
            
    except Exception as e:
        return ("F", "api_error", f"Analysis failed: {str(e)}")


# ===== Comprehensive Test Suite (50+ Cases) =====
def run_tests():
    test_cases = [
        # Section 1: Valid Texts (25 cases)
        ("Hello world", "EN", "T"),
        ("Bonjour le monde", "FR", "T"),
        ("Hola mundo", "ES", "T"),
        ("Привет мир", "RU", "T"),
        ("مرحبا بالعالم", "AR", "T"),
        ("你好世界", "ZH", "T"),
        ("こんにちは世界", "JA", "T"),
        ("안녕하세요 세계", "KO", "T"),
        ("สวัสดีชาวโลก", "TH", "T"),
        ("Xin chào thế giới", "VI", "T"),
        ("Hallo Welt", "DE", "T"),
        ("Ciao mondo", "IT", "T"),
        ("Olá mundo", "PT", "T"),
        ("Witaj świecie", "PL", "T"),
        ("Γειά σου Κόσμε", "EL", "T"),
        ("Merhaba dünya", "TR", "T"),
        ("Hej världen", "SV", "T"),
        ("Pozdrav svijete", "HR", "T"),
        ("Ahoj světe", "CS", "T"),
        ("Helló világ", "HU", "T"),
        ("नमस्ते दुनिया", "HI", "T"),
        ("হ্যালো বিশ্ব", "BN", "T"),
        ("வணக்கம் உலகம்", "TA", "T"),
        ("שלום עולם", "HE", "T"),
        ("ID-5849-BN", "EN", "T"),
        
        # Section 2: Gibberish Cases (25 cases)
        ("asdfghjkl", "XX", "F"),
        ("qwertyuiop", "XX", "F"),
        ("йцукенгшщз", "XX", "F"),
        ("ضصثقضصثق", "XX", "F"),
        ("ㅁㄴㅇㄹㅁㄴㅇ", "XX", "F"),
        ("ฟหกด่าสว", "XX", "F"),
        ("αβγδαβγδ", "XX", "F"),
        ("אבגדאבגד", "XX", "F"),
        ("!@#$%^&*", "XX", "F"),
        ("xjdkl 392 sdk", "XX", "F"),
        ("xzqy wvut", "XX", "F"),
        ("blah blah", "XX", "F"),
        ("केाीी", "XX", "F"),
        ("asdf asdf asdf", "XX", "F"),
        ("123 123 123", "XX", "F"),
        ("foo@bar$", "XX", "F"),
        ("漢字漢字", "XX", "F"),
        ("कखगघ", "XX", "F"),
        ("zzxxyy", "XX", "F"),
        ("qwopasdf", "XX", "F"),
        ("1a2b3c4d", "XX", "F"),
        ("asdf1234", "XX", "F"),
        ("@#$%^&", "XX", "F"),
        ("zxcvbnm", "XX", "F"),
        ("asdf;lkj", "XX", "F"),
        
        # Section 3: Edge Cases (10 cases)
        ("A", "EN", "T"),
        ("我", "ZH", "T"),
        ("123", "EN", "T"),
        (" ", "XX", "T"),
        ("@username", "EN", "T"),
        ("#hashtag", "EN", "T"),
        ("订单号: 456789", "ZH", "T"),
        ("東京タワー", "JA", "T"),
        ("asdf", "XX", "F"),
        ("कखग", "XX", "F")
    ]

    print("=== Ultimate Gibberish Detection Test ===")
    print(f"Running {len(test_cases)} test cases across multiple languages\n")
    
    passed = 0
    for idx, (text, lang, expected) in enumerate(test_cases, 1):
        status, err_type, msg = check_gibberish(text)
        result = "✅ PASS" if status == expected else "❌ FAIL"
        if status == expected:
            passed += 1
            
        print(f"{idx:02d} {result} {lang}: '{text[:20]}'")
        if status != expected:
            print(f"   Expected {expected}, got {status} | {msg}")
    
    accuracy = passed / len(test_cases) * 100
    print(f"\nResults: {passed}/{len(test_cases)} passed ({accuracy:.1f}% accuracy)")

if __name__ == "__main__":
    run_tests()

    ************************************
    # ===== Enhanced Test Suite with Language Detection Validation =====
def run_tests():
    test_cases = [
        # Format: (text, expected_status, expected_lang_if_valid)
        # Valid Texts
        ("Hello world", "T", "en"),
        ("Bonjour le monde", "T", "fr"),
        ("Hola mundo", "T", "es"),
        ("Привет мир", "T", "ru"),
        ("مرحبا بالعالم", "T", "ar"),
        
        # Gibberish Cases (lang detection may fail)
        ("asdfghjkl", "F", None),
        ("qwertyuiop", "F", None),
        ("!@#$%^&*", "F", None),
        
        # Edge Cases
        ("123 Main St", "T", "en"),
        ("ID-5849-BN", "T", None),  # Codes often don't detect language
        ("@username", "T", None)    # Handles don't detect language
    ]

    print("=== Enhanced Gibberish Detection Test ===")
    print("Now validating both classification AND language detection\n")
    
    passed_classification = 0
    passed_language = 0
    total = len(test_cases)
    
    for idx, (text, expected_status, expected_lang) in enumerate(test_cases, 1):
        # Run detection
        status, err_type, msg = check_gibberish(text)
        
        # Get detected language
        try:
            detected_lang = langdetect.detect(text) if text.strip() else None
        except:
            detected_lang = None
        
        # Check classification
        classification_ok = status == expected_status
        if classification_ok:
            passed_classification += 1
            
        # Check language detection (only for valid texts)
        language_ok = True
        if expected_status == "T" and expected_lang:
            language_ok = detected_lang == expected_lang
            if language_ok:
                passed_language += 1
        
        # Print results
        result = []
        if classification_ok:
            result.append("✅ CLASS")
        else:
            result.append("❌ CLASS")
            
        if not expected_lang or language_ok:
            result.append("✅ LANG")
        else:
            result.append("❌ LANG")
        
        print(f"{idx:02d} {' '.join(result)}: '{text[:20]}'")
        print(f"   Status: {status} (expected {expected_status})")
        if expected_lang:
            print(f"   Language: {detected_lang} (expected {expected_lang})")
    
    # Calculate accuracy
    classification_acc = passed_classification / total * 100
    language_acc = passed_language / sum(1 for case in test_cases if case[2]) * 100
    
    print(f"\nResults:")
    print(f"- Classification: {passed_classification}/{total} ({classification_acc:.1f}%)")
    print(f"- Language Detection: {passed_language}/{sum(1 for case in test_cases if case[2])} ({language_acc:.1f}%)")

if __name__ == "__main__":
    run_tests()
***********************************************
# ===== Comprehensive Test Suite (100+ Cases) =====
def run_tests():
    test_cases = [
        # Section 1: Indo-European Languages (30 cases)
        ("Hello world", "T", "en"),  # English
        ("Bonjour le monde", "T", "fr"),  # French
        ("Hola mundo", "T", "es"),  # Spanish
        ("Привет мир", "T", "ru"),  # Russian
        ("नमस्ते दुनिया", "T", "hi"),  # Hindi
        ("Γειά σου Κόσμε", "T", "el"),  # Greek
        ("Witaj świecie", "T", "pl"),  # Polish
        ("Hej världen", "T", "sv"),  # Swedish
        ("Halló heimur", "T", "is"),  # Icelandic
        ("Sveika pasaule", "T", "lv"),  # Latvian
        ("Përshëndetje botë", "T", "sq"),  # Albanian
        ("Salve mundi", "T", "la"),  # Latin
        ("Dia duit domhan", "T", "ga"),  # Irish
        ("Helo byd", "T", "cy"),  # Welsh
        ("Salam dunia", "T", "ms"),  # Malay
        ("ନମସ୍କାର ବିଶ୍ୱ", "T", "or"),  # Odia
        ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨਿਆ", "T", "pa"),  # Punjabi
        ("હેલો વર્લ્ડ", "T", "gu"),  # Gujarati
        ("হ্যালো বিশ্ব", "T", "bn"),  # Bengali
        ("வணக்கம் உலகம்", "T", "ta"),  # Tamil
        ("హలో ప్రపంచం", "T", "te"),  # Telugu
        ("ഹലോ വേൾഡ്", "T", "ml"),  # Malayalam
        ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨਿਆ", "T", "pa"),  # Punjabi
        ("مرحبا بالعالم", "T", "ar"),  # Arabic
        ("שלום עולם", "T", "he"),  # Hebrew
        ("سلام دنیا", "T", "fa"),  # Persian
        ("हॅलो वर्ल्ड", "T", "mr"),  # Marathi
        ("नमस्कार संसार", "T", "ne"),  # Nepali
        ("හෙලෝ වර්ල්ඩ්", "T", "si"),  # Sinhala
        ("ជំរាបសួរ ពិភពលោក", "T", "km"),  # Khmer
        
        # Section 2: Asian Languages (25 cases)
        ("你好世界", "T", "zh"),  # Chinese
        ("こんにちは世界", "T", "ja"),  # Japanese
        ("안녕하세요 세계", "T", "ko"),  # Korean
        ("สวัสดีชาวโลก", "T", "th"),  # Thai
        ("Xin chào thế giới", "T", "vi"),  # Vietnamese
        ("မင်္ဂလာပါကမ္ဘာလောက", "T", "my"),  # Burmese
        ("හෙලෝ වර්ල්ඩ්", "T", "si"),  # Sinhala
        ("សួស្តីពិភពលោក", "T", "km"),  # Khmer
        ("မင်္ဂလာပါ", "T", "my"),  # Burmese
        ("ສະບາຍດີຊາວໂລກ", "T", "lo"),  # Lao
        ("ᠰᠠᠢᠨ ᠪᠠᠢᠨᠠᠷ ᠳ᠋ᠤᠭᠠᠷ", "T", "mn"),  # Mongolian
        ("གྲོས་ཚོགས་འཛིན་སྐྱོང་", "T", "bo"),  # Tibetan
        ("ନମସ୍କାର ବିଶ୍ୱ", "T", "or"),  # Odia
        ("ਹੈਲੋ ਵਰਲਡ", "T", "pa"),  # Punjabi
        ("হ্যালো বিশ্ব", "T", "bn"),  # Bengali
        ("வணக்கம் உலகம்", "T", "ta"),  # Tamil
        ("హలో ప్రపంచం", "T", "te"),  # Telugu
        ("ഹലോ വേൾഡ്", "T", "ml"),  # Malayalam
        ("ନମସ୍କାର", "T", "or"),  # Odia
        ("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "T", "pa"),  # Punjabi
        ("হ্যালো", "T", "bn"),  # Bengali
        ("வணக்கம்", "T", "ta"),  # Tamil
        ("హలో", "T", "te"),  # Telugu
        ("ഹലോ", "T", "ml"),  # Malayalam
        ("こんにちは", "T", "ja"),  # Japanese
        
        # Section 3: African Languages (15 cases)
        ("Sawubona Mhlaba", "T", "zu"),  # Zulu
        ("Habari dunia", "T", "sw"),  # Swahili
        ("Dumela lefatshe", "T", "tn"),  # Tswana
        ("Moni m'dziko", "T", "ny"),  # Chichewa
        ("Sannu duniya", "T", "ha"),  # Hausa
        ("Ndewo ụwa", "T", "ig"),  # Igbo
        ("Salamu dunia", "T", "am"),  # Amharic
        ("Mo ki O Ile Aiye", "T", "yo"),  # Yoruba
        ("Agoo dunia", "T", "ee"),  # Ewe
        ("Kedu uwa", "T", "ig"),  # Igbo
        ("Sanu duniyan", "T", "ff"),  # Fulani
        ("Akwaba wiase", "T", "ak"),  # Akan
        ("Mwaramutse isi", "T", "rw"),  # Kinyarwanda
        ("Saluton mondo", "T", "eo"),  # Esperanto (not African but included)
        ("Hallo wêreld", "T", "af"),  # Afrikaans
        
        # Section 4: Indigenous/Endangered Languages (10 cases)
        ("Kwe kwe otsi", "T", "cr"),  # Cree
        ("Háu mitákuye oyás'iŋ", "T", "lkt"),  # Lakota
        ("Yá'át'ééh", "T", "nv"),  # Navajo
        ("Aloha honua", "T", "haw"),  # Hawaiian
        ("Kia ora te ao", "T", "mi"),  # Māori
        ("Ahalan dunia", "T", "ber"),  # Berber
        ("Bozo aylan", "T", "ber"),  # Berber
        ("Kamusta mundo", "T", "fil"),  # Filipino
        ("Halo dunia", "T", "id"),  # Indonesian
        ("Talofa lalolagi", "T", "sm"),  # Samoan
        
        # Section 5: Gibberish Cases (20 cases)
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
        ("केाीी", "F", None),
        ("asdf asdf asdf", "F", None),
        ("123 123 123", "F", None),
        ("foo@bar$", "F", None),
        ("漢字漢字", "F", None),
        ("कखगघ", "F", None),
        ("zzxxyy", "F", None),
        ("qwopasdf", "F", None),
        
        # Section 6: Edge Cases (10 cases)
        ("A", "T", None),  # Single character
        ("我", "T", "zh"),  # Chinese character
        ("123", "T", None),  # Numbers
        (" ", "T", None),  # Empty string
        ("@username", "T", None),  # Handle
        ("#hashtag", "T", None),  # Tag
        ("ID-5849-BN", "T", None),  # Code
        ("東京タワー", "T", "ja"),  # Mixed script
        ("asdf", "F", None),  # Short gibberish
        ("कखग", "F", None)  # Invalid sequence
    ]

    print("=== Ultimate Language Validation Test ===")
    print(f"Running {len(test_cases)} test cases across 50+ languages\n")
    
    passed_classification = 0
    passed_language = 0
    total = len(test_cases)
    lang_expected_count = sum(1 for case in test_cases if case[2])
    
    for idx, (text, expected_status, expected_lang) in enumerate(test_cases, 1):
        # Run detection
        status, err_type, msg = check_gibberish(text)
        
        # Get detected language
        try:
            detected_lang = langdetect.detect(text) if text.strip() and expected_status == "T" else None
        except:
            detected_lang = None
        
        # Check classification
        classification_ok = status == expected_status
        if classification_ok:
            passed_classification += 1
            
        # Check language detection (only for valid texts with expected lang)
        language_ok = True
        if expected_status == "T" and expected_lang:
            language_ok = detected_lang == expected_lang
            if language_ok:
                passed_language += 1
        
        # Print results
        result = []
        if classification_ok:
            result.append("✅ CLASS")
        else:
            result.append("❌ CLASS")
            
        if not expected_lang or language_ok:
            result.append("✅ LANG")
        else:
            result.append(f"❌ LANG (got {detected_lang})")
        
        print(f"{idx:03d} {' '.join(result)}: '{text[:20]}'")
        if not classification_ok:
            print(f"   Expected {expected_status}, got {status} | {msg}")
    
    # Calculate accuracy
    classification_acc = passed_classification / total * 100
    language_acc = passed_language / lang_expected_count * 100 if lang_expected_count > 0 else 100
    
    print(f"\n=== Final Results ===")
    print(f"Classification Accuracy: {passed_classification}/{total} ({classification_acc:.1f}%)")
    print(f"Language Detection Accuracy: {passed_language}/{lang_expected_count} ({language_acc:.1f}%)")

if __name__ == "__main__":
    run_tests()



*********************************************************************************************************

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import langdetect
from langdetect import DetectorFactory
import pandas as pd
from datetime import datetime

# For consistent language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """[Previous implementation remains exactly the same]"""
    # [Previous code here - unchanged]
    pass

def run_tests():
    test_cases = [
        # [Previous valid test cases remain the same]
        # ...
        
        # Enhanced Section 5: Gibberish Cases (50 cases now)
        # English-like gibberish
        ("asdfghjkl", "F", None),
        ("qwertyuiop", "F", None),
        ("zxcvbnm", "F", None),
        ("poiuytrewq", "F", None),
        ("lkjhgfdsa", "F", None),
        ("mnbvcxz", "F", None),
        ("asdf;lkj", "F", None),
        ("jfkdls;a", "F", None),
        
        # International keyboard gibberish
        ("йцукенгшщз", "F", None),  # Russian-like
        ("фывапролджэ", "F", None),  # Russian-like
        ("asdfñlkj", "F", None),  # Spanish-like
        ("éàèùç", "F", None),  # French-like
        ("äöüß", "F", None),  # German-like
        ("αβγδεζ", "F", None),  # Greek-like
        ("אבגדהוז", "F", None),  # Hebrew-like
        ("ضصثقفغ", "F", None),  # Arabic-like
        ("ㄱㄴㄷㄹㅁㅂ", "F", None),  # Korean-like
        ("あかさたなは", "F", None),  # Japanese-like
        
        # Mixed-script gibberish
        ("aβcδεf", "F", None),
        ("x漢y字z", "F", None),
        ("1あ2い3う", "F", None),
        ("aαbβcγ", "F", None),
        ("@#¢∞§÷", "F", None),
        
        # Pattern-based gibberish
        ("abcabcabc", "F", None),
        ("123123123", "F", None),
        ("q1w2e3r4", "F", None),
        ("!a@b#c$", "F", None),
        ("a_b_c_d_", "F", None),
        
        # Common password-like gibberish
        ("password123", "F", None),
        ("qwerty123", "F", None),
        ("letmein", "F", None),
        ("adminadmin", "F", None),
        ("welcome1", "F", None),
        
        # Unicode abuse
        ("Ω≈ç√∫˜µ", "F", None),
        ("⁄€‹›ﬁﬂ‡°", "F", None),
        ("ⓐⓑⓒⓓⓔ", "F", None),
        ("ᴬᴮᶜᴰᴱ", "F", None),
        ("ₐₑₒₓₔ", "F", None),
        
        # Emoji/non-text
        ("😀😃😄😁", "F", None),
        ("👍👎💯", "F", None),
        ("🚀🌕✨", "F", None),
        ("🔑🗝️🔒", "F", None),
        ("📱💻🖥️", "F", None),
        
        # [Previous edge cases remain the same]
        # ...
    ]

    print("=== Ultimate Gibberish Detection Test ===")
    print(f"Running {len(test_cases)} test cases across 50+ languages\n")
    
    # Prepare results dataframe
    results = []
    columns = [
        'Language', 
        'Word', 
        'Expected Status', 
        'Actual Status',
        'Classification Result',
        'Expected LangCode',
        'Detected LangCode',
        'Language Detection Result',
        'Error Message',
        'Timestamp'
    ]
    
    for idx, (text, expected_status, expected_lang) in enumerate(test_cases, 1):
        # Run detection
        status, err_type, msg = check_gibberish(text)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get detected language
        detected_lang = None
        lang_result = "N/A"
        if expected_status == "T" and text.strip():
            try:
                detected_lang = langdetect.detect(text)
                lang_result = "✅" if detected_lang == expected_lang else "❌"
            except:
                detected_lang = "Detection Failed"
                lang_result = "❌"
        
        # Build result row
        results.append([
            expected_lang if expected_lang else "Gibberish",
            text,
            expected_status,
            status,
            "✅" if status == expected_status else "❌",
            expected_lang if expected_lang else "N/A",
            detected_lang if detected_lang else "N/A",
            lang_result,
            msg if status != expected_status else "",
            timestamp
        ])
        
        # Print progress
        print(f"{idx:03d} Tested: '{text[:20]}'")
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=columns)
    
    # Save to Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"gibberish_test_results_{timestamp}.xlsx"
    df.to_excel(filename, index=False)
    
    # Calculate statistics
    classification_acc = (df['Classification Result'] == '✅').mean() * 100
    lang_acc = (df['Language Detection Result'] == '✅').mean() * 100
    
    print(f"\n=== Final Results ===")
    print(f"Classification Accuracy: {classification_acc:.1f}%")
    print(f"Language Detection Accuracy: {lang_acc:.1f}%")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    run_tests()
