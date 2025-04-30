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

==================================
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Language-specific error messages with the exact required format
ERROR_MESSAGES = {
    'EN': "Langcode-EN expected error - The given English word is nonsense.",
    'FR': "Langcode-FR expected error - Le mot français donné est un non-sens.",
    'ES': "Langcode-ES expected error - El texto en español no tiene sentido.",
    'RU': "Langcode-RU expected error - Данное русское слово бессмысленно.",
    'AR': "Langcode-AR expected error - الكلمة العربية المعطاة غير منطقية.",
    'ZH': "Langcode-ZH expected error - 中文文本是乱码。",
    'JA': "Langcode-JA expected error - 日本語のテキストは無意味です。",
    'KO': "Langcode-KO expected error - 주어진 한국어 단어는 무의미합니다.",
    'HI': "Langcode-HI expected error - दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'BN': "Langcode-BN expected error - প্রদত্ত বাংলা শব্দটি অর্থহীন।",
    'PA': "Langcode-PA expected error - ਦਿੱਤਾ ਪੰਜਾਬੀ ਸ਼ਬਦ ਬਕਵਾਸ ਹੈ।",
    'TA': "Langcode-TA expected error - கொடுக்கப்பட்ட தமிழ் சொல் அர்த்தமற்றது.",
    'TE': "Langcode-TE expected error - ఇచ్చిన తెలుగు పదం అర్థంలేనిది.",
    'MR': "Langcode-MR expected error - दिलेले मराठी शब्द निरर्थक आहे.",
    'UR': "Langcode-UR expected error - دیا گیا اردو لفظ بکواس ہے۔",
    'GU': "Langcode-GU expected error - આપેલ ગુજરાતી શબ્દ નિરર્થક છે.",
    'KN': "Langcode-KN expected error - ನೀಡಿದ ಕನ್ನಡ ಪದವು ಅರ್ಥಹೀನವಾಗಿದೆ.",
    'OR': "Langcode-OR expected error - ଦିଆଯାଇଥିବା ଓଡିଆ ଶବ୍ଦ ଅର୍ଥହୀନ।",
    'ML': "Langcode-ML expected error - നൽകിയ മലയാളം വാക്ക് അർത്ഥശൂന്യമാണ്.",
    'DE': "Langcode-DE expected error - Das gegebene deutsche Wort ist sinnlos.",
    'IT': "Langcode-IT expected error - La parola italiana data non ha senso.",
    'PT': "Langcode-PT expected error - O texto em português é nonsense.",
    'PL': "Langcode-PL expected error - Podane polskie słowo jest bez sensu.",
    'TR': "Langcode-TR expected error - Verilen Türkçe kelime anlamsızdır.",
    'NL': "Langcode-NL expected error - Het gegeven Nederlandse woord is onzin.",
    'SV': "Langcode-SV expected error - Det givna svenska ordet är nonsens.",
    'FI': "Langcode-FI expected error - Annettu suomenkielinen sana on järjetön.",
    'DA': "Langcode-DA expected error - Det givne danske ord er nonsens.",
    'NO': "Langcode-NO expected error - Det gitte norske ordet er nonsens.",
    'HE': "Langcode-HE expected error - המילה העברית הנתונה היא חסרת משמעות.",
    'FA': "Langcode-FA expected error - کلمه فارسی داده شده بی معنی است.",
    'TH': "Langcode-TH expected error - คำภาษาไทยที่ให้มานั้นไร้ความหมาย",
    'VI': "Langcode-VI expected error - Từ tiếng Việt đã cho là vô nghĩa.",
    'ID': "Langcode-ID expected error - Kata bahasa Indonesia yang diberikan tidak masuk akal.",
    'MS': "Langcode-MS expected error - Perkataan Melayu yang diberikan tidak bermakna.",
    'FIL': "Langcode-FIL expected error - Ang ibinigay na salitang Filipino ay walang kahulugan.",
    'SW': "Langcode-SW expected error - Neno la Kiswahili lililopewa halina maana.",
    'HA': "Langcode-HA expected error - Kalmar Hausa da aka bayar ba ta da ma'ana.",
    'YO': "Langcode-YO expected error - Ọrọ Yoruba ti a fun ni alailẹgbẹ.",
    'IG': "Langcode-IG expected error - Okwu Igbo enyere enweghị isi.",
    'ZU': "Langcode-ZU expected error - Igama lesiZulu elinikeziwe alinalo ukuqonda.",
    'XH': "Langcode-XH expected error - Igama lesiXhosa elinikiweyo alinalo nto ithethayo.",
    'ST': "Langcode-ST expected error - Lentsoe la Sesotho le fanoeng ha le na moelelo.",
    'SN': "Langcode-SN expected error - Izwi reShona rakapihwa harina revo.",
    'AM': "Langcode-AM expected error - የተሰጠው አማርኛ ቃል ምንም ትርጉም የለውም።",
    'SO': "Langcode-SO expected error - Erayga Soomaaliga la siiyay ma lahan macno.",
    'HAW': "Langcode-HAW expected error - ʻŌlelo Hawaiʻi i hāʻawi ʻia ʻaʻohe manaʻo.",
    'MI': "Langcode-MI expected error - Ko te kupu Māori i hoatu kaore he tikanga.",
    'SM': "Langcode-SM expected error - O le upu Samoa na tuʻuina atu e leai se uiga.",
    'TO': "Langcode-TO expected error - Ko e lea faka-Tonga naʻe foaki ʻoku ʻikai ha ʻuhinga.",
    'FJ': "Langcode-FJ expected error - Na vosa vakaviti e solia e sega ni ibalebale.",
    'EL': "Langcode-EL expected error - Η δοθείσα ελληνική λέξη δεν έχει νόημα.",
    'HU': "Langcode-HU expected error - Az adott magyar szó értelmetlen.",
    'CS': "Langcode-CS expected error - Dané české slovo je nesmysl.",
    'SK': "Langcode-SK expected error - Dané slovenské slovo je nezmysel.",
    'HR': "Langcode-HR expected error - Dana hrvatska riječ je besmislena.",
    'SR': "Langcode-SR expected error - Data srpska reč je besmislena.",
    'SL': "Langcode-SL expected error - Dana slovenska beseda je nesmiselna.",
    'BG': "Langcode-BG expected error - Дадената българска дума е безсмислена.",
    'UK': "Langcode-UK expected error - Дане українське слово не має сенсу.",
    'BE': "Langcode-BE expected error - Дадзенае беларускае слова бессэнсоўнае.",
    'KK': "Langcode-KK expected error - Берілген қазақ сөзі мағынасыз.",
    'UZ': "Langcode-UZ expected error - Berilgan o'zbekcha so'z mavhum.",
    'KY': "Langcode-KY expected error - Берилген кыргыз сөзү маанисиз.",
    'TG': "Langcode-TG expected error - Калимаи тоҷикӣ додашуда бе маъно аст.",
    'MN': "Langcode-MN expected error - Өгсөн монгол үг утгагүй.",
    'BO': "Langcode-BO expected error - བོད་སྐད་ཀྱི་ཚིག་དེ་དོན་མེད་པ་རེད།",
    'NE': "Langcode-NE expected error - दिइएको नेपाली शब्द अर्थहीन छ।",
    'SI': "Langcode-SI expected error - දෙන ලද සිංහල වචනය අර්ථ විරහිතය.",
    'KM': "Langcode-KM expected error - ពាក្យខ្មែរដែលបានផ្តល់គឺគ្មានន័យ។",
    'LO': "Langcode-LO expected error - ຄຳລາວທີ່ໃຫ້ແມ່ນບໍ່ມີຄວາມຫມາຍ.",
    'MY': "Langcode-MY expected error - ပေးထားသော မြန်မာစကားလုံးသည် အဓိပ္ပါယ်မရှိပါ။",
    'KA': "Langcode-KA expected error - მოცემული ქართული სიტყვა უაზროა.",
    'HY': "Langcode-HY expected error - Տրված հայերեն բառն անիմաստ է։",
    'AZ': "Langcode-AZ expected error - Verilən Azərbaycan sözü mənasızdır.",
    'TK': "Langcode-TK expected error - Berlen Türkmen sözi manyşyz.",
    'ET': "Langcode-ET expected error - Antud eesti sõna on mõttetu.",
    'LV': "Langcode-LV expected error - Dotais latviešu vārds ir bezjēdzīgs.",
    'LT': "Langcode-LT expected error - Pateiktas lietuvių kalbos žodis yra beprasmiškas.",
    'CY': "Langcode-CY expected error - Mae'r gair Cymraeg a roddwyd yn ddiystyr.",
    'GA': "Langcode-GA expected error - Tá an focal Gaeilge a tugadh gan bhrí.",
    'GD': "Langcode-GD expected error - Tha am facal Gàidhlig a chaidh a thoirt seachad gun bhrìgh.",
    'MT': "Langcode-MT expected error - Il-kelma Maltija mogħtija hija bla sens.",
    'EU': "Langcode-EU expected error - Emandako euskal hitza zentzugabea da.",
    'CA': "Langcode-CA expected error - La paraula catalana donada no té sentit.",
    'GL': "Langcode-GL expected error - A palabra galega dada non ten sentido.",
    'AF': "Langcode-AF expected error - Die gegewe Afrikaanse woord is nonsens.",
    'IS': "Langcode-IS expected error - Gefið íslenskt orð er tilgangslaust.",
    'FO': "Langcode-FO expected error - Heta føroyska orðið er menningarlaust.",
    'SA': "Langcode-SA expected error - दत्तः संस्कृतशब्दः निरर्थकः अस्ति।",
    'LA': "Langcode-LA expected error - Verbum Latinum datum inane est.",
    'EO': "Langcode-EO expected error - La donita Esperanta vorto estas sensenca.",
    'DEFAULT': "Langcode-XX expected error - The text appears to be gibberish."
}

# Rest of your code remains exactly the same...
# [Keep all other functions and logic unchanged from previous implementation]

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

++++++++++++++++++++++


import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Language-specific error messages with the exact required format
ERROR_MESSAGES = {
    'EN': "Langcode-EN expected error - The given English word is nonsense.",
    'FR': "Langcode-FR expected error - Le mot français donné est un non-sens.",
    'ES': "Langcode-ES expected error - El texto en español no tiene sentido.",
    'RU': "Langcode-RU expected error - Данное русское слово бессмысленно.",
    'AR': "Langcode-AR expected error - الكلمة العربية المعطاة غير منطقية.",
    'ZH': "Langcode-ZH expected error - 中文文本是乱码。",
    'JA': "Langcode-JA expected error - 日本語のテキストは無意味です。",
    'KO': "Langcode-KO expected error - 주어진 한국어 단어는 무의미합니다.",
    'HI': "Langcode-HI expected error - दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'BN': "Langcode-BN expected error - প্রদত্ত বাংলা শব্দটি অর্থহীন।",
    'PA': "Langcode-PA expected error - ਦਿੱਤਾ ਪੰਜਾਬੀ ਸ਼ਬਦ ਬਕਵਾਸ ਹੈ।",
    'TA': "Langcode-TA expected error - கொடுக்கப்பட்ட தமிழ் சொல் அர்த்தமற்றது.",
    'TE': "Langcode-TE expected error - ఇచ్చిన తెలుగు పదం అర్థంలేనిది.",
    'MR': "Langcode-MR expected error - दिलेले मराठी शब्द निरर्थक आहे.",
    'UR': "Langcode-UR expected error - دیا گیا اردو لفظ بکواس ہے۔",
    'GU': "Langcode-GU expected error - આપેલ ગુજરાતી શબ્દ નિરર્થક છે.",
    'KN': "Langcode-KN expected error - ನೀಡಿದ ಕನ್ನಡ ಪದವು ಅರ್ಥಹೀನವಾಗಿದೆ.",
    'OR': "Langcode-OR expected error - ଦିଆଯାଇଥିବା ଓଡିଆ ଶବ୍ଦ ଅର୍ଥହୀନ।",
    'ML': "Langcode-ML expected error - നൽകിയ മലയാളം വാക്ക് അർത്ഥശൂന്യമാണ്.",
    'DE': "Langcode-DE expected error - Das gegebene deutsche Wort ist sinnlos.",
    'IT': "Langcode-IT expected error - La parola italiana data non ha senso.",
    'PT': "Langcode-PT expected error - O texto em português é nonsense.",
    'PL': "Langcode-PL expected error - Podane polskie słowo jest bez sensu.",
    'TR': "Langcode-TR expected error - Verilen Türkçe kelime anlamsızdır.",
    'NL': "Langcode-NL expected error - Het gegeven Nederlandse woord is onzin.",
    'SV': "Langcode-SV expected error - Det givna svenska ordet är nonsens.",
    'FI': "Langcode-FI expected error - Annettu suomenkielinen sana on järjetön.",
    'DA': "Langcode-DA expected error - Det givne danske ord er nonsens.",
    'NO': "Langcode-NO expected error - Det gitte norske ordet er nonsens.",
    'HE': "Langcode-HE expected error - המילה העברית הנתונה היא חסרת משמעות.",
    'FA': "Langcode-FA expected error - کلمه فارسی داده شده بی معنی است.",
    'TH': "Langcode-TH expected error - คำภาษาไทยที่ให้มานั้นไร้ความหมาย",
    'VI': "Langcode-VI expected error - Từ tiếng Việt đã cho là vô nghĩa.",
    'ID': "Langcode-ID expected error - Kata bahasa Indonesia yang diberikan tidak masuk akal.",
    'MS': "Langcode-MS expected error - Perkataan Melayu yang diberikan tidak bermakna.",
    'FIL': "Langcode-FIL expected error - Ang ibinigay na salitang Filipino ay walang kahulugan.",
    'SW': "Langcode-SW expected error - Neno la Kiswahili lililopewa halina maana.",
    'HA': "Langcode-HA expected error - Kalmar Hausa da aka bayar ba ta da ma'ana.",
    'YO': "Langcode-YO expected error - Ọrọ Yoruba ti a fun ni alailẹgbẹ.",
    'IG': "Langcode-IG expected error - Okwu Igbo enyere enweghị isi.",
    'ZU': "Langcode-ZU expected error - Igama lesiZulu elinikeziwe alinalo ukuqonda.",
    'XH': "Langcode-XH expected error - Igama lesiXhosa elinikiweyo alinalo nto ithethayo.",
    'ST': "Langcode-ST expected error - Lentsoe la Sesotho le fanoeng ha le na moelelo.",
    'SN': "Langcode-SN expected error - Izwi reShona rakapihwa harina revo.",
    'AM': "Langcode-AM expected error - የተሰጠው አማርኛ ቃል ምንም ትርጉም የለውም።",
    'SO': "Langcode-SO expected error - Erayga Soomaaliga la siiyay ma lahan macno.",
    'HAW': "Langcode-HAW expected error - ʻŌlelo Hawaiʻi i hāʻawi ʻia ʻaʻohe manaʻo.",
    'MI': "Langcode-MI expected error - Ko te kupu Māori i hoatu kaore he tikanga.",
    'SM': "Langcode-SM expected error - O le upu Samoa na tuʻuina atu e leai se uiga.",
    'TO': "Langcode-TO expected error - Ko e lea faka-Tonga naʻe foaki ʻoku ʻikai ha ʻuhinga.",
    'FJ': "Langcode-FJ expected error - Na vosa vakaviti e solia e sega ni ibalebale.",
    'EL': "Langcode-EL expected error - Η δοθείσα ελληνική λέξη δεν έχει νόημα.",
    'HU': "Langcode-HU expected error - Az adott magyar szó értelmetlen.",
    'CS': "Langcode-CS expected error - Dané české slovo je nesmysl.",
    'SK': "Langcode-SK expected error - Dané slovenské slovo je nezmysel.",
    'HR': "Langcode-HR expected error - Dana hrvatska riječ je besmislena.",
    'SR': "Langcode-SR expected error - Data srpska reč je besmislena.",
    'SL': "Langcode-SL expected error - Dana slovenska beseda je nesmiselna.",
    'BG': "Langcode-BG expected error - Дадената българска дума е безсмислена.",
    'UK': "Langcode-UK expected error - Дане українське слово не має сенсу.",
    'BE': "Langcode-BE expected error - Дадзенае беларускае слова бессэнсоўнае.",
    'KK': "Langcode-KK expected error - Берілген қазақ сөзі мағынасыз.",
    'UZ': "Langcode-UZ expected error - Berilgan o'zbekcha so'z mavhum.",
    'KY': "Langcode-KY expected error - Берилген кыргыз сөзү маанисиз.",
    'TG': "Langcode-TG expected error - Калимаи тоҷикӣ додашуда бе маъно аст.",
    'MN': "Langcode-MN expected error - Өгсөн монгол үг утгагүй.",
    'BO': "Langcode-BO expected error - བོད་སྐད་ཀྱི་ཚིག་དེ་དོན་མེད་པ་རེད།",
    'NE': "Langcode-NE expected error - दिइएको नेपाली शब्द अर्थहीन छ।",
    'SI': "Langcode-SI expected error - දෙන ලද සිංහල වචනය අර්ථ විරහිතය.",
    'KM': "Langcode-KM expected error - ពាក្យខ្មែរដែលបានផ្តល់គឺគ្មានន័យ។",
    'LO': "Langcode-LO expected error - ຄຳລາວທີ່ໃຫ້ແມ່ນບໍ່ມີຄວາມຫມາຍ.",
    'MY': "Langcode-MY expected error - ပေးထားသော မြန်မာစကားလုံးသည် အဓိပ္ပါယ်မရှိပါ။",
    'KA': "Langcode-KA expected error - მოცემული ქართული სიტყვა უაზროა.",
    'HY': "Langcode-HY expected error - Տրված հայերեն բառն անիմաստ է։",
    'AZ': "Langcode-AZ expected error - Verilən Azərbaycan sözü mənasızdır.",
    'TK': "Langcode-TK expected error - Berlen Türkmen sözi manyşyz.",
    'ET': "Langcode-ET expected error - Antud eesti sõna on mõttetu.",
    'LV': "Langcode-LV expected error - Dotais latviešu vārds ir bezjēdzīgs.",
    'LT': "Langcode-LT expected error - Pateiktas lietuvių kalbos žodis yra beprasmiškas.",
    'CY': "Langcode-CY expected error - Mae'r gair Cymraeg a roddwyd yn ddiystyr.",
    'GA': "Langcode-GA expected error - Tá an focal Gaeilge a tugadh gan bhrí.",
    'GD': "Langcode-GD expected error - Tha am facal Gàidhlig a chaidh a thoirt seachad gun bhrìgh.",
    'MT': "Langcode-MT expected error - Il-kelma Maltija mogħtija hija bla sens.",
    'EU': "Langcode-EU expected error - Emandako euskal hitza zentzugabea da.",
    'CA': "Langcode-CA expected error - La paraula catalana donada no té sentit.",
    'GL': "Langcode-GL expected error - A palabra galega dada non ten sentido.",
    'AF': "Langcode-AF expected error - Die gegewe Afrikaanse woord is nonsens.",
    'IS': "Langcode-IS expected error - Gefið íslenskt orð er tilgangslaust.",
    'FO': "Langcode-FO expected error - Heta føroyska orðið er menningarlaust.",
    'SA': "Langcode-SA expected error - दत्तः संस्कृतशब्दः निरर्थकः अस्ति।",
    'LA': "Langcode-LA expected error - Verbum Latinum datum inane est.",
    'EO': "Langcode-EO expected error - La donita Esperanta vorto estas sensenca.",
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
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code) if lang_code else "Gibberish"

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def process_gibberish_response(response_text: str) -> Tuple[str, str, str]:
    """
    Process the response from the gibberish detection API
    Returns: (status, error_type, message)
    """
    if response_text == "Valid":
        return ("T", "", "")
    
    parts = response_text.split('|')
    if len(parts) >= 3:
        lang_code = parts[1].upper()
        reason = parts[2]
        error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
        return ("F", "gibberish_error", error_msg)
    
    return ("F", "gibberish_error", ERROR_MESSAGES['DEFAULT'])

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
        
        user_prompt = f"""
        Analyze: "{text}"
        
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
        return process_gibberish_response(result)
            
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
        'Language Name',
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
            expected_lang if expected_lang else "XX",
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
    print(f"Language: {hindi_case['Language']}")
    print(f"Language Name: {hindi_case['Language Name']}")

if __name__ == "__main__":
    load_dotenv()
    run_tests()
**********************************************************


***************************************************
        import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Language-specific error messages with the exact required format
ERROR_MESSAGES = {
    'EN': "The given English word is nonsense.",
    'FR': "Le mot français donné est un non-sens.",
    'ES': "El texto en español no tiene sentido.",
    'RU': "Данное русское слово бессмысленно.",
    'AR': "الكلمة العربية المعطاة غير منطقية.",
    'ZH': "中文文本是乱码。",
    'JA': "日本語のテキストは無意味です。",
    'KO': "주어진 한국어 단어는 무의미합니다.",
    'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'BN': "প্রদত্ত বাংলা শব্দটি অর্থহীন।",
    'PA': "ਦਿੱਤਾ ਪੰਜਾਬੀ ਸ਼ਬਦ ਬਕਵਾਸ ਹੈ।",
    'TA': "கொடுக்கப்பட்ட தமிழ் சொல் அர்த்தமற்றது.",
    'TE': "ఇచ్చిన తెలుగు పదం అర్థంలేనిది.",
    'MR': "दिलेले मराठी शब्द निरर्थक आहे.",
    'UR': "دیا گیا اردو لفظ بکواس ہے۔",
    'GU': "આપેલ ગુજરાતી શબ્દ નિરર્થક છે.",
    'KN': "ನೀಡಿದ ಕನ್ನಡ ಪದವು ಅರ್ಥಹೀನವಾಗಿದೆ.",
    'OR': "ଦିଆଯାଇଥିବା ଓଡିଆ ଶବ୍ଦ ଅର୍ଥହୀନ।",
    'ML': "നൽകിയ മലയാളം വാക്ക് അർത്ഥശൂന്യമാണ്.",
    'DE': "Das gegebene deutsche Wort ist sinnlos.",
    'IT': "La parola italiana data non ha senso.",
    'PT': "O texto em português é nonsense.",
    'PL': "Podane polskie słowo jest bez sensu.",
    'TR': "Verilen Türkçe kelime anlamsızdır.",
    'NL': "Het gegeven Nederlandse woord is onzin.",
    'SV': "Det givna svenska ordet är nonsens.",
    'FI': "Annettu suomenkielinen sana on järjetön.",
    'DA': "Det givne danske ord er nonsens.",
    'NO': "Det gitte norske ordet er nonsens.",
    'HE': "המילה העברית הנתונה היא חסרת משמעות.",
    'FA': "کلمه فارسی داده شده بی معنی است.",
    'TH': "คำภาษาไทยที่ให้มานั้นไร้ความหมาย",
    'VI': "Từ tiếng Việt đã cho là vô nghĩa.",
    'ID': "Kata bahasa Indonesia yang diberikan tidak masuk akal.",
    'MS': "Perkataan Melayu yang diberikan tidak bermakna.",
    'FIL': "Ang ibinigay na salitang Filipino ay walang kahulugan.",
    'SW': "Neno la Kiswahili lililopewa halina maana.",
    'HA': "Kalmar Hausa da aka bayar ba ta da ma'ana.",
    'YO': "Ọrọ Yoruba ti a fun ni alailẹgbẹ.",
    'IG': "Okwu Igbo enyere enweghị isi.",
    'ZU': "Igama lesiZulu elinikeziwe alinalo ukuqonda.",
    'XH': "Igama lesiXhosa elinikiweyo alinalo nto ithethayo.",
    'ST': "Lentsoe la Sesotho le fanoeng ha le na moelelo.",
    'SN': "Izwi reShona rakapihwa harina revo.",
    'AM': "የተሰጠው አማርኛ ቃል ምንም ትርጉም የለውም።",
    'SO': "Erayga Soomaaliga la siiyay ma lahan macno.",
    'HAW': "ʻŌlelo Hawaiʻi i hāʻawi ʻia ʻaʻohe manaʻo.",
    'MI': "Ko te kupu Māori i hoatu kaore he tikanga.",
    'SM': "O le upu Samoa na tuʻuina atu e leai se uiga.",
    'TO': "Ko e lea faka-Tonga naʻe foaki ʻoku ʻikai ha ʻuhinga.",
    'FJ': "Na vosa vakaviti e solia e sega ni ibalebale.",
    'EL': "Η δοθείσα ελληνική λέξη δεν έχει νόημα.",
    'HU': "Az adott magyar szó értelmetlen.",
    'CS': "Dané české slovo je nesmysl.",
    'SK': "Dané slovenské slovo je nezmysel.",
    'HR': "Dana hrvatska riječ je besmislena.",
    'SR': "Data srpska reč je besmislena.",
    'SL': "Dana slovenska beseda je nesmiselna.",
    'BG': "Дадената българска дума е безсмислена.",
    'UK': "Дане українське слово не має сенсу.",
    'BE': "Дадзенае беларускае слова бессэнсоўнае.",
    'KK': "Берілген қазақ сөзі мағынасыз.",
    'UZ': "Berilgan o'zbekcha so'z mavhum.",
    'KY': "Берилген кыргыз сөзү маанисиз.",
    'TG': "Калимаи тоҷикӣ додашуда бе маъно аст.",
    'MN': "Өгсөн монгол үг утгагүй.",
    'BO': "བོད་སྐད་ཀྱི་ཚིག་དེ་དོན་མེད་པ་རེད།",
    'NE': "दिइएको नेपाली शब्द अर्थहीन छ।",
    'SI': "දෙන ලද සිංහල වචනය අර්ථ විරහිතය.",
    'KM': "ពាក្យខ្មែរដែលបានផ្តល់គឺគ្មានន័យ។",
    'LO': "ຄຳລາວທີ່ໃຫ້ແມ່ນບໍ່ມີຄວາມຫມາຍ.",
    'MY': "ပေးထားသော မြန်မာစကားလုံးသည် အဓိပ္ပါယ်မရှိပါ။",
    'KA': "მოცემული ქართული სიტყვა უაზროა.",
    'HY': "Տրված հայերեն բառն անիմաստ է։",
    'AZ': "Verilən Azərbaycan sözü mənasızdır.",
    'TK': "Berlen Türkmen sözi manyşyz.",
    'ET': "Antud eesti sõna on mõttetu.",
    'LV': "Dotais latviešu vārds ir bezjēdzīgs.",
    'LT': "Pateiktas lietuvių kalbos žodis yra beprasmiškas.",
    'CY': "Mae'r gair Cymraeg a roddwyd yn ddiystyr.",
    'GA': "Tá an focal Gaeilge a tugadh gan bhrí.",
    'GD': "Tha am facal Gàidhlig a chaidh a thoirt seachad gun bhrìgh.",
    'MT': "Il-kelma Maltija mogħtija hija bla sens.",
    'EU': "Emandako euskal hitza zentzugabea da.",
    'CA': "La paraula catalana donada no té sentit.",
    'GL': "A palabra galega dada non ten sentido.",
    'AF': "Die gegewe Afrikaanse woord is nonsens.",
    'IS': "Gefið íslenskt orð er tilgangslaust.",
    'FO': "Heta føroyska orðið er menningarlaust.",
    'SA': "दत्तः संस्कृतशब्दः निरर्थकः अस्ति।",
    'LA': "Verbum Latinum datum inane est.",
    'EO': "La donita Esperanta vorto estas sensenca.",
    'DEFAULT': "The text appears to be gibberish."
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
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code) if lang_code else "Gibberish"

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def format_error_response(word: str, lang_code: str) -> str:
    """
    Formats the error response in the exact required format
    Example: "word-केाीी Langcode-HI expected error -दिए गए हिंदी शब्द एक बकवास शब्द है।"
    """
    error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
    return f"word-{word} Langcode-{lang_code} expected error -{error_msg}"

def process_gibberish_response(word: str, response_text: str) -> Tuple[str, str, str]:
    """
    Process the response from the gibberish detection API
    Returns: (status, error_type, formatted_message)
    """
    if response_text == "Valid":
        return ("T", "", "")
    
    parts = response_text.split('|')
    if len(parts) >= 3:
        lang_code = parts[1].upper()
        formatted_response = format_error_response(word, lang_code)
        return ("F", "gibberish_error", formatted_response)
    
    formatted_response = format_error_response(word, "XX")
    return ("F", "gibberish_error", formatted_response)

def get_system_prompt():
    return """
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
    You MUST respond in EXACTLY one of these formats:
    - "Valid" (if the text is meaningful)
    - "Invalid|<reason>|<detected_lang>" (if the text is gibberish)
    Where <reason> is one of:
    - random_characters
    - impossible_combinations
    - nonsense_repetition
    - no_meaningful_units
    - mixed_scripts
    And <detected_lang> is the 2-letter language code if detectable, or "XX" if unknown
    """

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """
    Gibberish detector that returns formatted responses
    Returns:
        Tuple: (status, error_type, message)
        - status: 'T' (valid) or 'F' (invalid)
        - error_type: '' or 'gibberish_error'
        - message: Formatted error message if invalid
    """
    client = get_client()
    
    try:
        if not text.strip():
            return ("T", "", "")
        
        user_prompt = f"""
        Analyze: "{text}"
        
        Compare against these examples:
        [Valid] "Paris", "123 Main St", "안녕", "@username"
        [Gibberish] "xjdkl", "asdf1234", "!@#$%^", "कखगघ"
        
        Your analysis (must use exact response format):"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        result = response.choices[0].message.content.strip()
        return process_gibberish_response(text, result)
            
    except Exception as e:
        formatted_response = format_error_response(text, "XX")
        return ("F", "api_error", formatted_response)

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
        'Lang code', 
        'Language Name',
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
            expected_lang if expected_lang else "XX",
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
    print(hindi_case['Error Message'])  # This will print in your exact requested format

if __name__ == "__main__":
    load_dotenv()
    run_tests()
**************************
        *******************
        **************
        import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Load environment variables
load_dotenv()

# Language-specific error messages with the exact required format
ERROR_MESSAGES = {
    'EN': "The given English word is nonsense.",
    'FR': "Le mot français donné est un non-sens.",
    'ES': "El texto en español no tiene sentido.",
    'RU': "Данное русское слово бессмысленно.",
    'AR': "الكلمة العربية المعطاة غير منطقية.",
    'ZH': "中文文本是乱码。",
    'JA': "日本語のテキストは無意味です。",
    'KO': "주어진 한국어 단어는 무의미합니다.",
    'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    # ... (keep all other language entries)
    'DEFAULT': "The text appears to be gibberish."
}

# Language code to name mapping (existing from your code)
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    # ... (keep all other language mappings)
}

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def format_error_response(word: str, lang_code: str) -> str:
    """Formats the error response in the exact required format"""
    error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
    return f"word-{word} Langcode-{lang_code} expected error -{error_msg}"

def process_gibberish_response(word: str, response_text: str) -> Tuple[str, str, str]:
    """Process the API response and return formatted output"""
    if response_text == "Valid":
        return ("T", "", "")
    
    parts = response_text.split('|')
    if len(parts) >= 3:
        lang_code = parts[1].upper()
        if lang_code in ERROR_MESSAGES:
            return ("F", "gibberish_error", format_error_response(word, lang_code))
    
    return ("F", "gibberish_error", format_error_response(word, "XX"))

def get_system_prompt():
    return """You are a language detection system. Respond with:
    - "Valid" for real words
    - "Invalid|<reason>|<lang>" for gibberish
    Where <lang> is a 2-letter language code from our supported languages."""

def check_gibberish(text: str) -> Tuple[str, str, str]:
    """Main function to check if text is gibberish"""
    client = get_client()
    
    try:
        if not text.strip():
            return ("T", "", "")
        
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
        return process_gibberish_response(text, result)
            
    except Exception:
        return ("F", "api_error", format_error_response(text, "XX"))

def run_tests():
    """Run all test cases and save results to Excel"""
    test_cases = [
        # Valid Texts
        ("Hello world", "T", "EN"),
        ("Bonjour", "T", "FR"),
        ("こんにちは", "T", "JA"),
        
        # Gibberish Cases
        ("asdfghjkl", "F", None),
        ("केाीी", "F", "HI"),
        ("ضصثقضصثق", "F", "AR"),
        ("漢字漢字", "F", "ZH"),
        ("qwertyuiop", "F", None),
        ("कखगघ", "F", "HI"),
        # Add more test cases as needed
    ]

    print("Running gibberish detection tests...")
    results = []
    
    for text, expected_status, expected_lang in test_cases:
        status, err_type, msg = check_gibberish(text)
        
        # Extract detected language code
        detected_lang = "XX"
        if status == "F":
            parts = msg.split()
            if len(parts) > 1 and parts[1].startswith("Langcode-"):
                detected_lang = parts[1].split("-")[1]
        
        results.append({
            'Word': text,
            'Expected Status': expected_status,
            'Actual Status': status,
            'Detected Lang Code': detected_lang,
            'Language Name': get_language_name(detected_lang),
            'Error Message': msg if status == "F" else ""
        })
    
    # Create and save DataFrame
    df = pd.DataFrame(results)
    filename = "gibberish_test_results.xlsx"
    df.to_excel(filename, index=False)
    
    print(f"\nAll test results saved to {filename}")
    print("\nSample results:")
    print(df.head())

if __name__ == "__main__":
    run_tests()


***********************************
**************************
***************************
*******************

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import Tuple
import pandas as pd

# Load environment variables
load_dotenv()

# Language-specific error messages
ERROR_MESSAGES = {
    'EN': "The given English word is nonsense.",
    'FR': "Le mot français donné est un non-sens.",
    # ... (all other language messages)
    'DEFAULT': "The text appears to be gibberish."
}

# Complete language code to name mapping
LANGUAGE_NAMES = {
    'EN': 'English',
    'FR': 'French',
    'ES': 'Spanish',
    'RU': 'Russian',
    'AR': 'Arabic',
    'ZH': 'Chinese',
    'JA': 'Japanese',
    'KO': 'Korean',
    'HI': 'Hindi',
    # ... (all other languages)
}

def get_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-07-01-preview"
    )

def detect_language(text: str) -> str:
    """Detect language for any text (both valid and gibberish)"""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Detect the language of this text. Respond with just the 2-letter language code."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=10
        )
        lang_code = response.choices[0].message.content.strip().upper()
        return lang_code if lang_code in LANGUAGE_NAMES else "XX"
    except Exception:
        return "XX"

def format_response(word: str, lang_code: str, is_valid: bool) -> str:
    """Format response for both valid and invalid words"""
    if is_valid:
        return f"word-{word} Langcode-{lang_code} (Valid {LANGUAGE_NAMES.get(lang_code, 'Unknown')})"
    else:
        error_msg = ERROR_MESSAGES.get(lang_code, ERROR_MESSAGES['DEFAULT'])
        return f"word-{word} Langcode-{lang_code} expected error -{error_msg}"

def check_text(text: str) -> Tuple[str, str, str, str]:
    """Check text and return status, lang_code, lang_name, and formatted message"""
    if not text.strip():
        return ("T", "XX", "Unknown", "")
    
    # First detect language
    lang_code = detect_language(text)
    lang_name = LANGUAGE_NAMES.get(lang_code, "Unknown")
    
    # Then check if gibberish
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Respond with 'Valid' or 'Invalid'"},
                {"role": "user", "content": f"Is this text valid: '{text}'"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        
        if result == "Valid":
            return ("T", lang_code, lang_name, format_response(text, lang_code, True))
        else:
            return ("F", lang_code, lang_name, format_response(text, lang_code, False))
    except Exception:
        return ("F", "XX", "Unknown", format_response(text, "XX", False))

def run_tests():
    test_cases = [
        ("Hello", "T", "EN"),
        ("Bonjour", "T", "FR"),
        ("こんにちは", "T", "JA"),
        ("asdfghjkl", "F", None),
        ("केाीी", "F", "HI"),
        ("مرحبا", "T", "AR"),
        ("漢字", "T", "ZH"),
        ("qwertyuiop", "F", None),
        ("Привет", "T", "RU")
    ]

    print("Running language detection tests...")
    results = []
    
    for text, expected_status, expected_lang in test_cases:
        status, lang_code, lang_name, msg = check_text(text)
        
        results.append({
            'Word': text,
            'Expected Status': expected_status,
            'Actual Status': status,
            'Detected Lang Code': lang_code,
            'Language Name': lang_name,
            'Message': msg
        })
    
    # Create and save DataFrame
    df = pd.DataFrame(results)
    filename = "language_test_results.xlsx"
    df.to_excel(filename, index=False)
    
    print(f"\nAll test results saved to {filename}")
    print("\nSample results:")
    print(df.head())

if __name__ == "__main__":
    run_tests()
