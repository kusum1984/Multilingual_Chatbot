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
