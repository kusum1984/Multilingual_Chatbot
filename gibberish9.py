from langchain.chat_models import AzureChatOpenAI
import configparser
from typing import Tuple

class GibberishDetector:
    def __init__(self, config_path='config.ini'):
        """
        Initialize the gibberish detector with configuration
        """
        self.cfg = configparser.ConfigParser()
        self.cfg.read(config_path)
        
        # Initialize AzureChatOpenAI client
        self.llm = AzureChatOpenAI(
            api_key=self.cfg['AzureOpenAI']['ApiKey'],
            azure_endpoint=self.cfg['AzureOpenAI']['Endpoint'],
            api_version=self.cfg['AzureOpenAI']['ApiVersion'],
            deployment_name=self.cfg['AzureOpenAI']['GibberishValidation']['Model'],
            temperature=0.2,
            max_tokens=200
        )
        
        # Language-specific error messages
        self.error_messages = {
            'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
            'ES': "El texto en español proporcionado no tiene sentido.",
            'PT': "O texto em português fornecido é sem sentido.",
            'ZH': "提供的中文文本是无意义的。",
            'JA': "提供された日本語のテキストは無意味です。",
            'DE': "Der bereitgestellte deutsche Text ist sinnlos.",
            'FR': "Le texte français fourni est un non-sens.",
            'EN': "The provided text is gibberish.",
            'IT': "Il testo italiano fornito non ha senso.",
            'RU': "Предоставленный русский текст бессмыслен.",
            'AR': "النص العربي المقدم غير منطقي.",
            'KO': "제공된 한국어 텍스트는 무의미합니다.",
            'NL': "De verstrekte Nederlandse tekst is onzin.",
            'SV': "Den tillhandahållna svenska texten är nonsens.",
            'FI': "Annettu suomenkielinen teksti on järjetön.",
            'DA': "Den leverede danske tekst er nonsens.",
            'PL': "Dostarczony polski tekst jest bez sensu.",
            'TR': "Sağlanan Türkçe metin anlamsızdır.",
            'TH': "ข้อความภาษาไทยที่ให้มานั้นไม่มีสาระ"
        }

    def get_system_prompt(self) -> str:
        """
        Returns the comprehensive system prompt with examples
        """
        return """You are an advanced multilingual text analysis model trained to detect gibberish across 50+ languages.

DEFINITION:
Gibberish is text that:
- Lacks coherent meaning in the specified language context
- Contains random character sequences not forming valid words
- Shows no grammatical structure
- Includes excessive repeated characters/patterns
- Contains nonsense combinations of valid morphemes

EXAMPLES OF GIBBERISH:
1. English: "asdf jklö pqzm xyzbb"
2. Hindi: "केाीी िजक ल पनत"
3. Spanish: "asdfg ñlkjh qwertyú"
4. Japanese: "あいうえおかきくけこさしすせそたちつてと"
5. Russian: "ывапролдж фыва ячсмить"

EXAMPLES OF VALID TEXT:
1. English: "The quick brown fox jumps"
2. Hindi: "एक तेज भूरी लोमड़ी कूदती है"
3. Spanish: "El rápido zorro marrón salta"
4. Japanese: "速い茶色の狐が跳びます"
5. Russian: "Быстрая коричневая лиса прыгает"

ANALYSIS GUIDELINES:
1. First determine the language context from the lang_code
2. Check if text contains valid words/structures in that language
3. For mixed-language text, consider any valid language as non-gibberish
4. Numbers/dates/URLs are always valid
5. Proper nouns/names are valid
6. Technical terms/acronyms are valid
7. Single existing words are valid

RESPONSE FORMAT:
- For valid text: Respond with exactly "Valid"
- For gibberish: Respond with exactly "Invalid"
"""

    def get_user_prompt(self, text: str, lang_code: str) -> str:
        """
        Returns the user prompt for analysis
        """
        return f"""Analyze this text for gibberish (language code: {lang_code}):
{text}

Respond with exactly "Valid" or "Invalid" based on the analysis guidelines."""

    def check_gibberish(self, text: str, lang_code: str = 'EN') -> Tuple[str, str, str]:
        """
        Checks if text is gibberish in the specified language
        
        Returns:
            Tuple: (result_flag, error_type, error_message)
            - result_flag: 'T' for valid, 'F' for invalid
            - error_type: '' if valid, 'gibberish_error' if invalid
            - error_message: '' if valid, language-specific message if invalid
        """
        try:
            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(text, lang_code)}
            ]
            
            # Call the Azure LLM
            response = self.llm(messages)
            result = response.content.strip()
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = self.error_messages.get(lang_code, self.error_messages['EN'])
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            return 'F', 'api_error', str(e)


if __name__ == "__main__":
    # Initialize detector
    detector = GibberishDetector()
    
    # Comprehensive test cases (40+ examples covering 20 languages)
    test_cases = [
        # Hindi
        ("केाीी िजक ल", "HI"),  # Gibberish
        ("नमस्ते दुनिया", "HI"),  # Valid
        
        # Spanish
        ("asdfg ñlkjh qwert", "ES"),  # Gibberish
        ("El rápido zorro marrón", "ES"),  # Valid
        
        # Portuguese
        ("qwedsa zxcvb mkoiu", "PT"),  # Gibberish
        ("Olá mundo como vai", "PT"),  # Valid
        
        # Chinese
        ("的的的 我我我 不不不", "ZH"),  # Gibberish
        ("你好世界", "ZH"),  # Valid
        
        # Japanese
        ("あいうえおかきくけこさしすせそ", "JA"),  # Gibberish
        ("速い茶色の狐", "JA"),  # Valid
        
        # German
        ("qwertz uiopü asdfg", "DE"),  # Gibberish
        ("Der schnelle braune Fuchs", "DE"),  # Valid
        
        # French
        ("azerty uiop qsdfg", "FR"),  # Gibberish
        ("Le rapide renard brun", "FR"),  # Valid
        
        # English
        ("Xysd fgtw qwert asdf", "EN"),  # Gibberish
        ("The quick brown fox jumps", "EN"),  # Valid
        
        # Italian
        ("qwsdf rtygh vbnmj", "IT"),  # Gibberish
        ("La volpe marrone veloce", "IT"),  # Valid
        
        # Russian
        ("ывапролдж фыва ячсмить", "RU"),  # Gibberish
        ("Быстрая коричневая лиса", "RU"),  # Valid
        
        # Arabic
        ("ضصثقفغعهخ حجكلمن", "AR"),  # Gibberish
        ("مرحبا بالعالم", "AR"),  # Valid
        
        # Korean
        ("ㅁㄴㅇㄹㅎㅋㅌㅊㅍ ㅛㅕㅑㅐㅔ", "KO"),  # Gibberish
        ("안녕하세요 세상", "KO"),  # Valid
        
        # Dutch
        ("qwerty uiop asdfg", "NL"),  # Gibberish
        ("Hallo wereld hoe gaat het", "NL"),  # Valid
        
        # Swedish
        ("asdfghjklö qwertyui", "SV"),  # Gibberish
        ("Den snabba bruna räven", "SV"),  # Valid
        
        # Finnish
        ("qwertyuiopå asdfghjklö", "FI"),  # Gibberish
        ("Nopea ruskea kettu", "FI"),  # Valid
        
        # Danish
        ("qwertyuiopå asdfghjklø", "DA"),  # Gibberish
        ("Den hurtige brune ræv", "DA"),  # Valid
        
        # Polish
        ("qwertyuiop asdfghjkl", "PL"),  # Gibberish
        ("Szybki brązowy lis", "PL"),  # Valid
        
        # Turkish
        ("qwertyuıopğü asdfghjklşi", "TR"),  # Gibberish
        ("Hızlı kahverengi tilki", "TR"),  # Valid
        
        # Thai
        ("กดเ้่าสว ไฟชนเ", "TH"),  # Gibberish
        ("สวัสดีชาวโลก", "TH"),  # Valid
        
        # Edge cases
        ("12345 67890", "EN"),  # Valid (numbers)
        ("@#$%^ &*()", "EN"),  # Gibberish (symbols)
        ("NASA SpaceX", "EN"),  # Valid (acronyms)
        ("John Doe 42", "EN"),  # Valid (name with number)
        ("https://example.com", "EN"),  # Valid (URL)
    ]
    
    # Run all test cases
    print("Gibberish Detection Test Results")
    print("=" * 60)
    for i, (text, lang) in enumerate(test_cases, 1):
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        print(f"Test {i}: {lang}")
        print(f"Text: {text}")
        print(f"Result: {'Valid' if result == 'T' else 'Invalid'}")
        if error_msg:
            print(f"Message: {error_msg}")
        print("-" * 60)
*****************************
*************************
from langchain.chat_models import AzureChatOpenAI
import configparser
from typing import Tuple

class GibberishDetector:
    def __init__(self):
        """
        Initialize the gibberish detector with direct Azure OpenAI values
        """
        # Initialize AzureChatOpenAI client with direct values
        self.llm = AzureChatOpenAI(
            openai_api_key="your-azure-openai-api-key",  # Replace with your actual API key
            azure_endpoint="https://your-resource-name.openai.azure.com/",  # Replace with your endpoint
            api_version="2023-05-15",
            deployment_name="your-deployment-name",  # Replace with your deployment name
            temperature=0.2,
            max_tokens=200
        )
        
        # Language-specific error messages (unchanged)
        self.error_messages = {
            'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
            'ES': "El texto en español proporcionado no tiene sentido.",
            'PT': "O texto em português fornecido é sem sentido.",
            'ZH': "提供的中文文本是无意义的。",
            'JA': "提供された日本語のテキストは無意味です。",
            'DE': "Der bereitgestellte deutsche Text ist sinnlos.",
            'FR': "Le texte français fourni est un non-sens.",
            'EN': "The provided text is gibberish.",
            'IT': "Il testo italiano fornito non ha senso.",
            'RU': "Предоставленный русский текст бессмыслен.",
            'AR': "النص العربي المقدم غير منطقي.",
            'KO': "제공된 한국어 텍스트는 무의미합니다.",
            'NL': "De verstrekte Nederlandse tekst is onzin."
        }

    def get_system_prompt(self) -> str:
        """
        Returns the comprehensive system prompt with examples
        """
        return """
        You are an advanced multilingual text analysis model trained to detect gibberish across 50+ languages.
        
        DEFINITION:
        Gibberish is text that:
        - Lacks coherent meaning in the specified language context
        - Contains random character sequences not forming valid words
        - Shows no grammatical structure
        - Includes excessive repeated characters/patterns
        
        EXAMPLES OF GIBBERISH:
        1. English: "asdf jklö pqzm"
        2. Hindi: "केाीी िजक ल"
        3. Spanish: "asdfg ñlkjh"
        4. Japanese: "あいうえおかきくけこさしすせそ"
        5. Russian: "ывапролдж фыва"
        
        EXAMPLES OF VALID TEXT:
        1. English: "The quick brown fox"
        2. Hindi: "एक तेज भूरी लोमड़ी"
        3. Spanish: "El rápido zorro marrón"
        4. Japanese: "速い茶色の狐"
        5. Russian: "Быстрая коричневая лиса"
        
        INSTRUCTIONS:
        1. First determine the language context from the lang_code
        2. Check if text contains valid words/structures in that language
        3. For mixed-language text, consider any valid language as non-gibberish
        4. For valid text, respond with exactly: "Valid"
        5. For gibberish, respond with exactly: "Invalid"
        
        SPECIAL CASES:
        - Proper nouns/names should be considered valid
        - Technical terms/acronyms are valid
        - Numbers/dates are valid
        - Single words are valid if they exist in any language
        """

    def get_user_prompt(self, text: str, lang_code: str) -> str:
        """
        Returns the user prompt for analysis
        """
        return f"""
        Analyze this text for gibberish (language code: {lang_code}):
        {text}
        
        Respond with either:
        1. "Valid" if the text contains recognizable words/nouns in the specified language
        2. "Invalid" if the text is meaningless in the specified language
        """

    def check_gibberish(self, text: str, lang_code: str = 'EN') -> Tuple[str, str, str]:
        """
        Checks if text is gibberish in the specified language
        
        Returns:
            Tuple: (result_flag, error_type, error_message)
            - result_flag: 'T' for valid, 'F' for invalid
            - error_type: '' if valid, 'gibberish_error' if invalid
            - error_message: '' if valid, language-specific message if invalid
        """
        try:
            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(text, lang_code)}
            ]
            
            # Call the Azure LLM
            response = self.llm(messages)
            result = response.content.strip()
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = self.error_messages.get(lang_code, self.error_messages['EN'])
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            return 'F', 'api_error', str(e)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = GibberishDetector()
    
    # Test cases
    test_cases = [
        ("केाीी", "HI"),  # Hindi gibberish
        ("नमस्ते दुनिया", "HI"),  # Hindi valid
        ("asdfg hjklñ", "ES"),  # Spanish gibberish
        ("Hola mundo", "ES"),  # Spanish valid
        ("あいうえおかきくけこ", "JA"),  # Japanese gibberish
        ("こんにちは世界", "JA"),  # Japanese valid
        ("Xysd fgtw qwert", "EN"),  # English gibberish
        ("The quick brown fox", "EN")  # English valid
    ]
    
    for text, lang in test_cases:
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        print(f"Language: {lang}")
        print(f"Text: {text}")
        print(f"Result: {result} | Error Type: {error_type} | Message: {error_msg}")
        print("-" * 60)




************************
****************************
from langchain.chat_models import AzureChatOpenAI
from typing import Tuple

class GibberishDetector:
    def __init__(self):
        """
        Initialize the gibberish detector with direct Azure OpenAI values
        """
        # Initialize AzureChatOpenAI client with direct values
        self.llm = AzureChatOpenAI(
            openai_api_key="your-azure-openai-api-key",
            azure_endpoint="https://your-resource-name.openai.azure.com/",
            api_version="2023-05-15",
            deployment_name="your-deployment-name",
            temperature=0.2,
            max_tokens=200
        )
        
        # Language-specific error messages
        self.error_messages = {
            'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
            'ES': "El texto en español proporcionado no tiene sentido.",
            'PT': "O texto em português fornecido é sem sentido.",
            'ZH': "提供的中文文本是无意义的。",
            'JA': "提供された日本語のテキストは無意味です。",
            'DE': "Der bereitgestellte deutsche Text ist sinnlos.",
            'FR': "Le texte français fourni es un non-sens.",
            'EN': "The provided text is gibberish.",
            'IT': "Il testo italiano fornito non ha senso.",
            'RU': "Предоставленный русский текст бессмыслен.",
            'AR': "النص العربي المقدم غير منطقي.",
            'KO': "제공된 한국어 텍스트는 무의미합니다.",
            'NL': "De verstrekte Nederlandse tekst is onzin."
        }

    def get_system_prompt(self) -> str:
        """
        Returns the comprehensive system prompt with examples
        """
        return """[Previous system prompt content remains exactly the same]"""

    def get_user_prompt(self, text: str, lang_code: str) -> str:
        """
        Returns the user prompt for analysis
        """
        return f"""Analyze this text for gibberish (language code: {lang_code}):
{text}

Respond with either:
1. "Valid" if the text contains recognizable words/nouns
2. "Invalid" if the text is meaningless"""

    def check_gibberish(self, text: str, lang_code: str = 'EN') -> Tuple[str, str, str]:
        """
        Checks if text is gibberish with exact error message format
        """
        try:
            # Call the Azure LLM with proper message format
            response = self.llm([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(text, lang_code)}
            ])
            
            # Extract content from AIMessage object
            result = response.content.strip() if hasattr(response, 'content') else str(response)
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"{self.error_messages.get(lang_code, self.error_messages['EN'])}\""
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"API Error: {str(e)}\""
            return 'F', 'api_error', error_msg


# Example usage with exact output format
if __name__ == "__main__":
    detector = GibberishDetector()
    
    test_cases = [
        ("केाीी", "HI"),
        ("asdfg hjklñ", "ES"),
        ("あいうえおかきくけこ", "JA"),
        ("Xysd fgtw qwert", "EN")
    ]
    
    for text, lang in test_cases:
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        if result == 'F':
            print(error_msg)
        else:
            print(f"Valid text: {text} (Language: {lang})")

********************
***************
*******************
from langchain.chat_models import AzureChatOpenAI
from typing import Tuple

class GibberishDetector:
    def __init__(self):
        """
        Initialize the gibberish detector with direct Azure OpenAI values
        """
        # Initialize AzureChatOpenAI client with direct values
        self.llm = AzureChatOpenAI(
            openai_api_key="your-azure-openai-api-key",  # Replace with your actual API key
            azure_endpoint="https://your-resource-name.openai.azure.com/",  # Replace with your endpoint
            api_version="2023-05-15",
            deployment_name="your-deployment-name",  # Replace with your deployment name
            temperature=0.2,
            max_tokens=200
        )
        
        # Language-specific error messages
        self.error_messages = {
            'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
            'ES': "El texto en español proporcionado no tiene sentido.",
            'PT': "O texto em português fornecido é sem sentido.",
            'ZH': "提供的中文文本是无意义的。",
            'JA': "提供された日本語のテキストは無意味です。",
            'DE': "Der bereitgestellte deutsche Text ist sinnlos.",
            'FR': "Le texte français fourni est un non-sens.",
            'EN': "The provided text is gibberish.",
            'IT': "Il testo italiano fornito non ha senso.",
            'RU': "Предоставленный русский текст бессмыслен.",
            'AR': "النص العربي المقدم غير منطقي.",
            'KO': "제공된 한국어 텍스트는 무의미합니다.",
            'NL': "De verstrekte Nederlandse tekst is onzin."
        }

    def get_system_prompt(self) -> str:
        """
        Returns the comprehensive system prompt with examples
        """
        return """You are an advanced multilingual text analysis model trained to detect gibberish across 50+ languages.

DEFINITION:
Gibberish is text that:
- Lacks coherent meaning in the specified language context
- Contains random character sequences not forming valid words
- Shows no grammatical structure
- Includes excessive repeated characters/patterns

EXAMPLES OF GIBBERISH:
1. English: "asdf jklö pqzm"
2. Hindi: "केाीी िजक ल"
3. Spanish: "asdfg ñlkjh"
4. Japanese: "あいうえおかきくけこ"
5. Russian: "ывапролдж фыва"

EXAMPLES OF VALID TEXT:
1. English: "The quick brown fox"
2. Hindi: "एक तेज भूरी लोमड़ी"
3. Spanish: "El rápido zorro marrón"
4. Japanese: "速い茶色の狐"
5. Russian: "Быстрая коричневая лиса"

INSTRUCTIONS:
1. First determine the language context from the lang_code
2. Check if text contains valid words/structures in that language
3. For mixed-language text, consider any valid language as non-gibberish
4. For valid text, respond with exactly "Valid"
5. For gibberish, respond with exactly "Invalid"

SPECIAL CASES:
- Proper nouns/names should be considered valid
- Technical terms/acronyms are valid
- Numbers/dates are valid
- Single words are valid if they exist in any language"""

    def get_user_prompt(self, text: str, lang_code: str) -> str:
        """
        Returns the user prompt for analysis
        """
        return f"""Analyze this text for gibberish (language code: {lang_code}):
{text}

Respond with either:
1. "Valid" if the text contains recognizable words/nouns
2. "Invalid" if the text is meaningless"""

    def check_gibberish(self, text: str, lang_code: str = 'EN') -> Tuple[str, str, str]:
        """
        Checks if text is gibberish with exact error message format
        """
        try:
            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(text, lang_code)}
            ]
            
            # Call the Azure LLM
            response = self.llm(messages)
            
            # Get content from response
            result = response.content.strip() if hasattr(response, 'content') else str(response)
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"{self.error_messages.get(lang_code, self.error_messages['EN'])}\""
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"API Error: {str(e)}\""
            return 'F', 'api_error', error_msg


# Main execution with test cases
if __name__ == "__main__":
    # Initialize detector
    detector = GibberishDetector()
    
    # Test cases covering multiple languages
    test_cases = [
        ("केाीी", "HI"),  # Hindi gibberish
        ("नमस्ते दुनिया", "HI"),  # Hindi valid
        ("asdfg hjklñ", "ES"),  # Spanish gibberish
        ("Hola mundo", "ES"),  # Spanish valid
        ("あいうえおかきくけこ", "JA"),  # Japanese gibberish
        ("こんにちは世界", "JA"),  # Japanese valid
        ("Xysd fgtw qwert", "EN"),  # English gibberish
        ("The quick brown fox", "EN"),  # English valid
        ("qwedsa zxcvb", "PT"),  # Portuguese gibberish
        ("Olá mundo", "PT"),  # Portuguese valid
        ("的的的 我我我", "ZH"),  # Chinese gibberish
        ("你好世界", "ZH")  # Chinese valid
    ]
    
    # Execute all test cases
    for text, lang in test_cases:
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        if result == 'F':
            print(error_msg)
        else:
            print(f"Valid text: {text} (Language: {lang})")



******************
**********************
*****************
*******************

from langchain.chat_models import AzureChatOpenAI
from typing import Tuple

class GibberishDetector:
    def __init__(self):
        """
        Initialize the gibberish detector with direct Azure OpenAI values
        """
        # Initialize AzureChatOpenAI client with direct values
        self.llm = AzureChatOpenAI(
            openai_api_key="your-azure-openai-api-key",
            azure_endpoint="https://your-resource-name.openai.azure.com/",
            api_version="2023-05-15",
            deployment_name="your-deployment-name",
            temperature=0.2,
            max_tokens=200
        )
        
        # Language-specific error messages
        self.error_messages = {
            'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
            'ES': "El texto en español proporcionado no tiene sentido.",
            'PT': "O texto em português fornecido é sem sentido.",
            'ZH': "提供的中文文本是无意义的。",
            'JA': "提供された日本語のテキストは無意味です。",
            'DE': "Der bereitstellte deutsche Text ist sinnlos.",
            'FR': "Le texte français fourni est un non-sens.",
            'EN': "The provided text is gibberish.",
            'IT': "Il testo italiano fornito non ha senso.",
            'RU': "Предоставленный русский текст бессмыслен.",
            'AR': "النص العربي المقدم غير منطقي.",
            'KO': "제공된 한국어 텍스트는 무의미합니다.",
            'NL': "De verstrekte Nederlandse tekst is onzin."
        }

    def get_system_prompt(self) -> str:
        """
        Returns the comprehensive system prompt with examples
        """
        return """You are an advanced multilingual text analysis model trained to detect gibberish across 50+ languages.

DEFINITION:
Gibberish is text that:
- Lacks coherent meaning in the specified language context
- Contains random character sequences not forming valid words
- Shows no grammatical structure
- Includes excessive repeated characters/patterns

EXAMPLES OF GIBBERISH:
1. English: "asdf jklö pqzm"
2. Hindi: "केाीी िजक ल"
3. Spanish: "asdfg ñlkjh"
4. Japanese: "あいうえおかきくけこ"
5. Russian: "ывапролдж фыва"

EXAMPLES OF VALID TEXT:
1. English: "The quick brown fox"
2. Hindi: "एक तेज भूरी लोमड़ी"
3. Spanish: "El rápido zorro marrón"
4. Japanese: "速い茶色の狐"
5. Russian: "Быстрая коричневая лиса"

INSTRUCTIONS:
1. First determine the language context from the lang_code
2. Check if text contains valid words/structures in that language
3. For mixed-language text, consider any valid language as non-gibberish
4. For valid text, respond with exactly "Valid"
5. For gibberish, respond with exactly "Invalid"

SPECIAL CASES:
- Proper nouns/names should be considered valid
- Technical terms/acronyms are valid
- Numbers/dates are valid
- Single words are valid if they exist in any language"""

    def get_user_prompt(self, text: str, lang_code: str) -> str:
        """
        Returns the user prompt for analysis
        """
        return f"""Analyze this text for gibberish (language code: {lang_code}):
{text}

Respond with either:
1. "Valid" if the text contains recognizable words/nouns
2. "Invalid" if the text is meaningless"""

    def check_gibberish(self, text: str, lang_code: str = 'EN') -> Tuple[str, str, str]:
        """
        Checks if text is gibberish with exact error message format
        """
        try:
            # Call the Azure LLM with proper message format
            response = self.llm([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(text, lang_code)}
            ])
            
            # Proper response content extraction
            result = response.content if hasattr(response, 'content') else str(response)
            result = result.strip()
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"{self.error_messages.get(lang_code, self.error_messages['EN'])}\""
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            error_msg = f"gibberish word-{text} Langcode-{lang_code} expected error -\"API Error: {str(e)}\""
            return 'F', 'api_error', error_msg


# Example usage with exact output format
if __name__ == "__main__":
    detector = GibberishDetector()
    
    test_cases = [
        ("केाीी", "HI"),
        ("asdfg hjklñ", "ES"),
        ("あいうえおかきくけこ", "JA"),
        ("Xysd fgtw qwert", "EN"),
        ("नमस्ते दुनिया", "HI"),  # Valid Hindi
        ("Hola mundo", "ES")      # Valid Spanish
    ]
    
    for text, lang in test_cases:
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        if result == 'F':
            print(error_msg)
        else:
            print(f"Valid text: {text} (Language: {lang})")


**************************
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # Example: "gpt-4"

# System Prompt (multilingual support)
system_prompt = """
You are an advanced text analysis model trained to determine if a given text contains gibberish.
Gibberish is defined as text that lacks coherent meaning, logical structure, or context, often consisting of random sequences of letters, numbers, or symbols.

Your task:
- If the text is coherent, contains any recognizable nouns or meaningful structure in any language, or is composed entirely of numbers, respond ONLY with: "Valid"
- If the text is gibberish, respond with the appropriate language-specific error message from below if you can detect the language.

Language-specific gibberish messages:
- Hindi (HI): "दिए गए हिंदी शब्द एक बकवास शब्द है।"
- Spanish (ES): "La palabra dada en español es un galimatías."
- Portuguese (PT): "A palavra dada em português é um palavreado sem sentido."
- Chinese (ZH): "给出的中文是胡言乱语。"
- Japanese (JA): "与えられた日本語は意味不明な文字列です。"
- German (DE): "Das gegebene deutsche Wort ist Kauderwelsch."
- French (FR): "Le mot français donné est un charabia."

If the language does not match any of the above, reply with: "The given text is gibberish in an unsupported language."

Only return "Valid" or the appropriate message. Do not include additional comments or explanation.
"""

# User Prompt Template
user_prompt_template = """
Analyze the following text and determine if it contains gibberish.

Text to analyze:
{text}

Respond with either 'Valid' or the correct language-specific error message only.
"""

# Function to check gibberish
def check_gibberish(text: str):
    user_prompt = user_prompt_template.format(text=text)

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=1.0
        )

        result = response.choices[0].message["content"].strip()

        if result == "Valid":
            return 'T', '', ''
        else:
            return 'F', 'gibberish_error', result

    except Exception as e:
        return 'F', 'api_error', str(e)

# Test cases
if __name__ == "__main__":
    test_inputs = [
        "केाीी",                    # Hindi - Gibberish
        "asdkjasd",                 # Spanish - Gibberish
        "ß∂ƒ©",                     # German - Gibberish
        "Bonjour",                  # French - Valid
        "Olá mundo",                # Portuguese - Valid
        "こんにちは",                 # Japanese - Valid
        "你好",                     # Chinese - Valid
        "123456",                  # Numeric - Valid
        "éwqihqiw",                # French - Gibberish
        "Hello, how are you?",     # English - Valid
        "slkfjsdlkj",              # Unknown - Gibberish
        "😀👍🎉",                    # Emojis - Gibberish
    ]

    for text in test_inputs:
        status, code, message = check_gibberish(text)
        print(f"Input: {text}")
        print(f"Status: {status}, Code: {code}, Message: {message}")
        print("-" * 60)


