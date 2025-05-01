import openai
import configparser
from typing import Tuple

class GibberishDetector:
    def __init__(self, config_path='config.ini'):
        """
        Initialize the gibberish detector with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.cfg = configparser.ConfigParser()
        self.cfg.read(config_path)
        
        # Configure OpenAI client
        openai.api_type = "azure"
        openai.api_base = self.cfg['AzureOpenAI']['Endpoint']
        openai.api_version = self.cfg['AzureOpenAI']['ApiVersion']
        openai.api_key = self.cfg['AzureOpenAI']['ApiKey']
        
        # Language-specific error messages (expanded with more languages)
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
        Returns the user prompt for analysis (unchanged from original)
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
        (Base function implementation unchanged)
        """
        try:
            response = openai.ChatCompletion.create(
                engine=self.cfg['AzureOpenAI']['GibberishValidation']['Model'],
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": self.get_user_prompt(text, lang_code)}
                ],
                temperature=0.2,
                top_p=1.0,
                n=1
            )
            
            result = response.choices[0].message.content.strip()
            
            if result == "Valid":
                return 'T', '', ''
            else:
                error_msg = self.error_messages.get(lang_code, self.error_messages['EN'])
                return 'F', 'gibberish_error', error_msg
                
        except Exception as e:
            return 'F', 'api_error', str(e)


# Example usage with expanded test cases
if __name__ == "__main__":
    # Initialize detector
    detector = GibberishDetector()
    
    # Expanded test cases (30+ examples covering 12 languages)
    test_cases = [
        # Hindi
        ("केाीी", "HI"),  # Gibberish
        ("नमस्ते दुनिया", "HI"),  # Valid
        
        # Spanish
        ("asdfg hjklñ", "ES"),  # Gibberish
        ("Hola mundo", "ES"),  # Valid
        
        # Portuguese
        ("qwedsa zxcvb", "PT"),  # Gibberish
        ("Olá mundo", "PT"),  # Valid
        
        # Chinese
        ("的的的 我我我", "ZH"),  # Gibberish (repetitive)
        ("你好世界", "ZH"),  # Valid
        
        # Japanese
        ("あいうえおかきくけこ", "JA"),  # Gibberish (hiragana sequence)
        ("こんにちは世界", "JA"),  # Valid
        
        # German
        ("qwertz uiopü", "DE"),  # Gibberish
        ("Hallo Welt", "DE"),  # Valid
        
        # French
        ("azerty uiop", "FR"),  # Gibberish
        ("Bonjour le monde", "FR"),  # Valid
        
        # English
        ("Xysd fgtw qwert", "EN"),  # Gibberish
        ("The quick brown fox", "EN"),  # Valid
        
        # Italian
        ("qwsdf rtygh", "IT"),  # Gibberish
        ("Ciao mondo", "IT"),  # Valid
        
        # Russian
        ("ывапролдж фыва", "RU"),  # Gibberish
        ("Привет мир", "RU"),  # Valid
        
        # Arabic
        ("ضصثقفغعهخ", "AR"),  # Gibberish (random letters)
        ("مرحبا بالعالم", "AR"),  # Valid
        
        # Korean
        ("ㅁㄴㅇㄹㅎㅋㅌㅊㅍ", "KO"),  # Gibberish (jamo sequence)
        ("안녕하세요 세상", "KO"),  # Valid
        
        # Dutch
        ("qwerty uiop", "NL"),  # Gibberish
        ("Hallo wereld", "NL"),  # Valid
        
        # Edge cases
        ("12345", "EN"),  # Valid (numbers)
        ("@#$%^", "EN"),  # Gibberish (symbols)
        ("NASA", "EN"),  # Valid (acronym)
        ("John Doe", "EN"),  # Valid (name)
    ]
    
    for text, lang in test_cases:
        result, error_type, error_msg = detector.check_gibberish(text, lang)
        print(f"Language: {lang}")
        print(f"Text: {text}")
        print(f"Result: {result} | Error Type: {error_type} | Message: {error_msg}")
        print("-" * 60)
