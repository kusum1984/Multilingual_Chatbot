import os
import re
from openai import AzureOpenAI

# Configuration dictionary
cfg = {
    'AzureOpenAI': {
        'GibberishValidation': {
            'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
            'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            'api_version': '2023-07-01-preview',
            'Model': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        }
    },
    'GIBBERISH_VALIDATION': {
        'system_prompt': (
            "You are an advanced text analysis model trained to determine if a given text contains gibberish. "
            "Gibberish is defined as text that lacks coherent meaning, logical structure or context, often consisting of "
            "random sequences of letters, numbers, or symbols. Your task is to analyze the text and decide if it is mostly "
            "gibberish or mostly coherent. If the text is coherent, contains any nouns, or is meaningful in any language, or is composed entirely of numbers, "
            "respond with only the word 'Valid'. If the text is mostly gibberish and lacks coherent content or nouns, provide a brief reason explaining why it is gibberish."
        ),
        'user_prompt': (
            "Analyze the following text and determine if it contains gibberish. "
            "If the text contains recognizable words, nouns, or coherent structure in any language, or is composed entirely of numbers, respond with only the word 'Valid'. "
            "If the text is truly gibberish and lacks any coherent content, provide a brief reason explaining why:\n\n{text}\n\nRespond with only 'Valid' or the reason, without any additional explanation."
        )
    }
}

# Language-specific gibberish error messages
LANGUAGE_ERRORS = {
    'HI': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'ES': "La palabra dada en español es un galimatías.",
    'PT': "A palavra dada em português é um palavreado sem sentido.",
    'ZH': "给出的中文是胡言乱语。",
    'JA': "与えられた日本语は意味不明な文字列です。",
    'DE': "Das gegebene deutsche Wort ist Kauderwelsch.",
    'FR': "Le mot français donné est un charabia.",
}

def get_client():
    return AzureOpenAI(
        api_key=cfg['AzureOpenAI']['GibberishValidation']['api_key'],
        api_version=cfg['AzureOpenAI']['GibberishValidation']['api_version'],
        azure_endpoint=cfg['AzureOpenAI']['GibberishValidation']['azure_endpoint']
    )

def check_gibberish(text):
    system_prompt = cfg['GIBBERISH_VALIDATION']['system_prompt']
    user_prompt = cfg['GIBBERISH_VALIDATION']['user_prompt'].format(text=text)

    client = get_client()
    response = client.chat.completions.create(
        model=cfg['AzureOpenAI']['GibberishValidation']['Model'],
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        top_p=1.0,
        n=1
    )

    result = response.choices[0].message.content.strip()
    if result.lower() == "valid":
        return 'T', '', ''
    else:
        return 'F', 'gibberish_error', result

def get_language_error(lang_code):
    return LANGUAGE_ERRORS.get(lang_code.upper(), "The given text appears to be gibberish in the specified language.")

# Example wrapper
def detect_text(text, lang_code):
    is_valid, error_type, message = check_gibberish(text)
    if is_valid == 'F':
        return {
            'is_gibberish': True,
            'lang_code': lang_code.upper(),
            'error': get_language_error(lang_code)
        }
    else:
        return {
            'is_gibberish': False,
            'lang_code': lang_code.upper(),
            'error': ''
        }

# --------- TEST CASES ---------
if __name__ == "__main__":
    samples = [
        ("केाीी", "HI"),
        ("asdkjashd", "ES"),
        ("2389238", "PT"),  # Should be Valid
        ("这是一段正常的文本", "ZH"),
        ("!@#$@#%", "JA"),
        ("blablabla", "DE"),
        ("bonjour", "FR")
    ]

    for text, lang in samples:
        result = detect_text(text, lang)
        print(f"Input: {text} | Language: {lang}")
        print("Output:", result)
        print("-" * 60)


***********
*************
import os
from openai import AzureOpenAI

# Config
cfg = {
    'AzureOpenAI': {
        'GibberishValidation': {
            'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
            'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            'api_version': '2023-07-01-preview',
            'Model': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        }
    },
    'GIBBERISH_VALIDATION': {
        'system_prompt': (
            "You are an advanced text analysis model trained to determine if a given text contains gibberish. "
            "Gibberish is defined as text that lacks coherent meaning, logical structure or context, often consisting of "
            "random sequences of letters, numbers, or symbols. Your task is to analyze the text and decide if it is mostly "
            "gibberish or mostly coherent. If the text is coherent, contains any nouns, or is meaningful in any language, or is composed entirely of numbers, "
            "respond with only the word 'Valid'. If the text is mostly gibberish and lacks coherent content or nouns, provide a brief reason explaining why it is gibberish."
        ),
        'user_prompt': (
            "Analyze the following text and determine if it contains gibberish. "
            "If the text contains recognizable words, nouns, or coherent structure in any language, or is composed entirely of numbers, respond with only the word 'Valid'. "
            "If the text is truly gibberish and lacks any coherent content, provide a brief reason explaining why:\n\n{text}\n\nRespond with only 'Valid' or the reason, without any additional explanation."
        )
    }
}

# Language-specific gibberish messages
LANGUAGE_ERRORS = {
    'hi': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'es': "La palabra dada en español es un galimatías.",
    'pt': "A palavra dada em português é um palavreado sem sentido.",
    'zh-cn': "给出的中文是胡言乱语。",
    'ja': "与えられた日本語は意味不明な文字列です。",
    'de': "Das gegebene deutsche Wort ist Kauderwelsch.",
    'fr': "Le mot français donné est un charabia.",
}

# Azure OpenAI Client
def get_client():
    return AzureOpenAI(
        api_key=cfg['AzureOpenAI']['GibberishValidation']['api_key'],
        api_version=cfg['AzureOpenAI']['GibberishValidation']['api_version'],
        azure_endpoint=cfg['AzureOpenAI']['GibberishValidation']['azure_endpoint']
    )

# LLM Gibberish detection
def check_gibberish(text):
    system_prompt = cfg['GIBBERISH_VALIDATION']['system_prompt']
    user_prompt = cfg['GIBBERISH_VALIDATION']['user_prompt'].format(text=text)

    client = get_client()
    response = client.chat.completions.create(
        model=cfg['AzureOpenAI']['GibberishValidation']['Model'],
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        top_p=1.0,
        n=1
    )

    result = response.choices[0].message.content.strip()
    return (False, "") if result.lower() == "valid" else (True, result)

# Match error message to lang code
def get_lang_code_and_error(reason):
    for code, message in LANGUAGE_ERRORS.items():
        if message in reason:
            return code, message
    return "unknown", reason  # fallback

# Final wrapper
def detect_gibberish(text):
    is_gibberish, reason = check_gibberish(text)

    if is_gibberish:
        lang_code, error_msg = get_lang_code_and_error(reason)
        print(f"Gibberish Word: {text}")
        print(f"Lang Code: {lang_code}")
        print(f"Error: {error_msg}")
    else:
        print(f"Gibberish Word: {text}")
        print(f"Lang Code: ")
        print("Error: ")

# -------------------- TESTING --------------------
if __name__ == "__main__":
    test_inputs = [
        "केाीी",        # Hindi gibberish
        "aaaa",         # Portuguese gibberish
        "2389238",      # Valid number
        "bonjour",      # Valid French
        "!@#$@#%",      # Symbols
        "这是乱码",     # Chinese gibberish
    ]

    for word in test_inputs:
        print("\n----------------------------")
        detect_gibberish(word)


**************
*************
**********
import os
from openai import AzureOpenAI

# Configurations
cfg = {
    'AzureOpenAI': {
        'GibberishValidation': {
            'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
            'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            'api_version': '2023-07-01-preview',
            'Model': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        }
    },
    'GIBBERISH_VALIDATION': {
        'system_prompt': (
            "You are an advanced text analysis model trained to determine if a given text contains gibberish. "
            "Gibberish is defined as text that lacks coherent meaning, logical structure, or context, often consisting of "
            "random sequences of letters, numbers, or symbols. Your task is to analyze the text and decide if it is mostly "
            "gibberish or mostly coherent. If the text is coherent, contains any nouns, is meaningful in any language, or is composed entirely of numbers, "
            "respond with only the word 'Valid'. If the text is mostly gibberish and lacks coherent content or nouns, provide a brief reason explaining why it is gibberish."
        ),
        'user_prompt': (
            "Analyze the following text and determine if it contains gibberish. "
            "If the text contains recognizable words, nouns, or coherent structure in any language, or is composed entirely of numbers, respond with only the word 'Valid'. "
            "If the text is truly gibberish and lacks any coherent content, provide a brief reason explaining why:\n\n{text}\n\nRespond with only 'Valid' or the reason, without any additional explanation."
        )
    }
}

# Predefined language-specific gibberish messages
LANGUAGE_ERRORS = {
    'hi': "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    'es': "La palabra dada en español es un galimatías.",
    'pt': "A palavra dada em português é um palavreado sem sentido.",
    'zh-cn': "给出的中文是胡言乱语。",
    'ja': "与えられた日本語は意味不明な文字列です。",
    'de': "Das gegebene deutsche Wort ist Kauderwelsch.",
    'fr': "Le mot français donné est un charabia."
}

# Initialize Azure OpenAI Client
def get_client():
    return AzureOpenAI(
        api_key=cfg['AzureOpenAI']['GibberishValidation']['api_key'],
        api_version=cfg['AzureOpenAI']['GibberishValidation']['api_version'],
        azure_endpoint=cfg['AzureOpenAI']['GibberishValidation']['azure_endpoint']
    )

# Call Azure OpenAI to validate gibberish
def check_gibberish(text):
    system_prompt = cfg['GIBBERISH_VALIDATION']['system_prompt']
    user_prompt = cfg['GIBBERISH_VALIDATION']['user_prompt'].format(text=text)

    client = get_client()
    response = client.chat.completions.create(
        model=cfg['AzureOpenAI']['GibberishValidation']['Model'],
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        top_p=1.0
    )

    result = response.choices[0].message.content.strip()
    return result

# Match model's output to a predefined language error
def get_langcode_and_error(message):
    for lang_code, error in LANGUAGE_ERRORS.items():
        if error in message:
            return lang_code, error
    return "unknown", message  # fallback for unrecognized responses

# Final wrapper function
def detect_gibberish(text):
    result = check_gibberish(text)

    if result.lower() == "valid":
        print(f"Gibberish Word: {text}")
        print("Lang Code: ")
        print("Error: ")
    else:
        lang_code, error_msg = get_langcode_and_error(result)
        print(f"Gibberish Word: {text}")
        print(f"Lang Code: {lang_code}")
        print(f"Error: {error_msg}")

# ---------- TESTING ----------
if __name__ == "__main__":
    test_inputs = [
        "केाीी",        # Hindi gibberish
        "aaaa",         # Possibly PT/ES
        "これはでこちゅ", # Japanese gibberish
        "!@#@#!",       # General gibberish
        "bonjour",      # Valid French
        "284910",       # Valid number
    ]

    for word in test_inputs:
        print("\n----------------------------")
        detect_gibberish(word)

**********************
import os
from openai import AzureOpenAI

# Configuration
cfg = {
    'AzureOpenAI': {
        'GibberishValidation': {
            'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
            'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            'api_version': '2023-07-01-preview',
            'Model': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        }
    },
    'GIBBERISH_VALIDATION': {
        'system_prompt': (
            "You are an advanced text analysis model trained to determine if a given text contains gibberish. "
            "Gibberish is defined as text that lacks coherent meaning, logical structure, or context, often consisting of "
            "random sequences of letters, numbers, or symbols. Your task is to analyze the text and decide if it is mostly "
            "gibberish or mostly coherent. If the text is coherent, contains any nouns, is meaningful in any language, or is composed entirely of numbers, "
            "respond with only the word 'Valid'. If the text is mostly gibberish and lacks coherent content or nouns, provide a brief reason explaining why it is gibberish."
        ),
        'user_prompt': (
            "Analyze the following text and determine if it contains gibberish. "
            "If the text contains recognizable words, nouns, or coherent structure in any language, or is composed entirely of numbers, respond with only the word 'Valid'. "
            "If the text is truly gibberish and lacks any coherent content, provide a brief reason explaining why:\n\n{text}\n\nRespond with only 'Valid' or the reason, without any additional explanation."
        )
    }
}

# Language error message dictionary
LANGUAGE_ERROR_MAP = {
    "hi": "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    "es": "La palabra dada en español es un galimatías.",
    "pt": "A palavra dada em português é un palavreado sem sentido.",
    "zh-cn": "给出的中文是胡言乱语。",
    "ja": "与えられた日本語は意味不明な文字列です。",
    "de": "Das gegebene deutsche Wort ist Kauderwelsch.",
    "fr": "Le mot français donné est un charabia."
}

# Reverse error to langcode mapping
ERROR_TO_LANG = {v: k for k, v in LANGUAGE_ERROR_MAP.items()}

# Azure OpenAI client
def get_client():
    return AzureOpenAI(
        api_key=cfg['AzureOpenAI']['GibberishValidation']['api_key'],
        api_version=cfg['AzureOpenAI']['GibberishValidation']['api_version'],
        azure_endpoint=cfg['AzureOpenAI']['GibberishValidation']['azure_endpoint']
    )

# Function to check for gibberish
def check_gibberish(text):
    system_prompt = cfg['GIBBERISH_VALIDATION']['system_prompt']
    user_prompt = cfg['GIBBERISH_VALIDATION']['user_prompt'].format(text=text)

    client = get_client()
    response = client.chat.completions.create(
        model=cfg['AzureOpenAI']['GibberishValidation']['Model'],
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        top_p=1.0
    )
    result = response.choices[0].message.content.strip()
    return result

# Function to determine lang code based on error message
def get_langcode_from_error(error_msg):
    for err, lang_code in ERROR_TO_LANG.items():
        if err in error_msg:
            return lang_code, err
    return "unknown", error_msg

# Main wrapper
def detect_gibberish(text):
    result = check_gibberish(text)

    if result.lower() == "valid":
        print(f"Gibberish Word: {text}")
        print("Lang Code: ")
        print("Error: ")
    else:
        lang_code, error_msg = get_langcode_from_error(result)
        print(f"Gibberish Word: {text}")
        print(f"Lang Code: {lang_code}")
        print(f"Error: {error_msg}")

# ----------------- TEST -----------------
if __name__ == "__main__":
    test_words = [
        "केाीी",        # Hindi gibberish
        "aaaa",         # Portuguese or Spanish gibberish
        "これはでこちゅ",  # Japanese gibberish
        "!!@@##",       # Symbol gibberish
        "bonjour",      # Valid French
        "12345",        # Valid number
    ]

    for word in test_words:
        print("\n--------------------------")
        detect_gibberish(word)
