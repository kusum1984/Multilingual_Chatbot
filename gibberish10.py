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
import re

# Configuration
cfg = {
    "GIBBERISH_VALIDATION": {
        "system_prompt": """You are an advanced text analysis model trained to determine if a given text contains gibberish.
Gibberish is defined as text that lacks coherent meaning, logical structure, or context, often consisting of a random sequence of letters, numbers, or symbols.
Your task is to analyze the text and decide if it is mostly gibberish or mostly coherent.
IF the text is coherent, contains any nouns or meaningful content in any language, or is composed entirely of numbers, you must respond with only the word "Valid" and nothing else.
If the text is mostly gibberish and lacks coherent content or nouns, provide a brief reason in the local language explaining why it is considered gibberish.""",
        "user_prompt": """Analyze the following text and determine if it contains gibberish.
If the text contains any recognizable words, nouns, or coherent structure in any language, or is composed entirely of numbers, respond with only the word "Valid" and nothing else.
If the text is truly gibberish and lacks any coherent content, provide a brief reason in the native language explaining why:

{text}

Respond with only 'Valid' or the reason, without any additional explanation."""
    },
    "AzureOpenAI": {
        "GibberishValidation": {
            "Model": "gpt-35-turbo",
            "api_key": "your-azure-openai-api-key",
            "azure_endpoint": "https://your-resource.openai.azure.com/",
            "deployment_name": "your-deployment-name"
        }
    }
}

from openai import AzureOpenAI

# Setup client
client = AzureOpenAI(
    api_key=cfg["AzureOpenAI"]["GibberishValidation"]["api_key"],
    azure_endpoint=cfg["AzureOpenAI"]["GibberishValidation"]["azure_endpoint"],
    api_version="2023-12-01-preview"
)

# Error message mapping
language_errors = {
    "HI": "दिए गए हिंदी शब्द एक बकवास शब्द है।",
    "ES": "La palabra dada en español es un galimatías.",
    "PT": "A palavra dada em português é um palavreado sem sentido.",
    "ZH": "给出的中文是胡言乱语。",
    "JA": "与えられた日本語は意味不明な文字列です。",
    "DE": "Das gegebene deutsche Wort ist Kauderwelsch.",
    "FR": "Le mot français donné est un charabia."
}

def get_langcode_from_error(error_message):
    for code, message in language_errors.items():
        if message.strip() == error_message.strip():
            return code
    return "UNKNOWN"

# Gibberish detector
def check_gibberish(text):
    system_prompt = cfg["GIBBERISH_VALIDATION"]["system_prompt"]
    user_prompt = cfg["GIBBERISH_VALIDATION"]["user_prompt"].format(text=text)

    try:
        response = client.chat.completions.create(
            model=cfg["AzureOpenAI"]["GibberishValidation"]["Model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=1.0,
            n=1
        )
        result = response.choices[0].message.content.strip()

        if result.lower() == "valid":
            print(f"Input: {text}\nIs Gibberish: False\n")
        else:
            lang_code = get_langcode_from_error(result)
            print(f"Input: {text}\nIs Gibberish: True\nLangCode: {lang_code}\nError: {result}\n")
    except Exception as e:
        print(f"Error during validation: {e}")

# 🧪 Test Cases
test_inputs = [
    "केाीी",        # Hindi Gibberish
    "asdufhwq",     # English-like Gibberish
    "ñplsdja",      # Spanish-like Gibberish
    "ああああああ",   # Japanese Gibberish
    "12345",        # Valid numeric
    "Hallo",        # Valid German
    "kjashdfkjasd", # Random string
    "给出的中文是胡言乱语。",  # Chinese
]

for input_text in test_inputs:
    check_gibberish(input_text)

*****************
*******************
*****************

import re
import pandas as pd
from openai import AzureOpenAI

# Azure OpenAI client setup
client = AzureOpenAI(
    api_key="YOUR_API_KEY",
    azure_endpoint="https://YOUR_RESOURCE_NAME.openai.azure.com/",
    api_version="2023-07-01"
)

# Model name deployed on Azure
model_name = "YOUR_DEPLOYED_MODEL_NAME"  # e.g., "gpt-35-turbo"

# Error phrases mapped to language codes
lang_error_df = pd.DataFrame([
    {"lang_code": "HI", "error_phrase": "हिंदी शब्द एक बकवास"},
    {"lang_code": "ES", "error_phrase": "español es un galimatías"},
    {"lang_code": "PT", "error_phrase": "português é um palavreado"},
    {"lang_code": "ZH", "error_phrase": "中文是胡言乱语"},
    {"lang_code": "JA", "error_phrase": "日本語は意味不明"},
    {"lang_code": "DE", "error_phrase": "deutsche Wort ist Kauderwelsch"},
    {"lang_code": "FR", "error_phrase": "français donné est un charabia"}
])

# Prompts
system_prompt = (
    "You are an advanced text analysis model trained to determine if a given text contains gibberish. "
    "Gibberish is defined as text that lacks coherent meaning, logical structure, or context, often consisting "
    "of random sequences of letters, numbers, or symbols. Your task is to analyze the text and decide if it is mostly "
    "gibberish or mostly coherent. If it is coherent or composed entirely of numbers or meaningful words in any language, "
    "respond only with the word 'Valid'. If it's gibberish, provide a short sentence explaining why."
)

user_prompt_template = (
    "Analyze the following text and determine if it contains gibberish. "
    "If the text contains recognizable words or structure in any language, or is made up entirely of numbers, "
    "respond only with 'Valid'. If the text is truly gibberish and lacks coherence, provide a short reason why:\n\n{text}\n\n"
    "Respond with only 'Valid' or the reason, and nothing else."
)

# Function to detect gibberish and infer lang code from known error phrases
def check_gibberish(text):
    user_prompt = user_prompt_template.format(text=text)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=1.0
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.lower() == "valid":
            return False, "", ""  # Not gibberish

        # Language detection by matching known error phrases
        for _, row in lang_error_df.iterrows():
            if row["error_phrase"] in result:
                return True, row["lang_code"], result
        
        return True, "Unknown", result  # Gibberish but unknown language

    except Exception as e:
        return False, "", f"Error occurred: {str(e)}"

# Example test words
test_words = [
    "केाीी",      # Hindi gibberish
    "aaaaa",      # Portuguese gibberish
    "asdkjha",    # German gibberish
    "漢字",       # Chinese
    "これは",      # Japanese
    "bonjour",    # Valid French
    "hello123",   # Valid English
]

# Test execution
for word in test_words:
    is_gibberish, lang_code, error = check_gibberish(word)
    if is_gibberish:
        print(f"Gibberish Word: {word} | LangCode: {lang_code} | Error: {error}")
    else:
        print(f"'{word}' is valid.")
