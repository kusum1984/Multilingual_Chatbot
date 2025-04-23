import streamlit as st
from datetime import datetime
from openai import OpenAI
import os

# Initialize OpenAI client - IMPORTANT: Replace with your actual API key management
# For production, use environment variables or secrets management
OPENAI_API_KEY = "sk-proj-...YOUR_API_KEY..."  # Replace with your actual key

def get_response(question, target_language):
    """Get response from OpenAI in the target language"""
    try:
        client = OpenAI(api_key="API_KEY")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Respond ONLY in {target_language}."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return f"Sorry, I couldn't generate a response in {target_language}."

# --- Initialize Session State (Stores user data) ---
if "language" not in st.session_state:
    st.session_state.language = "English"
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- Streamlit UI ---
st.title("🌍 Multilingual Chatbot")
st.markdown("Ask anything in any language, get answers in **your selected language!**")

# --- Language Selection with unique key ---
language = st.selectbox(
    "**Choose your language:**",
    ["English", "Spanish", "French", "German", "Hindi"],
    index=["English", "Spanish", "French", "German", "Hindi"].index(st.session_state.language),
    key="language_selector"  # Added unique key here
)

# Update language if changed
if language != st.session_state.language:
    st.session_state.language = language
    st.session_state.conversation = []  # Reset conversation on language change
    st.rerun()

# --- Display Conversation History ---
st.subheader("🗨️ Conversation")

if not st.session_state.conversation:
    st.info("No messages yet. Ask a question below!")
else:
    for msg in st.session_state.conversation:
        st.markdown(f"**You ({msg['time']}):** {msg['question']}")
        st.markdown(f"**Bot:** {msg['response']}")
        st.divider()

# --- User Input ---
user_input = st.chat_input("Type your question here...", key="user_input")

if user_input:
    # Get response in selected language
    response = get_response(user_input, st.session_state.language)
    
    # Store in conversation history
    st.session_state.conversation.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": user_input,
        "response": response
    })
    
    # Refresh to show new message
    st.rerun()