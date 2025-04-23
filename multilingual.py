from datetime import datetime
import openai  # Updated import
import os

# Initialize OpenAI client
openai.api_key ="API_KEY"
# Replace with your actual key

def get_response(question, target_language):
    """Get response from OpenAI in the target language"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Respond ONLY in {target_language}."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}. Sorry, I couldn't generate a response in {target_language}."

class MultilingualChatbot:
    def __init__(self):
        self.language = "English"
        self.conversation = []

    def set_language(self, language):
        """Change the target language"""
        if language != self.language:
            self.language = language
            self.conversation = []  # Reset conversation on language change

    def add_message(self, question):
        """Process a user question and add to conversation history"""
        response = get_response(question, self.language)
        self.conversation.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "response": response
        })
        return response

# Example usage
if __name__ == "__main__":
    bot = MultilingualChatbot()
    
    # Test the bot
    bot.set_language("Spanish")
    response = bot.add_message("How are you?")
    print(f"Bot Response: {response}")
    
    # Print conversation history
    print("\nConversation History:")
    for msg in bot.conversation:
        print(f"[{msg['time']}] You: {msg['question']}")
        print(f"Bot: {msg['response']}\n")
