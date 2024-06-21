import chainlit as cl
import firebase_admin
from firebase_admin import credentials, firestore
from chatbot.model import Chat_bot
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] ="chat-bot"

chain = Chat_bot()
cred = credentials.Certificate("credentials/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

@cl.on_message
async def main(message: str):
    user_input = message.content

    response = chain.process_input(user_input, db)
    await cl.Message(
        content=response,
        author="Tee Chatbot"
    ).send()