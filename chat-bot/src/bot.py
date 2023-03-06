import os
import telebot
import requests
from bs4 import BeautifulSoup as bs
from langchain import OpenAI
from gpt_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
)
from dotenv import load_dotenv as ld

ld()

bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))

aiKey = os.getenv("OPEN_AI_KEY")

url = "https://www.medium.com/@vefi.official"

index = None


def construct_index(directory_path):
    # set the maximum input size
    max_input_size = 4096

    # set the number of output tokens
    num_outputs = 256

    # set the maximum chunk overlap
    max_chunk_overlap = 20

    # set the chunk size limit
    chunk_size_limit = 600

    # define the Logical learning Machine
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
            openai_api_key=aiKey,
        )
    )

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    response = requests.get(url)

    soup = bs(response.content, "lxml")

    text = soup.get_text()

    if response.status_code == 200:
        documents = [{"text": text}]
    else:
        documents = []
    global index
    index = GPTSimpleVectorIndex(
        documents,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
        verbose=True,
    )

    index.save_to_disk("index.json")


@bot.message_handler(commands=["start", "help"])
def start_handler(message):
    bot.send_message(
        chat_id=message.chat.id,
        text="Hello! I'm Vefi, your virtual assistant. How may I be of assistance?",
    )


@bot.message_handler(func=lambda message: True)
def message_handler(message):
    query = message.text
    response = index.query(query, response_mode="compact", verbose=False)
    bot.send_message(chat_id=message.chat.id, text=response.response)


if __name__ == "__main__":
    if aiKey:
        construct_index(url)
        index = GPTSimpleVectorIndex.load_from_disk("index.json")
        bot.polling()
    else:
        print("OpenAI key not found. Set it in your environment variables.")




        OPEN_AI_KEY = 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

TELEGRAM_BOT_TOKEN=5869687994:AAHN38mSntZO9LQ6k8o6uBJsHlE6tcuRIWQ
