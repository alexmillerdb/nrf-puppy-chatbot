import os

import chainlit as cl
from dotenv import load_dotenv

from chains import ChainlitChat, BatchResponse
from chains.partner_bot import PartnerBot
from chains.puppy_bot_local import PuppyBot

# from chains.puppy_bot import PuppyBotCompletion

load_dotenv()

PUPPY_BOT_NAME = "Puppy Bot"
PARTNER_BOT_NAME = "Partner Bot"


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name=PUPPY_BOT_NAME,
            markdown_description="Puppy Bot Rag.",
            icon="public/dog.png",
        ),
        cl.ChatProfile(
            name=PARTNER_BOT_NAME,
            markdown_description="Partner Bot Rag.",
            icon="public/partner.png",
        ),
    ]


@cl.on_chat_start
async def chat_init():
    active_chat_profile = cl.user_session.get("chat_profile")

    if active_chat_profile == PUPPY_BOT_NAME:
        completion = PuppyBot()
    else:
        completion = PartnerBot()

    cl.user_session.set("completion", completion)

    await completion.intro_message().send()

    # if active_chat_profile == PUPPY_BOT_NAME:
    #     await cl.Message("Welcome to the Databricks NRF Puppy Chat Bot in collaboration with "
    #                      "PetSmart! Ask me a question about how to best care for a puppy or a dog 🐶.").send()
    # else:
    #     await cl.Message("Welcome to the Databricks NRF Partner Assistant! "
    #                      "Please feel free to ask me questions about the various data, "
    #                      "technology and service partners in the Databricks partner ecosystem.").send()


def update_chat_history(message: cl.Message, response: cl.Message):
    history = cl.user_session.get("history")
    if history is None:
        history = []
    history.append({"role": "user", "content": f"{message.content}"})
    history.append({"role": "assistant", "content": f"{response.content}"})
    cl.user_session.set("history", history)


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    completion: ChainlitChat = cl.user_session.get("completion")
    # Send a response back to the user
    resp_msg = cl.Message(content="")
    completion_msg = await completion.complete(message.content, message, resp_msg)
    if isinstance(completion_msg, BatchResponse):
        # if its batch lets plop that all into the content for response
        resp_msg.content = completion_msg.response
    await resp_msg.update()
    update_chat_history(message, resp_msg)
