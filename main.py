import os
import requests
import chainlit as cl
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    function_tool
)

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env file")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Tool to get price from Binance
@function_tool
def get_crypto_price(symbol: str) -> str:
    """
    Fetch the current price of a cryptocurrency from Binance.
    If the symbol is 'BTC', 'ETH', etc., it will automatically convert it to 'BTCUSDT', 'ETHUSDT', etc.
    """
    try:
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        response.raise_for_status()
        price = response.json()["price"]
        return f"💰 The current price of **{symbol}** is **${float(price):,.2f}**"
    except Exception as e:
        return f"❌ Failed to fetch price for {symbol}. Error: {e}"

# Define the agent
crypto_agent = Agent(
    name="CryptoDataAgent",
    instructions="""
You are a crypto assistant.

When the user asks for a coin price (e.g., BTC, ETH, SOL), 
use the `get_crypto_price` tool with the correct symbol.

Assume the user always means the USDT pair (e.g., BTC → BTCUSDT).
Never ask which exchange — always default to Binance and add 'USDT' if needed.
""",
    tools=[get_crypto_price]
)

# Chainlit message handler
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(
        crypto_agent,
        input=message.content,
        run_config=config
    )
    await cl.Message(content=result.final_output).send()
