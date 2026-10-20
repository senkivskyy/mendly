import os
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
for m in client.models.list():
    if "nemotron" in m.id.lower():
        print(m.id)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
print(tavily.search("replacement dishwasher drain pump price")["results"][0])