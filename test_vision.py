import base64, os, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
MODEL = "nvidia/Nemotron-3.5-Super-VL"

r = client.chat.completions.create(
    model=MODEL, max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print("TEXT TEST:", r.choices[0].message.content)

with open(sys.argv[1], "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = client.chat.completions.create(
    model=MODEL, max_tokens=500, temperature=0,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in two sentences."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}],
)
print("IMAGE TEST:", r.choices[0].message.content[:500])
print("finish_reason:", r.choices[0].finish_reason)