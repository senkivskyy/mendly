# Mendly

**Point at what's broken and get the fix, step by step, plus the right replacement part at the best price.**

Mendly is an AI repair assistant. Take a photo of something broken, add a short description, and Mendly identifies the item, works out the likely faulty part, writes safety-first repair steps, and finds where to buy the replacement with prices and links.

- **Live demo:** https://mendly-8qtw.onrender.com
- **Demo video:** YOUR_YOUTUBE_URL_HERE
- **Track:** Best Apps and Agents (Nebius x NVIDIA Global AI Hackathon)

## What it does

1. **Look.** You upload a photo (any phone photo, including iPhone HEIC) and optionally describe the problem.
2. **Identify.** A vision model names the item, the likely problem and the suspected part, and says how confident it is.
3. **Ask once.** Parts only fit specific models, so Mendly asks for the brand and model number (printed on a sticker or label) before it searches. You can skip it.
4. **Plan.** A reasoning model writes the difficulty, time estimate, tools, safety notes and numbered steps. Safety notes come first, and high-voltage or gas work tells the user to stop and call a professional.
5. **Find the part.** Mendly searches live on the web for the replacement part, reads the store pages for prices, and checks whether each result mentions your exact model number. Results that match are marked and listed first; the rest say "check fit before buying".

## Models and tools

Every model call runs on **Nebius Token Factory** through its OpenAI-compatible API. Mendly uses three **NVIDIA Nemotron** models:

| Step | Model |
| --- | --- |
| Photo understanding | `nvidia/Nemotron-3.5-Super-VL` |
| Repair reasoning and planning | `nvidia/nemotron-3-super-120b-a12b` |
| Fast extraction of parts and prices | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` |

**Tavily** provides live web search and page extraction at runtime, which is how Mendly finds current parts, prices and store links.

## Built to be trustworthy

- **No invented prices, part numbers or links.** The extraction step may only use text from the search results, and any part whose URL did not come from the search is discarded.
- **Fit is checked, not assumed.** A part is only labeled "Matches your model" if the page actually mentions your model number.
- **Safety first.** Safety warnings appear above the steps, and the plan never tells users to touch high-voltage parts.
- **Honest about uncertainty.** If the vision model is not confident, Mendly asks a clarifying question. If no price is found, it says so and links to the store.

## Run it locally

You need Python 3.10+ and API keys for [Nebius Token Factory](https://tokenfactory.nebius.com) and [Tavily](https://tavily.com).

```bash
git clone https://github.com/senkivskyy/mendly.git
cd mendly
pip install -r requirements.txt
cp .env.example .env        # then paste your two keys into .env
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

To try the pipeline from the command line without the web app:

```bash
python3 mendly.py photo.jpg "the door won't latch"
```

## Configuration

Set these in `.env` (or as environment variables on your host):

| Variable | Purpose | Default |
| --- | --- | --- |
| `NEBIUS_API_KEY` | Nebius Token Factory key | required |
| `TAVILY_API_KEY` | Tavily key | required |
| `PER_IP_DAILY` | Free repairs per visitor per day | 8 |
| `GLOBAL_DAILY` | Free repairs for everyone per day | 300 |

The daily limits protect the demo's Token Factory credits.

## Project layout

```
app.py             FastAPI server, upload handling, daily usage limits
mendly.py          The pipeline: identify -> plan -> find parts
static/index.html  Mobile-first web interface
render.yaml        One-click deploy config for Render
```

## Limitations

- Prices are read from store pages and can be missing or out of date. Always confirm the price and fit before buying.
- Usage counters are kept in memory, so they reset when the server restarts.
- Mendly gives general guidance, not professional advice.

## License

MIT. See [LICENSE](LICENSE).
