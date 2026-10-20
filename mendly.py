import base64
import io
import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pillow_heif import register_heif_opener
from tavily import TavilyClient

load_dotenv()
register_heif_opener()  # lets Pillow open iPhone HEIC photos

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Swap these if you want to try other Nemotron models from your model list.
VISION_MODEL = "nvidia/Nemotron-3.5-Super-VL"
REASONING_MODEL = "nvidia/nemotron-3-super-120b-a12b"
FAST_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def image_to_data_url(path):
    """Open any photo (JPEG, PNG, HEIC), shrink it, return a JPEG data URL."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(model, content, max_tokens=1500):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def parse_json(text):
    """Pull the JSON object out of a model reply, ignoring any <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output:\n" + text[:300])
    return json.loads(text[start : end + 1])


def ask_json(model, content, max_tokens=1500, tries=3):
    """Ask for JSON; if the model's JSON is malformed, try again."""
    last = None
    for _ in range(tries):
        try:
            return parse_json(ask(model, content, max_tokens))
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            if os.getenv("MENDLY_DEBUG"):
                print(f"  [debug] bad JSON from {model}: {e}")
    raise last


def identify(image_url, description):
    """Step 1 (vision model): what is it, and what is probably broken?"""
    prompt = f"""You are an expert repair technician. Look at the photo and the owner's description.
Owner's description: {description or "none given"}

Reply with ONLY a JSON object with these keys:
"item": what the product is
"brand": the brand, or null
"model_number": only if it is visible in the photo, otherwise null
"problem": what is probably wrong
"suspected_part": the part most likely to need replacing
"confidence": "low", "medium" or "high"
"clarifying_question": one question that would raise your confidence, or null"""
    return ask_json(
        VISION_MODEL,
        [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    )


def plan_repair(info):
    """Step 2 (reasoning model): how do I fix it?"""
    prompt = f"""You are a careful repair expert. Based on this diagnosis, write a repair plan.
Diagnosis: {json.dumps(info)}

Reply with ONLY a JSON object with these keys:
"difficulty": "easy", "moderate" or "hard"
"time_estimate": a short string
"tools": a list of tools needed
"safety_notes": a list of safety warnings (electrical, gas, fuel, high voltage, and when to call a professional). Never tell the user to discharge a capacitor or touch high-voltage parts; say instead that those parts stay dangerous after unplugging and to stop and call a professional if the repair requires them
"brand": the brand if the model number clearly indicates it, otherwise null
"steps": an ordered list of short step strings
"search_queries": a list of 2 or 3 different web search queries to find the replacement part and its price. Include the brand and model number when known, and words like "replacement part" and "buy"""
    return ask_json(REASONING_MODEL, prompt, max_tokens=2500)


SHOP_DOMAINS = [
    "repairclinic.com", "partselect.com", "amazon.com", "ebay.com",
    "searspartsdirect.com", "appliancepartspros.com", "walmart.com",
    "homedepot.com", "autozone.com", "rockauto.com", "partsgeek.com",
]


def price_value(price):
    """Turn '$12.99' into 12.99 so we can sort; None if there is no number."""
    m = re.search(r"\d+(?:[.,]\d+)?", str(price or "").replace(",", ""))
    return float(m.group()) if m else None


PRICE_RE = re.compile(r"\$\s?\d{1,4}(?:,\d{3})*(?:\.\d{2})")


def price_from_page(text, part_number=None):
    """Pick the price for one product out of a store page's text, or None if unsure."""
    text = text or ""
    if part_number and part_number.lower() in text.lower():
        i = text.lower().index(part_number.lower())
        window = text[max(0, i - 300): i + 500]
        m = PRICE_RE.search(window)
        if m:
            return m.group().replace(" ", "")
    prices = list(dict.fromkeys(PRICE_RE.findall(text)))
    # A single product page shows only a few prices; a category page shows many.
    if 1 <= len(prices) <= 3:
        return prices[0].replace(" ", "")
    return None


def fill_prices_from_pages(parts, model=None):
    """Open the store pages of parts that still have no price and read the price."""
    need = [p for p in parts if not p.get("price") and p.get("url")][:5]
    if not need:
        return
    try:
        pages = tavily.extract(urls=[p["url"] for p in need]).get("results", [])
    except Exception as e:
        print(f"  (page lookup failed: {e})")
        return
    by_url = {pg.get("url"): pg.get("raw_content", "") for pg in pages}
    for p in need:
        page = by_url.get(p["url"], "")
        price = price_from_page(page, p.get("part_number"))
        if price:
            p["price"] = price
        if model and page and norm(model) in norm(page):
            p["fits_model"] = True


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def find_parts(queries, wanted, model=None, brand=None):
    """Step 3 (Tavily + fast model): where do I buy the part, and for how much?"""
    if isinstance(queries, str):
        queries = [queries]
    queries = list(queries)[:3]
    if model:  # always search the exact model number first
        who = f"{brand} " if brand else ""
        queries.insert(0, f'{who}"{model}" {wanted} replacement part')
    elif brand:
        queries.insert(0, f"{brand} {wanted} replacement part")
    seen, results = set(), []
    for q in queries[:4]:
        try:
            found = tavily.search(
                q, max_results=5, include_domains=SHOP_DOMAINS
            )["results"]
        except Exception as e:
            print(f"  (search failed for '{q}': {e})")
            continue
        for r in found:
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    if not results:
        return []

    snippets = "\n\n".join(
        f"URL: {r['url']}\nTITLE: {r['title']}\nTEXT: {r['content']}" for r in results
    )
    prompt = f"""The part we need is: {wanted}.
From ONLY the search results below, list replacement options for THAT part only.
Skip unrelated parts (glass, trays, switches, and so on) and general category pages
unless nothing better exists. Skip WHOLE products such as complete sinks, faucets or
appliances: we only want the small replacement component. If a result says which models
it fits, copy that into a "fits" field (a short string), otherwise null. Never invent a price, part number or URL. Use null when the results do not state it.
Copy each URL exactly as given. Skip results that are not a page selling or describing a part.
Reply with ONLY a JSON object shaped like:
{{"parts": [{{"name": "...", "part_number": null, "price": null, "url": "...", "fits": null}}]}}

{snippets}"""
    parts = None
    for model in (FAST_MODEL, REASONING_MODEL):  # fall back to the bigger model
        try:
            parts = ask_json(model, prompt, max_tokens=4000).get("parts", [])
            break
        except (ValueError, json.JSONDecodeError):
            continue
    if parts is None:
        print("  (could not read the parts list, skipping)")
        return []

    # Second pass: for parts with a part number but no price, search that part number.
    for p in parts:
        if p.get("part_number") and not p.get("price"):
            try:
                hits = tavily.search(
                    f"{p['part_number']} {p.get('name', '')} price",
                    max_results=5, include_domains=SHOP_DOMAINS,
                )["results"]
            except Exception:
                continue
            for h in hits:
                m = re.search(r"\$\s?\d{1,4}(?:\.\d{2})", h.get("content", ""))
                if m and p["part_number"].lower() in (h["title"] + h["content"] + h["url"]).lower():
                    p["price"], p["url"] = m.group().replace(" ", ""), h["url"]
                    results.append(h)
                    break

    # Keep only URLs that really came from the search, never video links.
    valid = {r["url"] for r in results}
    parts = [
        p for p in parts
        if p.get("url") in valid and "youtube.com" not in p["url"] and "youtu.be" not in p["url"]
    ]
    # Does the page mention the customer's model number? If so, it likely fits.
    if model:
        text_by_url = {r["url"]: r["title"] + " " + r["content"] + " " + r["url"] for r in results}
        for p in parts:
            p["fits_model"] = norm(model) in norm(text_by_url.get(p["url"], ""))

    # Last resort: open the store pages themselves to read the price.
    fill_prices_from_pages(parts, model)

    # Cheapest first; parts with no price go last.
    parts.sort(key=lambda p: (not p.get("fits_model"), price_value(p.get("price")) is None, price_value(p.get("price")) or 0))
    return parts


def job_sheet(info, plan, parts):
    lines = [
        f"# Repair job: {info.get('item')}",
        "",
        f"**Brand:** {info.get('brand') or 'Not identified'}  |  "
        f"**Model:** {info.get('model_number') or 'Not identified (check the label or sticker on the item)'}",
        f"**Problem:** {info.get('problem')}",
        f"**Suspected part:** {info.get('suspected_part')} "
        f"(confidence: {info.get('confidence')})",
        f"**Difficulty:** {plan.get('difficulty')}  |  **Time:** {plan.get('time_estimate')}",
        "",
        "## Tools",
        *[f"- {t}" for t in plan.get("tools", [])],
        "",
        "## Safety",
        *[f"- {s}" for s in plan.get("safety_notes", [])],
        "",
        "## Steps",
        *[f"{i}. {s}" for i, s in enumerate(plan.get("steps", []), 1)],
        "",
        "## Parts",
    ]
    if not parts:
        lines.append("- No matching parts found. Try adding the model number and run again.")
    for p in parts:
        price = p.get("price") or "check price at link"
        pn = f" (#{p['part_number']})" if p.get("part_number") else ""
        fit = "fits your model" if p.get("fits_model") else "check fit"
        lines.append(f"- {p.get('name')}{pn}: {price} ({fit}) - {p.get('url')}")
    lines += ["", "_Check that a part fits your exact model before buying._"]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 mendly.py photo.jpg "what seems to be wrong"')
    path = sys.argv[1]
    description = " ".join(sys.argv[2:])

    image_url = image_to_data_url(path)
    print("Identifying the item...")
    info = identify(image_url, description)

    if info.get("confidence") == "low" and info.get("clarifying_question"):
        answer = input(f"\n{info['clarifying_question']}\n> ")
        info = identify(image_url, f"{description} {answer}")

    if not info.get("model_number"):
        model = input("\nModel number (on a sticker or label, Enter to skip)\n> ").strip()
        if model:
            info["model_number"] = model

    print("Planning the repair...")
    plan = plan_repair(info)
    if not info.get("brand") and plan.get("brand"):
        info["brand"] = plan["brand"]

    print("Searching for parts...")
    base = " ".join(
        str(x) for x in (info.get("brand"), info.get("model_number"), info.get("item"),
                         info.get("suspected_part")) if x
    )
    queries = plan.get("search_queries") or []
    if isinstance(queries, str):
        queries = [queries]
    queries = queries or [f"{base} replacement part buy"]
    parts = find_parts(queries, info.get("suspected_part") or "the broken part", info.get("model_number"), info.get("brand"))

    sheet = job_sheet(info, plan, parts)
    print("\n" + sheet)
    with open("job_sheet.md", "w") as f:
        f.write(sheet)
    print("\nSaved to job_sheet.md")


if __name__ == "__main__":
    main()
