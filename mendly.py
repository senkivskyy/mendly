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
"part_keywords": a list of 3 to 6 alternative names for the suspected part, as parts stores would call it (for a fridge light bulb: "light", "lamp", "LED", "bulb")
"steps": an ordered list of short step strings
"search_queries": a list of 2 or 3 different web search queries to find the replacement part and its price. Include the brand and model number when known, and words like "replacement part" and "buy"""
    return ask_json(REASONING_MODEL, prompt, max_tokens=2500)


PLUMBING_DOMAINS = [
    "plumbersstock.com", "firstsupply.com", "ferguson.com", "fergusonhome.com", "build.com",
    "faucetdirect.com", "supplyhouse.com", "homedepot.com", "lowes.com", "deltafaucet.com",
    "brizo.com", "moen.com", "kohler.com", "efaucets.com",
]


SHOP_DOMAINS = [
    "repairclinic.com", "partselect.com", "amazon.com", "ebay.com",
    "searspartsdirect.com", "appliancepartspros.com", "walmart.com",
    "homedepot.com", "autozone.com", "rockauto.com", "partsgeek.com",
]
SHOP_DOMAINS = SHOP_DOMAINS + [d for d in PLUMBING_DOMAINS if d not in SHOP_DOMAINS]


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
        if model and page and matches_model(page, model):
            p["fits_model"] = True


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def model_variants(model):
    """Labels often add suffixes ("LFDS22520S/04", "63075LF-SSLHP") that parts sites drop.
    Try the full number first, then shorter forms."""
    model = (model or "").strip()
    out = []
    for v in (model, re.split(r"[/\s]", model)[0], re.split(r"[/\s-]", model)[0]):
        if len(norm(v)) >= 5 and v not in out:
            out.append(v)
    return out


def matches_model(text, model):
    t = norm(text)
    return any(norm(v) in t for v in model_variants(model))


# Sites that publish a parts list for each specific model number.
MODEL_DOMAINS = [
    "partselect.com", "repairclinic.com", "searspartsdirect.com",
    "appliancepartspros.com", "ereplacementparts.com", "fix.com", "encompass.com",
    "plumbersstock.com", "firstsupply.com", "ferguson.com", "fergusonhome.com", "build.com",
    "supplyhouse.com", "faucetdirect.com",
]
STOP_WORDS = {"the", "and", "assembly", "part", "parts", "replacement", "for", "with", "broken", "kit"}


def keywords_for(wanted, extra=None):
    words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", wanted or "")} - STOP_WORDS
    words |= {str(w).lower() for w in (extra or []) if str(w).strip()}
    return sorted(words)


def relevant_excerpt(text, keywords, limit=6000):
    """Keep only the lines of a long parts page that mention what we need."""
    lines = (text or "").splitlines()
    keep = set()
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in keywords):
            keep.update(range(max(0, i - 1), min(len(lines), i + 3)))
    return "\n".join(lines[i] for i in sorted(keep))[:limit]


def model_page_parts(model, brand, wanted, keywords):
    """Find the parts list for this exact model, and read the right part off it.

    Returns (parts, pages). These parts are known to belong to the model, so they
    are better than generic search results (for example a real fridge light, not a
    generic bulb).
    """
    pages = []
    for variant in model_variants(model):  # full number first, then shorter forms
        try:
            hits = tavily.search(
                f"{brand or ''} {variant} parts list".strip(), max_results=8, include_domains=MODEL_DOMAINS
            )["results"]
        except Exception as e:
            print(f"  (model page search failed: {e})")
            continue
        pages = [h for h in hits if norm(variant) in norm(h["url"] + h["title"])][:3]
        if pages:
            break
    if not pages:  # no parts-site page: look for any page that names this exact model
        for variant in model_variants(model):
            try:
                hits = tavily.search(f"{brand or ''} {variant} replacement parts".strip(), max_results=8)["results"]
            except Exception:
                continue
            pages = [
                h for h in hits
                if matches_model(h["url"] + " " + h["title"] + " " + h["content"], variant)
                and "youtube.com" not in h["url"]
            ][:3]
            if pages:
                break
    if os.getenv("MENDLY_DEBUG"):
        print("  [debug] model pages:", [h["url"] for h in pages])
    if not pages:
        return [], []
    try:
        extracted = tavily.extract(urls=[h["url"] for h in pages]).get("results", [])
    except Exception:
        extracted = []
    raw_by_url = {e.get("url"): e.get("raw_content", "") for e in extracted}
    blocks = []
    for h in pages:
        excerpt = relevant_excerpt(raw_by_url.get(h["url"]) or h["content"], keywords)
        if excerpt:
            blocks.append(f"PAGE: {h['url']}\n{excerpt}")
    if not blocks:
        return [], pages

    prompt = f"""These are parts-list pages for the exact model {brand or ''} {model}.
We need: {wanted} (also called: {', '.join(keywords)}).
List ONLY parts from the text below that match what we need. The part must be a
component made for this appliance, not a generic item. Include the part number and
price only if the text states them. Never invent anything. The url must be one of the
PAGE urls, copied exactly.
Reply with ONLY a JSON object: {{"parts": [{{"name": "...", "part_number": null, "price": null, "url": "..."}}]}}

""" + "\n\n".join(blocks)
    for llm in (FAST_MODEL, REASONING_MODEL):
        try:
            found = ask_json(llm, prompt, max_tokens=4000).get("parts", [])
            break
        except (ValueError, json.JSONDecodeError):
            found = None
    if not found:
        return [], pages
    valid = {h["url"] for h in pages}
    out = [p for p in found if p.get("url") in valid]
    for p in out:
        p["fits_model"] = True  # taken from this model's own parts list
    return out, pages


def find_series(brand, model):
    """Many models belong to a named product line (for example Brizo "Odin"). Parts pages
    are often listed under that name, not the model number. Read the name from search results."""
    if not model:
        return None
    try:
        hits = tavily.search(f"{brand or ''} {model_variants(model)[-1]}".strip(), max_results=6)["results"]
    except Exception:
        return None
    hits = [h for h in hits if matches_model(h["url"] + " " + h["title"] + " " + h["content"], model)]
    if not hits:
        return None
    text = "\n\n".join(f"TITLE: {h['title']}\nTEXT: {h['content'][:500]}" for h in hits)
    prompt = f"""Model number {brand or ''} {model}. From ONLY the text below, what is the product
line or series name of this model (a short name such as "Odin" or "Solna")?
Reply with ONLY a JSON object: {{"series": null}} or {{"series": "Name"}}. Do not guess.

{text}"""
    try:
        s = ask_json(FAST_MODEL, prompt, max_tokens=800).get("series")
    except (ValueError, json.JSONDecodeError):
        return None
    return s.strip() if isinstance(s, str) and 2 <= len(s.strip()) <= 30 else None


def find_parts(queries, wanted, model=None, brand=None, keywords=None):
    """Step 3 (Tavily + fast model): where do I buy the part, and for how much?"""
    if isinstance(queries, str):
        queries = [queries]
    keywords = keywords_for(wanted, keywords)

    # 1) Best source: the parts list for this exact model number.
    model_parts, model_pages = ([], [])
    if model:
        model_parts, model_pages = model_page_parts(model, brand, wanted, keywords)

    # 2) Wider search of parts stores.
    queries = list(queries)[:3]
    series = find_series(brand, model)
    if os.getenv("MENDLY_DEBUG"):
        print("  [debug] series:", series)
    if series:
        queries.insert(0, f"{brand or ''} {series} {wanted} replacement part".strip())
    if model:  # always search the exact model number first
        who = f"{brand} " if brand else ""
        queries.insert(0, f'{who}"{model}" {wanted} replacement part')
    elif brand:
        queries.insert(0, f"{brand} {wanted} replacement part")
    seen, results = set(), list(model_pages)
    seen.update(h["url"] for h in model_pages)
    for q in queries[:5]:
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

    parts = []
    shop_results = [r for r in results if r not in model_pages]
    if shop_results:
        snippets = "\n\n".join(
            f"URL: {r['url']}\nTITLE: {r['title']}\nTEXT: {r['content']}" for r in shop_results
        )
        prompt = f"""The part we need is: {wanted} (also called: {', '.join(keywords)}).
From ONLY the search results below, list replacement options for THAT part only.
Skip unrelated parts (glass, trays, switches, and so on) and general category pages
unless nothing better exists. Skip WHOLE products such as complete sinks, faucets or
appliances, and skip generic items (for example an ordinary light bulb) when the result
does not say it is made for {brand or 'this'} {model or 'appliance'}. We only want the small
replacement component. If a result says which models it fits, copy that into a "fits"
field (a short string), otherwise null. Never invent a price, part number or URL. Use null
when the results do not state it. Copy each URL exactly as given.
Reply with ONLY a JSON object shaped like:
{{"parts": [{{"name": "...", "part_number": null, "price": null, "url": "...", "fits": null}}]}}

{snippets}"""
        for llm in (FAST_MODEL, REASONING_MODEL):  # fall back to the bigger model
            try:
                parts = ask_json(llm, prompt, max_tokens=4000).get("parts", [])
                break
            except (ValueError, json.JSONDecodeError):
                continue
        else:
            print("  (could not read the parts list, skipping)")

    # Merge: parts from the model's own list come first; drop duplicates.
    have = {norm(p.get("part_number")) for p in model_parts if p.get("part_number")}
    parts = model_parts + [p for p in parts if not p.get("part_number") or norm(p["part_number"]) not in have]
    if not parts:
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
                price = PRICE_RE.search(h.get("content", ""))
                if price and norm(p["part_number"]) in norm(h["title"] + h["content"] + h["url"]):
                    p["price"], p["url"] = price.group().replace(" ", ""), h["url"]
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
            p["fits_model"] = bool(p.get("fits_model")) or matches_model(text_by_url.get(p["url"], ""), model)

    # Last resort: open the store pages themselves to read the price.
    fill_prices_from_pages(parts, model)

    # Matches first, then cheapest; parts with no price go last. Keep the list short.
    parts.sort(key=lambda p: (not p.get("fits_model"), price_value(p.get("price")) is None, price_value(p.get("price")) or 0))
    # If we found parts confirmed for this model, drop the unconfirmed generic ones.
    if any(p.get("fits_model") for p in parts):
        parts = [p for p in parts if p.get("fits_model")]
    return parts[:6]


NEW_DOMAINS = SHOP_DOMAINS + ["lowes.com", "bestbuy.com", "target.com", "wayfair.com", "costco.com"]


def find_new_price(info):
    """What would a brand-new one cost? Used for the fix-or-replace comparison."""
    item, model, brand = info.get("item"), info.get("model_number"), info.get("brand")
    if not item:
        return None
    query = " ".join(str(x) for x in (brand, model, item) if x) + " buy new price"
    try:
        results = tavily.search(query, max_results=6, include_domains=NEW_DOMAINS)["results"]
    except Exception as e:
        print(f"  (new-price search failed: {e})")
        return None
    if not results:
        return None
    snippets = "\n\n".join(
        f"URL: {r['url']}\nTITLE: {r['title']}\nTEXT: {r['content']}" for r in results
    )
    prompt = f"""We want the price of a NEW, complete {' '.join(str(x) for x in (brand, model, item) if x)}.
From ONLY the search results below, pick the single best match: a whole new product,
not a replacement part or accessory. Never invent a price or URL. Use null if the
results do not state a price. Copy the URL exactly.
Reply with ONLY a JSON object: {{"name": "...", "price": null, "url": "..."}}

{snippets}"""
    try:
        data = ask_json(FAST_MODEL, prompt, max_tokens=2500)
    except (ValueError, json.JSONDecodeError):
        return None
    by_url = {r["url"]: r for r in results}
    if data.get("url") not in by_url:
        return None
    hit = by_url[data["url"]]
    price = data.get("price")
    if price_value(price) is None:  # snippet had no price: read it from the page
        try:
            page = tavily.extract(urls=[hit["url"]]).get("results", [{}])[0].get("raw_content", "")
            price = price_from_page(page)
        except Exception:
            price = None
    if price_value(price) is None:
        return None
    text = hit["title"] + " " + hit["content"] + " " + hit["url"]
    return {
        "name": data.get("name") or hit["title"],
        "price": price,
        "url": hit["url"],
        "exact": bool(model) and matches_model(text, model),
    }


def fix_or_replace(plan, parts, new):
    """Compare the cost of the part with the cost of a new one. Plain arithmetic, no AI."""
    fit = [price_value(p["price"]) for p in parts if p.get("fits_model") and price_value(p.get("price")) is not None]
    anyp = [price_value(p["price"]) for p in parts if price_value(p.get("price")) is not None]
    repair = min(fit) if fit else (min(anyp) if anyp else None)
    cost_new = price_value(new.get("price")) if new else None
    if repair is None or cost_new is None:
        return None

    ratio = repair / cost_new
    hard = (plan.get("difficulty") or "").lower() == "hard"
    if ratio <= 0.4 and not (hard and ratio > 0.25):
        call, headline = "fix", "Fix it."
    elif ratio <= 0.75:
        call, headline = "close", "Close call."
    else:
        call, headline = "replace", "Consider replacing it."

    saved = cost_new - repair
    if call == "fix":
        detail = f"The part costs about ${repair:,.2f} and a new one about ${cost_new:,.2f}. Repairing saves around ${saved:,.0f}."
    elif call == "close":
        detail = "The part costs a good share of a new one. Fix it if you're comfortable with the repair; otherwise replacing may be easier."
    else:
        detail = "The part costs close to a new one, so replacing may be the better deal unless you value the original."
    notes = ["Part price only. Your time and any tools are not included."]
    if not fit:
        notes.append("Confirm the part fits your exact model before you buy.")
    if new and not new.get("exact"):
        notes.append("The new-item price is for the closest match we found.")
    if hard:
        notes.append("This is a harder repair; a professional may add to the cost.")
    return {
        "call": call, "headline": headline, "detail": detail, "notes": notes,
        "repair_cost": repair, "new_cost": cost_new,
        "new_name": new.get("name"), "new_url": new.get("url"),
    }


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
    parts = find_parts(queries, info.get("suspected_part") or "the broken part", info.get("model_number"), info.get("brand"), plan.get("part_keywords"))

    sheet = job_sheet(info, plan, parts)
    print("\n" + sheet)
    with open("job_sheet.md", "w") as f:
        f.write(sheet)
    print("\nSaved to job_sheet.md")


if __name__ == "__main__":
    main()
