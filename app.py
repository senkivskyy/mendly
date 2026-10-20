"""Mendly web app: upload a photo of something broken, get the fix and the part.

Run locally:  uvicorn app:app --reload
"""
import io
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import mendly  # the pipeline: identify -> plan_repair -> find_parts

# ---- Usage limits (protect the Token Factory / Tavily credits) -------------
PER_IP_DAILY = int(os.getenv("PER_IP_DAILY", "8"))        # repairs per visitor per day
GLOBAL_DAILY = int(os.getenv("GLOBAL_DAILY", "300"))      # repairs for everyone per day
ATTEMPTS_PER_IP_DAILY = int(os.getenv("ATTEMPTS_PER_IP_DAILY", "40"))  # raw requests
MAX_UPLOAD_MB = 12

BASE = Path(__file__).parent
app = FastAPI(title="Mendly")

_lock = threading.Lock()
_day = None
_used_ip, _attempts_ip, _used_global = {}, {}, 0


def _reset_if_new_day():
    global _day, _used_ip, _attempts_ip, _used_global
    today = datetime.now(timezone.utc).date()
    if today != _day:
        _day, _used_ip, _attempts_ip, _used_global = today, {}, {}, 0


def client_ip(request: Request):
    # Behind a host's proxy the real address arrives in X-Forwarded-For.
    # It can be spoofed, which is why the global cap exists as a backstop.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def try_consume(ip):
    """Reserve one repair. Returns (ok, message)."""
    global _used_global
    with _lock:
        _reset_if_new_day()
        _attempts_ip[ip] = _attempts_ip.get(ip, 0) + 1
        if _attempts_ip[ip] > ATTEMPTS_PER_IP_DAILY:
            return False, "Too many requests today. Please try again tomorrow."
        if _used_global >= GLOBAL_DAILY:
            return False, "Mendly has hit its free daily limit. Please try again tomorrow."
        if _used_ip.get(ip, 0) >= PER_IP_DAILY:
            return False, f"You've used your {PER_IP_DAILY} free repairs for today. Come back tomorrow."
        _used_ip[ip] = _used_ip.get(ip, 0) + 1
        _used_global += 1
        return True, ""


def refund(ip):
    global _used_global
    with _lock:
        if _used_ip.get(ip, 0) > 0:
            _used_ip[ip] -= 1
            _used_global = max(0, _used_global - 1)


def remaining(ip):
    with _lock:
        _reset_if_new_day()
        return max(0, min(PER_IP_DAILY - _used_ip.get(ip, 0), GLOBAL_DAILY - _used_global))


def error(status, message):
    return JSONResponse({"status": "error", "message": message}, status_code=status)


# ---- API ---------------------------------------------------------------------
@app.get("/api/usage")
def usage(request: Request):
    return {"remaining": remaining(client_ip(request))}


@app.post("/api/analyze")
def analyze(
    request: Request,
    photo: UploadFile = File(...),
    description: str = Form(""),
    model_number: str = Form(""),
    brand: str = Form(""),
    answer: str = Form(""),
    asked: str = Form(""),
):
    ip = client_ip(request)

    data = photo.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return error(400, f"That photo is too large (max {MAX_UPLOAD_MB} MB).")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return error(400, "That file doesn't look like a photo. Try a JPG, PNG or iPhone photo.")

    description = description.strip()[:500]
    model_number = model_number.strip()[:60]
    brand = brand.strip()[:40]
    answer = answer.strip()[:300]

    ok, message = try_consume(ip)
    if not ok:
        return error(429, message)

    try:
        image_url = mendly.image_to_data_url(io.BytesIO(data))
        info = mendly.identify(image_url, f"{description} {answer}".strip())

        if model_number:
            info["model_number"] = model_number
        if brand:
            info["brand"] = brand

        # Ask once for a model number (parts only fit specific models).
        if not info.get("model_number") and not asked:
            refund(ip)  # nothing expensive has run yet
            question = info.get("clarifying_question") if info.get("confidence") == "low" else None
            return {
                "status": "needs_info",
                "item": info.get("item"),
                "brand": info.get("brand"),
                "question": question,
                "remaining": remaining(ip),
            }

        plan = mendly.plan_repair(info)
        if not info.get("brand") and plan.get("brand"):
            info["brand"] = plan["brand"]

        base = " ".join(
            str(x) for x in (info.get("brand"), info.get("model_number"),
                             info.get("item"), info.get("suspected_part")) if x
        )
        queries = plan.get("search_queries") or []
        if isinstance(queries, str):
            queries = [queries]
        queries = queries or [f"{base} replacement part buy"]
        parts = mendly.find_parts(queries, info.get("suspected_part") or "the broken part", info.get("model_number"), info.get("brand"))

        return {
            "status": "ok",
            "info": info,
            "plan": plan,
            "parts": parts,
            "remaining": remaining(ip),
        }
    except Exception as e:  # keep the error friendly, log the real one
        print("analyze failed:", repr(e))
        refund(ip)
        return error(500, "Something went wrong analyzing that photo. Please try again.")


# ---- Front end ---------------------------------------------------------------
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
