"""  
Integration and robust model call layer — Heji agent

There are two important decisions here:

1. No model names are hardcoded into the code. They all come from .env.

Nvidia retires models quickly; this way, changing a model
means changing a line in .env, not touching the code.

2. Model calls never kill the program. It first waits and tries again, then goes to the alternate model, and if none of them work

it returns None so that the agent can make a safe decision.


"""



import os
import time
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

if not API_KEY:
    raise SystemExit("NVIDIA_API_KEY در فایل .env پیدا نشد.")


def _chain(primary_key, fallback_key):
    primary = os.getenv(primary_key, "")
    extras = os.getenv(fallback_key, "")
    models = [primary] + [m.strip() for m in extras.split(",") if m.strip()]
    return [m for m in models if m]


GEN_CHAIN = _chain("GEN_MODEL", "GEN_FALLBACKS")
FAST_CHAIN = _chain("FAST_MODEL", "FAST_FALLBACKS")

CHROMA_DIR = "./chroma_db"
COLLECTION = "hozhi_scripts"

# مدل سریع باید سریع شکست بخورد تا زود برویم سراغ جایگزین.
# مدل بزرگ فرصت بیشتری می‌گیرد چون جواب طولانی می‌سازد.
FAST_TIMEOUT = 30
GEN_TIMEOUT = 90

_cache = {}


def _llm(model_id, temperature, timeout):
    key = (model_id, temperature, timeout)
    if key not in _cache:
        _cache[key] = ChatOpenAI(
            model=model_id,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=temperature,
            timeout=timeout,
            max_retries=0,   # خودمان مدیریت می‌کنیم
        )
    return _cache[key]


def call(prompt, fast=False, temperature=0.0, log=None):
    """
    یک فراخوانی مقاوم.
    اول مدل اصلی، با سه بار عقب‌نشینی نمایی روی ۴۲۹.
    اگر نشد، مدل بعدی زنجیره.
    اگر هیچ‌کدام نشد، None — نه کرش.
    """
    chain = FAST_CHAIN if fast else GEN_CHAIN
    timeout = FAST_TIMEOUT if fast else GEN_TIMEOUT

    for model_id in chain:
        wait = 4
        for attempt in range(3):
            try:
                out = _llm(model_id, temperature, timeout).invoke(prompt).content
                out = (out or "").strip()
                if not out:
                    raise ValueError("جواب خالی")
                if log is not None:
                    log.append(f"مدل: {model_id}")
                return out
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "Too Many Requests" in msg) and attempt < 2:
                    time.sleep(wait + random.uniform(0, 1.5))
                    wait *= 2
                    continue
                if log is not None:
                    log.append(f"{model_id} نشد ({type(e).__name__}) — بعدی")
                break

    if log is not None:
        log.append("هیچ مدلی جواب نداد — مسیر امن")
    return None


def embedder():
    """امبدینگ لوکال. رایگان، بدون ریت‌لیمیت، بدون تاریخ انقضا."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )