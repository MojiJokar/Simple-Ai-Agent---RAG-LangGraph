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
    raise SystemExit("NVIDIA_API_KEY  in file .env  was not found .")


def _chain(primary_key, fallback_key):
    primary = os.getenv(primary_key, "")
    extras = os.getenv(fallback_key, "")
    models = [primary] + [m.strip() for m in extras.split(",") if m.strip()]
    return [m for m in models if m]


GEN_CHAIN = _chain("GEN_MODEL", "GEN_FALLBACKS")
FAST_CHAIN = _chain("FAST_MODEL", "FAST_FALLBACKS")

CHROMA_DIR = "./chroma_db"
COLLECTION = "hozhi_scripts"
# The fast model should fail quickly so we can move to the fallback model sooner.

# The larger model gets more time because it generates longer responses.
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
            max_retries=0,   # we manage ourselves
        )
    return _cache[key]

# here agents talk with our model !
# those 3 models is  given as  a   clean list so, we understand better
def call(prompt, fast=False, temperature=0.0, log=None):
    """
    A robust model call.

    First, try the primary model, with three exponential backoff retries for 429 errors.

    If that doesn't work, move to the next model in the chain.

     none of the models work, return None — don't crash.
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
                    raise ValueError("Empty Answer")
                if log is not None:
                    log.append(f"ل Model: {model_id}")
                return out
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "Too Many Requests" in msg) and attempt < 2:
                    time.sleep(wait + random.uniform(0, 1.5))
                    wait *= 2
                    continue
                if log is not None:
                    log.append(f"{model_id} was not  ({type(e).__name__}) — the next one")
                break

    if log is not None:
        log.append("if any model answered, it would have returned already. None returned.")
    return None


def embedder(): #convert notion to number 
    """Local embeddings. Free, with no rate limit and no expiration date."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )