import os
import re

from memory.state import State
from utils.helpers import read_prompt, llm
from langchain_core.messages import AIMessage

# ── Regex patterns (compiled once at module load) ────────────────────────────

PROMPT_INJECTION = re.compile(
    r"ignore (all |previous |prior |your |the ){0,3}(instructions?|rules?|guidelines?|prompt|context)"
    r"|disregard (all |your |the ){0,3}(instructions?|rules?|guidelines?|prompt)"
    r"|you are now\b"
    r"|forget (everything|all|your instructions?)"
    r"|override (your |the )?(instructions?|rules?|guidelines?|system)"
    r"|new (instructions?|prompt|rules?|guidelines?|system prompt)",
    re.IGNORECASE,
)

JAILBREAK = re.compile(
    r"\bDAN\b"
    r"|do anything now"
    r"|pretend (you have no|there are no) (restrictions?|rules?|guidelines?|limits?)"
    r"|act as (an? )?(evil|unrestricted|unfiltered|uncensored|jailbroken)"
    r"|roleplay as (an? )?(evil|malicious|unrestricted|unfiltered)"
    r"|you have no (restrictions?|rules?|guidelines?|limits?)"
    r"|developer mode"
    r"|jailbreak",
    re.IGNORECASE,
)

PII_EMAIL       = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PII_PHONE       = re.compile(r"(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}")
PII_CREDIT_CARD = re.compile(r"\b(?:\d[ \-]?){13,16}\b")
PII_SSN         = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")

# ── Refusal messages per block reason ────────────────────────────────────────

REFUSAL_MESSAGES = {
    "prompt_injection": (
        "I'm not able to process that request. It looks like it may contain "
        "an attempt to override my instructions. Please ask your IT support question directly."
    ),
    "jailbreak": (
        "I'm here to help with IT support questions. I can't engage with that type of request."
    ),
    "pii": (
        "Your message appears to contain sensitive personal information (such as a credit card "
        "number or SSN). Please remove it and rephrase your IT support question."
    ),
    "off_topic_abuse": (
        "I'm an IT support assistant and can only help with technology-related questions. "
        "I'm not able to assist with that request."
    ),
}

# ── LLM fallback prompt (Layer 4) ────────────────────────────────────────────

_GUARDRAILS_SYSTEM_PROMPT = read_prompt(
    os.environ.get("GUARDRAILS_PROMPT_DIR", "prompts/guardrails_llm_prompt.txt")
)


def _check_pii(query: str):
    """Return True if the query contains high-confidence PII."""
    if PII_SSN.search(query):
        return True
    if PII_CREDIT_CARD.search(query):
        return True
    # Email/phone only blocked if multiple instances appear (reduces false positives)
    if len(PII_EMAIL.findall(query)) > 1:
        return True
    if len(PII_PHONE.findall(query)) > 1:
        return True
    return False


def guardrails_node(state: State) -> dict:
    """
    Guardrails Node — first node in the LangGraph pipeline.

    Performs four-layer safety checks in order of speed:
      1. Prompt injection (regex)
      2. Jailbreak patterns (regex)
      3. PII detection (regex)
      4. Off-topic abuse (LLM classifier — last resort only)

    Returns:
        guardrail_triggered=True + refusal message if blocked.
        guardrail_triggered=False (no message change) if safe.
    """
    print("-> GUARDRAILS ->")

    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict):
        query = last_msg.get("content", "")
    else:
        query = getattr(last_msg, "content", str(last_msg))

    print(f"[guardrails] Checking query: {query[:80]}...")

    # Layer 1: Prompt injection
    if PROMPT_INJECTION.search(query):
        print("[guardrails] BLOCKED — prompt_injection")
        return _blocked("prompt_injection")

    # Layer 2: Jailbreak
    if JAILBREAK.search(query):
        print("[guardrails] BLOCKED — jailbreak")
        return _blocked("jailbreak")

    # Layer 3: PII
    if _check_pii(query):
        print("[guardrails] BLOCKED — pii")
        return _blocked("pii")

    # Layer 4: LLM abuse classifier (only if all regex layers pass)
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=_GUARDRAILS_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        verdict = response.content.strip().upper()
        print(f"[guardrails] LLM verdict: {verdict}")
        if verdict == "UNSAFE":
            return _blocked("off_topic_abuse")
    except Exception as e:
        # LLM failure → pass through to avoid blocking legitimate queries
        print(f"[guardrails] LLM check failed ({e}), passing through.")

    print("[guardrails] PASSED")
    return {
        "guardrail_triggered": False,
        "guardrail_reason": None,
    }


def _blocked(reason: str) -> dict:
    return {
        "guardrail_triggered": True,
        "guardrail_reason": reason,
        "messages": [AIMessage(content=REFUSAL_MESSAGES[reason])],
    }


def route_guardrails(state: State) -> str:
    if state.get("guardrail_triggered", False):
        return "blocked"
    return "pass"
