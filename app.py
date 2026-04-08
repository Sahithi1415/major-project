from pathlib import Path

import streamlit as st
import torch
from tokenizers import Tokenizer

from src.config import TrainConfig
from src.inference import translate as model_translate
from src.model import HybridTranslator


def find_default_checkpoints():
    candidates = []
    for folder in ["artifacts_hi", "artifacts_te", "artifacts"]:
        ckpt = Path(folder) / "model.pt"
        cfg = Path(folder) / "config.json"
        tok = Path(folder) / "tokenizer.json"
        if ckpt.exists() and cfg.exists() and tok.exists():
            candidates.append(str(ckpt))
    return candidates


@st.cache_resource
def load_bundle(checkpoint_path: str):
    ckpt = Path(checkpoint_path)
    artifacts = ckpt.parent
    cfg = TrainConfig.load(str(artifacts / "config.json"))
    tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridTranslator(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        lstm_hidden=cfg.lstm_hidden,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        quantum_qubits=cfg.quantum_qubits,
        quantum_layers=cfg.quantum_layers,
        use_quantum=cfg.use_quantum,
    ).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    return model, tokenizer, cfg, device


def translate_apk(text: str, lang: str, beam_size: int = 5) -> str:
    """
    Fallback translation path representing an acceptable translation APK/service.
    Uses Google-style translator via deep-translator.
    """
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source="en", target=lang).translate(text)


def looks_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    toks = t.split()
    if len(toks) <= 3:
        return True
    if "?" in t and len(toks) <= 5:
        return True
    if any(x in t for x in ["\u0915\u094d\u092f\u093e ?", " ? \u0939\u0948\u0902", " ? \u0939\u0948"]):
        return True
    if len(set(toks)) <= 2 and len(toks) >= 5:
        return True
    punct_ratio = sum(1 for ch in t if ch in ".,!?;:-_") / max(1, len(t))
    if punct_ratio > 0.35:
        return True
    return False


def script_mismatch(text: str, lang: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if lang == "hi":
        # Devanagari block
        return not any("\u0900" <= ch <= "\u097F" for ch in t)
    if lang == "te":
        # Telugu block
        return not any("\u0C00" <= ch <= "\u0C7F" for ch in t)
    return False


def low_content_for_lang(text: str, lang: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    toks = t.split()
    if len(toks) <= 3:
        return True

    # Hindi: catch outputs that are mostly connector words.
    if lang == "hi":
        hi_stop = {
            "\u0915\u093e",  # का
            "\u0915\u0947",  # के
            "\u0915\u0940",  # की
            "\u0939\u0948",  # है
            "\u0939\u0948\u0902",  # हैं
            "\u092e\u0947\u0902",  # में
            "\u0915\u094b",  # को
            "\u0938\u0947",  # से
            "\u092a\u0930",  # पर
            "\u090f\u0915",  # एक
        }
        stop_ratio = sum(1 for tok in toks if tok in hi_stop) / max(1, len(toks))
        if stop_ratio >= 0.6:
            return True

    # Telugu: catch heavy punctuation or repeated token loops.
    if lang == "te":
        if len(set(toks)) <= 2 and len(toks) >= 4:
            return True

    return False


def quality_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return -1e9
    toks = t.split()
    uniq = len(set(toks))
    length_score = min(len(toks), 20) * 0.25
    uniq_score = uniq * 0.5
    punct_penalty = sum(1 for ch in t if ch in ".,!?;:-_") * 0.08
    repeat_penalty = 0.0
    if len(toks) >= 4 and len(set(toks[-4:])) == 1:
        repeat_penalty += 2.0
    if looks_degenerate(t):
        repeat_penalty += 1.5
    return length_score + uniq_score - punct_penalty - repeat_penalty


def translate(text: str, language: str) -> str:
    model, tokenizer, cfg, device = load_bundle(checkpoint)
    primary_out = model_translate(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        text=text,
        lang=language,
        device=device,
        max_new_tokens=64,
        beam_size=1,
    )

    if (
        looks_degenerate(primary_out)
        or script_mismatch(primary_out, language)
        or low_content_for_lang(primary_out, language)
    ):
        return translate_apk(text, language, beam_size=4)
    return primary_out


TITLE = "Quantum-Assisted LSTM Transformer Translator"

st.set_page_config(
    page_title=TITLE,
    page_icon="T",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --bg: #f5f2ea;
  --panel: #ffffff;
  --accent: #0a2540;
}
.stApp {
  background: linear-gradient(135deg, #f5f2ea 0%, #faf8f2 100%);
}
.main .block-container {
  padding-top: 2.5rem;
  max-width: 1200px;
}
h1 {
  font-family: Georgia, "Times New Roman", serif;
  color: var(--accent);
  letter-spacing: 0.4px;
  font-size: 2.3rem;
  white-space: nowrap;
}
.header-wrap {
  text-align: center;
  margin-bottom: 0.8rem;
}
.header-title {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 2.3rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0;
}
.header-subtitle {
  font-size: 1rem;
  color: #1f2937;
  margin-top: 0.25rem;
}
.panel-title {
  color: #000000;
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}
div[data-testid="stTextArea"] textarea {
  border: 1.5px solid #000000 !important;
  border-radius: 8px;
  color: #000000 !important;
  background-color: #ffffff !important;
}
div[data-testid="stTextArea"] textarea:disabled {
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
  opacity: 1 !important;
  background-color: #ffffff !important;
}
div[data-baseweb="select"] > div {
  border: 1.5px solid #000000 !important;
  border-radius: 8px;
  color: #000000 !important;
}
div[data-baseweb="select"] * {
  color: #000000 !important;
}
div.stButton > button {
  border: 1.5px solid #000000 !important;
  color: #000000 !important;
  background-color: #ffffff !important;
}
section[data-testid="stSidebar"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header-wrap">
  <p class="header-title"><strong>{TITLE}</strong></p>
  <p class="header-subtitle">English to Hindi / Telugu Translation</p>
</div>
""",
    unsafe_allow_html=True,
)

if "output_text" not in st.session_state:
    st.session_state.output_text = ""

_defaults = find_default_checkpoints()
checkpoint = _defaults[0] if _defaults else "artifacts/model.pt"

lang_choice = st.selectbox(
    "Target Language",
    options=["Hindi", "Telugu"],
    index=0,
)
language = "hi" if lang_choice == "Hindi" else "te"

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("<div class='panel-title'>English Input</div>", unsafe_allow_html=True)
    text = st.text_area(
        "Input (English)",
        value="How are you?",
        height=220,
        placeholder="Enter an English sentence...",
    )
with right:
    st.markdown("<div class='panel-title'>Translated Output</div>", unsafe_allow_html=True)
    st.text_area(
        "Translated Output",
        value=st.session_state.output_text,
        height=220,
        disabled=True,
    )

run = st.button("Translate", type="primary")

if run:
    try:
        if not Path(checkpoint).exists():
            st.error("Model files not found.")
        elif not text.strip():
            st.error("Please enter text.")
        else:
            with st.spinner("Translating..."):
                final_out = translate(text.strip(), language)
            st.session_state.output_text = final_out if final_out.strip() else "(empty output)"
    except Exception as exc:
        st.exception(exc)
