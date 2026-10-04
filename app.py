import os
import re
import json
import base64
from html import escape

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# ------------------------------------------------------------
# Hugging Face credentials
# Supports:
# 1. Local .env
# 2. Streamlit Cloud Secrets
# ------------------------------------------------------------

try:
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    HF_PROVIDER = st.secrets.get(
        "HF_PROVIDER",
        os.getenv("HF_PROVIDER", "featherless-ai")
    )
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")


MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_IMAGES = 6


st.set_page_config(
    page_title="StyleSwap — AI Personal Stylist",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "generated_looks" not in st.session_state:
    st.session_state.generated_looks = []

if "saved_looks" not in st.session_state:
    st.session_state.saved_looks = []

if "last_wardrobe_analysis" not in st.session_state:
    st.session_state.last_wardrobe_analysis = []

if "last_preferences" not in st.session_state:
    st.session_state.last_preferences = {}


# ============================================================
# HELPERS
# ============================================================

def html(content):
    """Render custom HTML using Streamlit native HTML."""
    st.html(content.strip())


def clean_ai_response(text):
    """Remove Qwen thinking blocks and markdown code fences."""
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def get_client():
    """Create Hugging Face client."""

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN was not found. "
            "Add HF_TOKEN to your local .env file or Streamlit Cloud Secrets."
        )

    return InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN
    )


def gallery(files):
    """Display uploaded clothing images."""

    cards = ""

    for file in files:
        image_base64 = base64.b64encode(
            file.getvalue()
        ).decode("utf-8")

        mime = file.type or "image/jpeg"

        cards += (
            f'<figure>'
            f'<img src="data:{mime};base64,{image_base64}" '
            f'alt="{escape(file.name)}">'
            f'<figcaption>{escape(file.name)}</figcaption>'
            f'</figure>'
        )

    html(
        f"""
        <div class="gallery">
            {cards}
        </div>
        """
    )


# ============================================================
# AI — CLOTHING IMAGE ANALYSIS
# ============================================================

def analyze_clothing_image(client, uploaded_file):

    mime_type = uploaded_file.type or "image/jpeg"

    image_base64 = base64.b64encode(
        uploaded_file.getvalue()
    ).decode("utf-8")

    image_url = (
        f"data:{mime_type};base64,{image_base64}"
    )

    prompt = """
You are the visual wardrobe analyst for an AI personal stylist
called StyleSwap.

Analyze ONLY what is visibly present in this clothing image.

Identify:

ITEM:
COLOR:
PATTERN:
STYLE:
WEATHER:
SUITABLE OCCASIONS:
DESCRIPTION:

Rules:

- Do not invent a brand.
- Do not invent material if it cannot be visually identified.
- If something is uncertain, say "Not clearly visible".
- Keep the answer concise.
- Focus specifically on clothing and fashion characteristics.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        max_tokens=350,
    )

    return clean_ai_response(
        response.choices[0].message.content
    )


# ============================================================
# AI — GENERATE 3 OUTFITS
# ============================================================

def generate_outfits(
    client,
    wardrobe_analysis,
    occasion,
    style,
    weather
):
    """
    Generate exactly three structured outfit recommendations.
    """

    wardrobe_text = ""

    for index, item in enumerate(
        wardrobe_analysis,
        start=1
    ):
        wardrobe_text += f"""
WARDROBE ITEM {index}

Filename:
{item["filename"]}

Visual Analysis:
{item["analysis"]}

-------------------------
"""

    prompt = f"""
You are StyleSwap, an expert AI personal stylist.

Create EXACTLY THREE distinct outfit recommendations
using the user's existing wardrobe.

USER PREFERENCES

Occasion: {occasion}
Preferred Style: {style}
Weather: {weather}

WARDROBE ANALYSIS

{wardrobe_text}

IMPORTANT RULES

1. Use the user's uploaded wardrobe items.
2. Do NOT invent clothing items that were not detected.
3. Each look must be realistic and wearable.
4. Consider occasion, preferred style and weather.
5. Try to make the three looks visibly different.
6. Look 1 should be the closest match to the user's requested style.
7. Look 2 should be a slightly more elevated/polished variation.
8. Look 3 should be a more relaxed variation.
9. If something important is missing, mention it only under
   "optional_addition".
10. Do not claim that a missing item exists in the wardrobe.
11. Keep descriptions concise.
12. Do not discuss your reasoning process.

RETURN ONLY VALID JSON.

Use exactly this structure:

{{
    "looks": [
        {{
            "title": "Short stylish title",
            "vibe": "Signature",
            "outfit": [
                "Wardrobe item 1 and how to wear it",
                "Wardrobe item 2 and how to wear it"
            ],
            "why_it_works": "Short explanation.",
            "styling_tip": "One or two practical styling tips.",
            "optional_addition": "Optional item or None"
        }},
        {{
            "title": "Short stylish title",
            "vibe": "Elevated",
            "outfit": [
                "Wardrobe item 1 and how to wear it",
                "Wardrobe item 2 and how to wear it"
            ],
            "why_it_works": "Short explanation.",
            "styling_tip": "One or two practical styling tips.",
            "optional_addition": "Optional item or None"
        }},
        {{
            "title": "Short stylish title",
            "vibe": "Relaxed",
            "outfit": [
                "Wardrobe item 1 and how to wear it",
                "Wardrobe item 2 and how to wear it"
            ],
            "why_it_works": "Short explanation.",
            "styling_tip": "One or two practical styling tips.",
            "optional_addition": "Optional item or None"
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1400,
    )

    raw_response = response.choices[0].message.content

    cleaned = clean_ai_response(raw_response)

    # --------------------------------------------------------
    # Extract JSON if Qwen adds extra text
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL
    )

    if not match:
        raise ValueError(
            "The AI returned an invalid outfit response."
        )

    json_text = match.group(0)

    data = json.loads(json_text)

    looks = data.get("looks", [])

    if len(looks) < 3:
        raise ValueError(
            "The AI did not return three outfit recommendations."
        )

    return looks[:3]


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
    '&family=Playfair+Display:ital,wght@500;600;700&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F7F2EB;
    color: #241F1B;
    overflow-x: hidden;
}

.block-container {
    width: 100% !important;
    max-width: none !important;
    padding:
        clamp(12px, 2.5vw, 24px)
        clamp(16px, 4vw, 64px)
        50px !important;
}


/* ============================================================
   NAVBAR
   ============================================================ */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;

    padding: 10px 4px 20px;

    border-bottom: 1px solid #DED2C5;

    margin-bottom:
        clamp(18px, 3vw, 30px);
}

.brand {
    font-size: clamp(20px, 2.6vw, 24px);
    font-weight: 700;
    letter-spacing: -1px;
}

.brand span {
    color: #B96B4D;
}

.nav-links {
    display: flex;
    gap: clamp(14px, 2.5vw, 28px);

    font-size: clamp(11px, 1.1vw, 13px);

    color: #756B62;

    letter-spacing: 1px;
}

.ai-pill {
    background: #27211D;
    color: white;

    border-radius: 30px;

    padding:
        8px 16px;

    font-size: 11px;

    letter-spacing: 1px;

    white-space: nowrap;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    display: grid;

    grid-template-columns:
        minmax(0, 1.15fr)
        minmax(0, 1fr);

    align-items: center;

    gap: clamp(12px, 3vw, 40px);

    padding: clamp(22px, 5vw, 60px);

    margin-bottom:
        clamp(28px, 5vw, 55px);

    background: #FFFDF9;

    border: 1px solid #DED2C5;

    border-radius:
        clamp(18px, 3vw, 28px);

    overflow: hidden;
}

.kicker {
    font-size: 10px;

    letter-spacing: 3px;

    color: #B96B4D;

    font-weight: 700;

    margin-bottom: 15px;
}

.hero-title {
    font-family: 'Playfair Display', serif;

    font-size:
        clamp(38px, 5.6vw, 82px);

    line-height: 1;

    letter-spacing: -0.035em;

    margin-bottom: 25px;
}

.hero-title em {
    color: #B96B4D;
}

.hero-copy {
    max-width: 550px;

    color: #756B62;

    line-height: 1.7;

    font-size:
        clamp(12.5px, 1.2vw, 15px);
}

.hero-tag {
    grid-column: 1 / -1;

    font-size: 9px;

    letter-spacing: 3px;

    color: #756B62;
}


/* ============================================================
   FASHION ANIMATION
   ============================================================ */

.fashion-stage {
    position: relative;

    justify-self: center;

    width: min(100%, 440px);

    aspect-ratio: 1 / 1;

    container-type: inline-size;
}

.ring {
    position: absolute;

    left: 50%;
    top: 50%;

    border: 1px solid #DDB59B;

    border-radius: 50%;

    transform:
        translate(-50%, -50%);
}

.ring.one {
    width: 78cqw;
    height: 78cqw;

    animation:
        rotateRing 14s linear infinite;
}

.ring.two {
    width: 58cqw;
    height: 58cqw;

    animation:
        rotateRingReverse 10s linear infinite;
}

.ring.three {
    width: 38cqw;
    height: 38cqw;

    animation:
        pulseRing 4s ease-in-out infinite;
}

.dress {
    position: absolute;

    left: 50%;
    top: 50%;

    width: 34cqw;
    height: 59cqw;

    transform:
        translate(-50%, -50%);

    animation:
        floatDress 4s ease-in-out infinite;
}

.dress-neck {
    position: absolute;

    width: 32%;
    height: 16%;

    left: 34%;
    top: 2%;

    background: #B96B4D;

    border-radius:
        15px 15px 8px 8px;
}

.dress-body {
    position: absolute;

    width: 73%;
    height: 69%;

    left: 13%;
    top: 11.5%;

    background: #B96B4D;

    clip-path:
        polygon(
            28% 0,
            72% 0,
            100% 100%,
            0 100%
        );

    border-radius:
        10px 10px 35px 35px;
}

.dress-belt {
    position: absolute;

    width: 61%;
    height: 3%;

    left: 19%;
    top: 34.5%;

    background: #27211D;

    border-radius: 10px;
}

.hanger {
    position: absolute;

    left: 50%;
    top: 9%;

    width: 24cqw;
    height: 15cqw;

    transform: translateX(-50%);

    border: 2px solid #27211D;

    border-top: none;

    opacity: .8;

    clip-path:
        polygon(
            0 0,
            50% 100%,
            100% 0,
            95% 0,
            50% 87%,
            5% 0
        );
}

.spark {
    position: absolute;

    color: #B96B4D;

    font-size:
        clamp(16px, 7cqw, 30px);

    animation:
        sparkle 2.5s ease-in-out infinite;
}

.s1 {
    right: 16%;
    top: 20%;
}

.s2 {
    left: 10%;
    top: 35%;

    animation-delay: .7s;
}

.s3 {
    right: 8%;
    bottom: 25%;

    animation-delay: 1.2s;
}

.s4 {
    left: 16%;
    bottom: 20%;

    animation-delay: 1.8s;
}

@keyframes floatDress {

    0%, 100% {
        transform:
            translate(-50%, -50%)
            translateY(0);
    }

    50% {
        transform:
            translate(-50%, -50%)
            translateY(-5cqw);
    }
}

@keyframes rotateRing {

    from {
        transform:
            translate(-50%, -50%)
            rotate(0);
    }

    to {
        transform:
            translate(-50%, -50%)
            rotate(360deg);
    }
}

@keyframes rotateRingReverse {

    from {
        transform:
            translate(-50%, -50%)
            rotate(360deg);
    }

    to {
        transform:
            translate(-50%, -50%)
            rotate(0);
    }
}

@keyframes pulseRing {

    0%, 100% {
        transform:
            translate(-50%, -50%)
            scale(1);

        opacity: .5;
    }

    50% {
        transform:
            translate(-50%, -50%)
            scale(1.12);

        opacity: 1;
    }
}

@keyframes sparkle {

    0%, 100% {
        transform: scale(.8);
        opacity: .4;
    }

    50% {
        transform: scale(1.25);
        opacity: 1;
    }
}


/* ============================================================
   SECTIONS
   ============================================================ */

.section-number {
    color: #B96B4D;

    font-size: 10px;

    letter-spacing: 3px;

    font-weight: 700;

    margin-bottom: 8px;
}

.section-title {
    font-family: 'Playfair Display', serif;

    font-size:
        clamp(28px, 4vw, 42px);

    line-height: 1.08;

    margin-bottom: 8px;
}

.section-description {
    color: #756B62;

    margin-bottom: 25px;

    font-size:
        clamp(12.5px, 1.2vw, 15px);

    line-height: 1.55;
}


/* ============================================================
   UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background: #FFFDF9;

    border: 1px dashed #CDBEAF;

    border-radius:
        clamp(14px, 2vw, 20px);

    padding:
        clamp(10px, 2vw, 18px);
}

[data-testid="stFileUploader"]:hover {
    border-color: #B96B4D;
}


/* ============================================================
   GALLERY
   ============================================================ */

.gallery {
    display: grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(
                clamp(105px, 22vw, 190px),
                1fr
            )
        );

    gap:
        clamp(8px, 1.6vw, 14px);

    margin: 10px 0;
}

.gallery figure {
    margin: 0;

    background: #FFFDF9;

    border: 1px solid #DED2C5;

    border-radius: 14px;

    padding: 8px;
}

.gallery img {
    width: 100%;

    aspect-ratio: 3 / 4;

    object-fit: cover;

    border-radius: 10px;

    display: block;
}

.gallery figcaption {
    margin-top: 6px;

    font-size: 11px;

    color: #756B62;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    background: #27211D !important;

    color: #fff !important;

    border: none !important;

    border-radius: 12px !important;

    min-height: 48px;

    padding:
        12px 20px !important;

    font-size: 13px !important;

    font-weight: 700 !important;

    letter-spacing: 1px !important;

    transition:
        all .3s ease !important;
}

.stButton > button:hover {
    background: #B96B4D !important;

    transform:
        translateY(-2px);
}


/* ============================================================
   RESULT HEADER
   ============================================================ */

.result {
    background: #27211D;

    color: white;

    border-radius:
        clamp(18px, 2.5vw, 25px);

    padding:
        clamp(20px, 4vw, 38px);

    margin:
        clamp(28px, 4vw, 45px)
        0
        25px;
}

.result-title {
    font-family: 'Playfair Display', serif;

    font-size:
        clamp(30px, 4.4vw, 44px);

    line-height: 1.1;

    margin-bottom: 12px;
}

.result-copy {
    color: #D7CEC5;

    line-height: 1.7;

    font-size:
        clamp(12px, 1.2vw, 15px);
}


/* ============================================================
   OUTFIT CARDS
   ============================================================ */

.looks-grid {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 20px;

    margin:
        20px 0 30px;
}

.look-card {
    background: #FFFDF9;

    border: 1px solid #DED2C5;

    border-radius: 20px;

    padding: 24px;

    min-width: 0;

    height: 100%;

    box-sizing: border-box;

    transition:
        transform .25s ease,
        box-shadow .25s ease;
}

.look-card:hover {
    transform:
        translateY(-5px);

    box-shadow:
        0 14px 35px rgba(
            64,
            45,
            34,
            .10
        );
}

.look-number {
    color: #B96B4D;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 2px;

    margin-bottom: 10px;
}

.look-vibe {
    display: inline-block;

    background: #F0E3D8;

    color: #8E4D35;

    padding:
        5px 9px;

    border-radius: 20px;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;

    margin-bottom: 14px;
}

.look-title {
    font-family: 'Playfair Display', serif;

    font-size:
        clamp(23px, 2.2vw, 31px);

    line-height: 1.1;

    margin-bottom: 20px;
}

.look-label {
    color: #B96B4D;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.5px;

    margin-top: 18px;

    margin-bottom: 7px;
}

.look-text {
    color: #5F564F;

    font-size: 13px;

    line-height: 1.65;
}

.outfit-list {
    margin: 0;

    padding-left: 18px;

    color: #3F3934;

    font-size: 13px;

    line-height: 1.7;
}

.outfit-list li {
    margin-bottom: 6px;
}

.optional {
    color: #756B62;

    font-style: italic;

    font-size: 12px;
}


/* ============================================================
   SAVED LOOKS
   ============================================================ */

.saved-look {
    background: #FFFDF9;

    border: 1px solid #DED2C5;

    border-radius: 16px;

    padding: 16px;

    margin-bottom: 10px;
}

.saved-look-title {
    font-family: 'Playfair Display', serif;

    font-size: 20px;

    margin-bottom: 5px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top:
        clamp(40px, 7vw, 80px);

    padding-top: 20px;

    border-top: 1px solid #DED2C5;

    display: flex;

    justify-content: space-between;

    gap: 10px;

    flex-wrap: wrap;

    color: #756B62;

    font-size: 11px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1000px) {

    .looks-grid {
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }

    .hero {
        grid-template-columns: 1fr;
    }

    .fashion-stage {
        width: min(100%, 380px);
    }
}


@media (max-width: 768px) {

    .nav-links {
        display: none;
    }

    .looks-grid {
        grid-template-columns: 1fr;
    }

    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 1rem !important;
    }

    div[data-testid="stHorizontalBlock"]
    > div[data-testid="stColumn"] {
        width: 100% !important;

        flex:
            1 1 100% !important;

        min-width:
            100% !important;
    }

    .stButton > button {
        width: 100% !important;
    }

    .look-card {
        padding: 20px;
    }
}


@media (max-width: 359px) {

    .ai-pill {
        display: none;
    }

}

</style>
"""
)


# ============================================================
# NAVBAR
# ============================================================

html(
    """
    <div class="navbar">

        <div class="brand">
            Style<span>Swap</span>
        </div>

        <div class="nav-links">
            <div>WARDROBE</div>
            <div>STYLE ME</div>
            <div>ABOUT</div>
        </div>

        <div class="ai-pill">
            ✦ AI STYLIST
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

html(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="kicker">
                ✦ PERSONAL AI STYLIST
            </div>

            <div class="hero-title">
                Style yourself.
                <br>
                <em>Without buying new.</em>
            </div>

            <div class="hero-copy">
                Turn the clothes already sitting in your wardrobe
                into outfits you'll actually want to wear.
                StyleSwap builds looks around your occasion,
                personal style and weather.
            </div>

        </div>


        <div class="fashion-stage">

            <div class="ring one"></div>
            <div class="ring two"></div>
            <div class="ring three"></div>

            <div class="dress">
                <div class="dress-neck"></div>
                <div class="dress-body"></div>
                <div class="dress-belt"></div>
            </div>

            <div class="hanger"></div>

            <div class="spark s1">✦</div>
            <div class="spark s2">✧</div>
            <div class="spark s3">✦</div>
            <div class="spark s4">·</div>

        </div>


        <div class="hero-tag">
            WARDROBE → AI → OUTFIT
        </div>

    </div>
    """
)


# ============================================================
# SECTION 01 — WARDROBE
# ============================================================

html(
    """
    <div class="section-number">
        01 — YOUR WARDROBE
    </div>

    <div class="section-title">
        Start with what you own.
    </div>

    <div class="section-description">
        Upload photos of the clothes you want to style.
        StyleSwap's vision model will identify and analyze them.
    </div>
    """
)


uploaded_files = st.file_uploader(
    "Upload your clothing photos",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True,
    help="Upload up to 6 clothing images.",
)


if uploaded_files:

    if len(uploaded_files) > MAX_IMAGES:

        st.warning(
            f"Please use up to {MAX_IMAGES} images."
        )

        uploaded_files = uploaded_files[:MAX_IMAGES]

    st.markdown("### Your wardrobe")

    gallery(uploaded_files)


# ============================================================
# SECTION 02 — STYLE
# ============================================================

html(
    """
    <div style="height:clamp(20px,4vw,35px)"></div>

    <div class="section-number">
        02 — YOUR STYLE
    </div>

    <div class="section-title">
        Tell us the vibe.
    </div>

    <div class="section-description">
        Choose where you're going, how you want to look,
        and what the weather is doing.
    </div>
    """
)


col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.markdown("### Occasion")

        occasion = st.selectbox(
            "Where are you going?",
            [
                "College",
                "Casual Day",
                "Interview",
                "Party",
                "Date",
                "Travel",
                "Dinner"
            ],
            label_visibility="collapsed",
        )


with col2:

    with st.container(border=True):

        st.markdown("### Style")

        style = st.selectbox(
            "How do you want to look?",
            [
                "Minimal",
                "Casual",
                "Smart Casual",
                "Formal",
                "Streetwear",
                "Elegant"
            ],
            label_visibility="collapsed",
        )


with col3:

    with st.container(border=True):

        st.markdown("### Weather")

        weather = st.selectbox(
            "What's the weather?",
            [
                "Hot",
                "Warm",
                "Cool",
                "Cold",
                "Rainy"
            ],
            label_visibility="collapsed",
        )


# ============================================================
# CREATE BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    create_look = st.button(
        "✦  CREATE MY LOOK",
        use_container_width=True
    )


# ============================================================
# AI GENERATION
# ============================================================

if create_look:

    if not uploaded_files:

        st.warning(
            "Please upload at least one clothing item first."
        )

    else:

        try:

            client = get_client()

            # ------------------------------------------------
            # AI HEADER
            # ------------------------------------------------

            html(
                """
                <div class="result">

                    <div class="kicker">
                        ✦ STYLESWAP AI
                    </div>

                    <div class="result-title">
                        Reading your wardrobe...
                    </div>

                    <div class="result-copy">
                        Qwen3-VL is analyzing your clothing
                        and understanding the pieces available
                        for your outfits.
                    </div>

                </div>
                """
            )

            # ------------------------------------------------
            # ANALYZE WARDROBE
            # ------------------------------------------------

            wardrobe_analysis = []

            progress = st.progress(0)

            for index, file in enumerate(
                uploaded_files
            ):

                with st.spinner(
                    f"Analyzing {file.name}..."
                ):

                    analysis = analyze_clothing_image(
                        client,
                        file
                    )

                    wardrobe_analysis.append(
                        {
                            "filename": file.name,
                            "analysis": analysis
                        }
                    )

                progress.progress(
                    (index + 1) /
                    len(uploaded_files)
                )

            progress.empty()

            # Save analysis
            st.session_state.last_wardrobe_analysis = (
                wardrobe_analysis
            )

            st.session_state.last_preferences = {
                "occasion": occasion,
                "style": style,
                "weather": weather
            }

            # ------------------------------------------------
            # DISPLAY AI ANALYSIS
            # ------------------------------------------------

            st.markdown(
                "### ✦ AI wardrobe analysis"
            )

            for item in wardrobe_analysis:

                with st.expander(
                    f"AI detected · {item['filename']}"
                ):

                    st.write(
                        item["analysis"]
                    )

            # ------------------------------------------------
            # GENERATE 3 OUTFITS
            # ------------------------------------------------

            with st.spinner(
                "Creating 3 personalized looks..."
            ):

                looks = generate_outfits(
                    client,
                    wardrobe_analysis,
                    occasion,
                    style,
                    weather
                )

            st.session_state.generated_looks = looks

            # ------------------------------------------------
            # RESULT HEADER
            # ------------------------------------------------

            html(
                """
                <div class="result">

                    <div class="kicker">
                        ✦ STYLESWAP AI
                    </div>

                    <div class="result-title">
                        Your looks are ready.
                    </div>

                    <div class="result-copy">
                        Three outfit directions have been
                        created from your wardrobe and matched
                        with your occasion, preferred style
                        and weather.
                    </div>

                </div>
                """
            )

            # ------------------------------------------------
            # PREFERENCES
            # ------------------------------------------------

            st.markdown("### Your styling preferences")

            pref1, pref2, pref3 = st.columns(3)

            with pref1:
                st.markdown(
                    f"**Occasion**  \n{occasion}"
                )

            with pref2:
                st.markdown(
                    f"**Style**  \n{style}"
                )

            with pref3:
                st.markdown(
                    f"**Weather**  \n{weather}"
                )

            # ------------------------------------------------
            # 3 LOOK CARDS
            # ------------------------------------------------

            st.markdown(
                "### ✦ Choose your look"
            )

            cards_html = ""

            for index, look in enumerate(
                looks[:3],
                start=1
            ):

                title = escape(
                    str(
                        look.get(
                            "title",
                            f"Look {index}"
                        )
                    )
                )

                vibe = escape(
                    str(
                        look.get(
                            "vibe",
                            "Style"
                        )
                    )
                )

                why = escape(
                    str(
                        look.get(
                            "why_it_works",
                            ""
                        )
                    )
                )

                styling_tip = escape(
                    str(
                        look.get(
                            "styling_tip",
                            ""
                        )
                    )
                )

                optional = escape(
                    str(
                        look.get(
                            "optional_addition",
                            "None"
                        )
                    )
                )

                outfit_items = look.get(
                    "outfit",
                    []
                )

                outfit_html = ""

                for item in outfit_items:

                    outfit_html += (
                        f"<li>{escape(str(item))}</li>"
                    )

                cards_html += f"""
                <div class="look-card">

                    <div class="look-number">
                        LOOK {index:02d}
                    </div>

                    <div class="look-vibe">
                        {vibe}
                    </div>

                    <div class="look-title">
                        {title}
                    </div>

                    <div class="look-label">
                        OUTFIT
                    </div>

                    <ul class="outfit-list">
                        {outfit_html}
                    </ul>

                    <div class="look-label">
                        WHY IT WORKS
                    </div>

                    <div class="look-text">
                        {why}
                    </div>

                    <div class="look-label">
                        STYLING TIP
                    </div>

                    <div class="look-text">
                        {styling_tip}
                    </div>

                    <div class="look-label">
                        OPTIONAL ADDITION
                    </div>

                    <div class="optional">
                        {optional}
                    </div>

                </div>
                """

            html(
                f"""
                <div class="looks-grid">
                    {cards_html}
                </div>
                """
            )

            # ------------------------------------------------
            # SAVE BUTTONS
            # ------------------------------------------------

            save1, save2, save3 = st.columns(3)

            for index, column in enumerate(
                [save1, save2, save3]
            ):

                with column:

                    if st.button(
                        f"♡ SAVE LOOK {index + 1}",
                        key=f"save_look_{index}",
                        use_container_width=True
                    ):

                        selected_look = looks[index]

                        # Avoid exact duplicates
                        if selected_look not in (
                            st.session_state.saved_looks
                        ):

                            st.session_state.saved_looks.append(
                                selected_look
                            )

                        st.success(
                            f"Look {index + 1} saved."
                        )

            # ------------------------------------------------
            # REGENERATE
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            regenerate_col1, regenerate_col2, regenerate_col3 = (
                st.columns([1, 2, 1])
            )

            with regenerate_col2:

                regenerate = st.button(
                    "↻  REGENERATE 3 LOOKS",
                    use_container_width=True,
                    key="regenerate_current"
                )

            if regenerate:

                with st.spinner(
                    "Creating fresh outfit ideas..."
                ):

                    new_looks = generate_outfits(
                        client,
                        wardrobe_analysis,
                        occasion,
                        style,
                        weather
                    )

                st.session_state.generated_looks = (
                    new_looks
                )

                st.rerun()

        except Exception as error:

            st.error(
                "Something went wrong while creating your looks."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error),
                    language="text"
                )

            st.info(
                "Check that your HF_TOKEN is valid and that "
                f"{MODEL} is available through the selected "
                f"Hugging Face provider ({HF_PROVIDER})."
            )


# ============================================================
# SAVED LOOKS
# ============================================================

if st.session_state.saved_looks:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "### ♡ Saved looks"
    )

    for index, look in enumerate(
        st.session_state.saved_looks,
        start=1
    ):

        title = escape(
            str(
                look.get(
                    "title",
                    f"Saved Look {index}"
                )
            )
        )

        vibe = escape(
            str(
                look.get(
                    "vibe",
                    ""
                )
            )
        )

        outfit_items = look.get(
            "outfit",
            []
        )

        outfit_text = " · ".join(
            str(item)
            for item in outfit_items
        )

        html(
            f"""
            <div class="saved-look">

                <div class="saved-look-title">
                    {title}
                </div>

                <div class="look-text">
                    {vibe} · {escape(outfit_text)}
                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="footer">

        <div>
            StyleSwap © 2026
        </div>

        <div>
            AI PERSONAL STYLING ·
            QWEN3-VL ·
            HUGGING FACE
        </div>

    </div>
    """
)