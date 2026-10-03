"""
Voice to Text (Speech Recognition) App
Transcribes audio files or live microphone recordings using OpenAI's
Whisper model via faster-whisper. Supports 90+ languages, auto language
detection, translation to English, and subtitle (.srt) export.
Runs locally. No API key needed.
"""

import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Voice to Text",
    page_icon="🎙️",
    layout="centered"
)


# ---------------- Custom CSS ----------------
st.markdown("""
<style>

/* ===== Main App ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #12002f 0%,
        #1a0b3d 45%,
        #0f172a 100%
    );
    color: #ffffff;
}

/* Main container */
.main .block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ===== Title ===== */
h1 {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #ffffff !important;
}


/* ===== Normal Text ===== */
.stApp p {
    color: #ddd6fe;
}


/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #17052f 0%,
        #0b1026 100%
    );
    border-right: 1px solid #6d28d9;
}

section[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] label {
    color: #ddd6fe !important;
    font-weight: 600;
}


/* ===== Select Boxes ===== */
div[data-baseweb="select"] > div {
    background-color: #241044;
    border: 1px solid #7c3aed;
    border-radius: 10px;
    color: #ffffff;
}


/* ===== Radio Buttons ===== */
div[role="radiogroup"] label {
    color: #ddd6fe !important;
}


/* ===== Tabs ===== */
button[data-baseweb="tab"] {
    color: #a78bfa !important;
    font-weight: 600;
    font-size: 1rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c084fc !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #a855f7 !important;
}


/* ===== File Uploader ===== */
section[data-testid="stFileUploaderDropzone"] {
    background: #1e1038;
    border: 2px dashed #7c3aed;
    border-radius: 16px;
    padding: 20px;
    transition: 0.3s ease;
}

section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #c084fc;
    background: #28134a;
}


/* ===== Buttons ===== */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #6d28d9;
    background: #241044;
    color: #ffffff;
    font-weight: 600;
    padding: 0.65rem 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #c084fc;
    color: #e9d5ff;
    transform: translateY(-1px);
}


/* ===== Primary Button ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #c026d3
    );
    border: none;
    color: #ffffff;
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(
        135deg,
        #6d28d9,
        #a21caf
    );
    color: #ffffff;
}


/* ===== Download Buttons ===== */
.stDownloadButton > button {
    width: 100%;
    border-radius: 10px;
    background: #241044;
    color: #ffffff;
    border: 1px solid #6d28d9;
    font-weight: 600;
    transition: 0.2s ease;
}

.stDownloadButton > button:hover {
    border-color: #c084fc;
    color: #e9d5ff;
}


/* ===== Text Area ===== */
textarea {
    background-color: #160b2e !important;
    color: #ffffff !important;
    border: 1px solid #7c3aed !important;
    border-radius: 12px !important;
}

textarea:focus {
    border-color: #c084fc !important;
}


/* ===== Audio Player ===== */
audio {
    width: 100%;
    border-radius: 10px;
    margin: 10px 0;
}


/* ===== Alert Boxes ===== */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #6d28d9;
}

div[data-testid="stAlert"] p {
    color: #ddd6fe !important;
}


/* ===== Expander ===== */
details {
    background: #1e1038;
    border: 1px solid #6d28d9;
    border-radius: 12px;
}

details summary {
    color: #ffffff !important;
    font-weight: 600;
}


/* ===== Horizontal Line ===== */
hr {
    border-color: #6d28d9;
}


/* ===== Mobile Responsive ===== */
@media (max-width: 768px) {

    h1 {
        font-size: 2.2rem !important;
    }

    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


LANGUAGES = {
    "Auto detect": None,
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Urdu": "ur",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
}


# ---------------- Session state ----------------
# Changing widget_key gives the uploader/recorder a new identity -> they reset
if "widget_key" not in st.session_state:
    st.session_state.widget_key = 0

if "result" not in st.session_state:
    st.session_state.result = None


def clear_audio():
    """Remove the current upload/recording and the old transcript."""
    st.session_state.widget_key += 1
    st.session_state.result = None


@st.cache_resource(show_spinner="Loading Whisper model (first run downloads it)...")
def load_model(size: str):
    # int8 on CPU = fast and low memory; change device to "cuda" if you have a GPU
    return WhisperModel(
        size,
        device="cpu",
        compute_type="int8"
    )


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(segments) -> str:
    lines = []

    for i, seg in enumerate(segments, 1):
        lines.append(
            f"{i}\n"
            f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n"
            f"{seg['text']}\n"
        )

    return "\n".join(lines)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

size = st.sidebar.selectbox(
    "Model size",
    ["tiny", "base", "small", "medium"],
    index=1,
    help="Bigger = more accurate but slower. 'base' is a good start; 'small' for Indian languages.",
)

lang_name = st.sidebar.selectbox(
    "Language",
    list(LANGUAGES.keys())
)

task = st.sidebar.radio(
    "Task",
    ["Transcribe", "Translate to English"]
)


# ---------------- Main ----------------
st.title("🎙️ Voice to Text")

st.write(
    "Upload an audio file or record your voice to convert speech into text."
)


key = st.session_state.widget_key

tab_upload, tab_record = st.tabs(
    ["📁 Upload audio", "🎤 Record"]
)

audio = None


with tab_upload:

    up = st.file_uploader(
        "Audio file",
        type=[
            "mp3",
            "wav",
            "m4a",
            "ogg",
            "flac",
            "webm",
            "mp4"
        ],
        key=f"upload_{key}",
    )

    if up:
        audio = up


with tab_record:

    rec = st.audio_input(
        "Click to record",
        key=f"record_{key}"
    )

    if rec:
        audio = rec


if audio is not None:

    st.audio(audio)

    # Unique id for this audio, so an old transcript isn't shown for a new file
    audio_id = f"{getattr(audio, 'name', 'rec')}_{audio.size}"

    col1, col2 = st.columns([1, 1])

    convert = col1.button(
        "📝 Convert to text",
        type="primary",
        use_container_width=True
    )

    col2.button(
        "🗑️ Clear / New audio",
        on_click=clear_audio,
        use_container_width=True
    )


    if convert:

        suffix = (
            os.path.splitext(
                getattr(audio, "name", "") or ""
            )[1]
            or ".wav"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(audio.getvalue())
            path = tmp.name


        try:

            model = load_model(size)

            with st.spinner(
                "Transcribing... (longer audio takes longer)"
            ):

                seg_iter, info = model.transcribe(
                    path,
                    language=LANGUAGES[lang_name],
                    task=(
                        "translate"
                        if task.startswith("Translate")
                        else "transcribe"
                    ),
                    beam_size=5,
                    vad_filter=True,  # skips silence
                )

                segments = [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text.strip()
                    }
                    for s in seg_iter
                ]

        finally:
            os.remove(path)


        # Save result so it survives reruns (e.g. clicking download)
        st.session_state.result = {
            "audio_id": audio_id,
            "text": " ".join(
                s["text"] for s in segments
            ),
            "segments": segments,
            "language": info.language,
            "prob": info.language_probability,
            "duration": info.duration,
        }


    # ---------------- Show result ----------------
    res = st.session_state.result

    if res and res["audio_id"] == audio_id:

        st.success(
            f"Detected language: **{res['language']}** "
            f"({res['prob'] * 100:.0f}% confidence) • "
            f"Duration: {res['duration']:.1f}s"
        )

        st.subheader("Transcript")

        st.text_area(
            "Text",
            res["text"],
            height=200
        )


        with st.expander("Timestamps"):

            for s in res["segments"]:

                st.write(
                    f"`[{s['start']:6.1f}s → "
                    f"{s['end']:6.1f}s]` {s['text']}"
                )


        c1, c2 = st.columns(2)

        c1.download_button(
            "⬇️ Download .txt",
            res["text"],
            "transcript.txt",
            "text/plain"
        )

        c2.download_button(
            "⬇️ Download .srt (subtitles)",
            to_srt(res["segments"]),
            "transcript.srt",
            "text/plain"
        )


else:

    st.info(
        "👆 Upload a file or record audio to begin."
    )