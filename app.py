import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go
import base64
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Golf Swing Validation App",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   METRIC SELECTION PAGE
   ============================================================ */

.metric-selection-card {
    background: #FFFFFF;
    border: 1px solid #DDE6DA;
    border-radius: 26px;
    padding: 28px 24px;
    height: 300px;
    box-shadow: 0 10px 28px rgba(15, 40, 30, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
}

.metric-selection-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(15, 40, 30, 0.14);
}

.metric-icon {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    background: #EAF4EC;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 16px;
}

.metric-selection-title {
    font-size: 24px;
    font-weight: 900;
    color: #0B3D2E;
    margin-bottom: 10px;
}

.metric-selection-text {
    font-size: 15px;
    color: #4B5563;
    line-height: 1.55;
    flex-grow: 1;
}

.metric-status-ready {
    display: inline-block;
    margin-top: auto;
    background: #EAF4EC;
    color: #0B3D2E;
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 13px;
    font-weight: 800;
    width: fit-content;
}

.metric-status-coming {
    display: inline-block;
    margin-top: auto;
    background: #F1F5F9;
    color: #64748B;
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 13px;
    font-weight: 800;
    width: fit-content;
}
/* ============================================================
   INTEGRATED INTERPRETATION PROFESSIONAL DASHBOARD
   ============================================================ */

.interpretation-dashboard {
    background: #FFFFFF;
    border: 1px solid #DDE6DA;
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 12px 30px rgba(15, 40, 30, 0.08);
    margin-top: 14px;
}

.interpretation-header-stable {
    background: linear-gradient(135deg, #EAF8EF 0%, #F7FFF9 100%);
    border-left: 8px solid #16A34A;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 22px;
}

.interpretation-header-warning {
    background: linear-gradient(135deg, #FFF7E6 0%, #FFFCF5 100%);
    border-left: 8px solid #E5A100;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 22px;
}

.interpretation-header-critical {
    background: linear-gradient(135deg, #FFF1F1 0%, #FFFAFA 100%);
    border-left: 8px solid #DC2626;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 22px;
}

.interpretation-title {
    font-size: 30px;
    font-weight: 900;
    color: #0B3D2E;
    margin-bottom: 8px;
}

.interpretation-subtitle {
    font-size: 16px;
    color: #4B5563;
    line-height: 1.55;
}

.insight-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 20px;
    min-height: 185px;
    margin-bottom: 16px;
}

.insight-card-title {
    font-size: 15px;
    font-weight: 850;
    color: #0B3D2E;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.insight-card-value {
    font-size: 28px;
    font-weight: 900;
    color: #07184A;
    margin-bottom: 8px;
}

.insight-card-text {
    font-size: 15px;
    color: #4B5563;
    line-height: 1.5;
}

.coach-advice-box {
    background: #F7FAFF;
    border-left: 7px solid #0B57D0;
    border-radius: 20px;
    padding: 22px;
    margin-top: 14px;
}

.coach-advice-title {
    font-size: 20px;
    font-weight: 900;
    color: #0B57D0;
    margin-bottom: 8px;
}

.coach-advice-text {
    font-size: 16px;
    color: #102A43;
    line-height: 1.55;
}

.stApp {
    background: linear-gradient(180deg, #F7FAF5 0%, #FFFFFF 45%);
    color: #07184A;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}
.hero {
    background: linear-gradient(135deg, #062B21 0%, #0B3D2E 55%, #DDEAD7 100%);
    padding: 44px 50px;
    border-radius: 30px;
    color: white;
    box-shadow: 0 20px 50px rgba(0,0,0,0.18);
    margin-bottom: 30px;
}
.hero-title {
    font-size: 48px;
    font-weight: 850;
    line-height: 1.05;
    margin-bottom: 12px;
    color: white;
}
.hero-subtitle {
    font-size: 19px;
    color: #EAF4EC;
    max-width: 980px;
    line-height: 1.45;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 8px 16px;
    margin-bottom: 18px;
    font-size: 13px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: white;
    font-weight: 700;
}
.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #0B3D2E;
    margin-top: 10px;
    margin-bottom: 14px;
}
.section-subtitle {
    font-size: 16px;
    color: #5B6470;
    margin-bottom: 20px;
}
.card {
    background: #FFFFFF;
    border: 1px solid #E1E8DF;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(15, 40, 30, 0.08);
    height: 210px;
    display: flex;
    flex-direction: column;
}
.card-title {
    font-size: 21px;
    font-weight: 800;
    color: #0B3D2E;
    margin-bottom: 8px;
}
.card-text {
    font-size: 15px;
    color: #4B5563;
    line-height: 1.55;
    flex-grow: 1;
}
.big-number {
    font-size: 38px;
    font-weight: 850;
    color: #0B57D0;
    margin-top: 10px;
}
.login-card {
    background: #FFFFFF !important;
    border: 1px solid #DDE6DA !important;
    border-radius: 28px !important;
    padding: 30px !important;
    box-shadow: 0 14px 35px rgba(15, 40, 30, 0.10) !important;
}
.profile-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FAF5 100%);
    border: 1px solid #DDE6DA;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 10px 28px rgba(15, 40, 30, 0.08);
    min-height: 250px;
}
.profile-icon {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: #EAF4EC;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    margin-bottom: 14px;
}
.metric-box {
    background: #FFFFFF;
    border-radius: 22px;
    border: 1px solid #DFE7DE;
    padding: 22px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    min-height: 165px;
}
.metric-name {
    font-size: 14px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.metric-main {
    font-size: 34px;
    font-weight: 850;
    color: #0B3D2E;
    margin-top: 4px;
}
.metric-desc {
    font-size: 14px;
    color: #4B5563;
    margin-top: 4px;
}
.status-normal {
    margin-top: 10px;
    font-size: 18px;
    font-weight: 800;
    color: #16A34A;
}
.status-altered {
    margin-top: 10px;
    font-size: 18px;
    font-weight: 800;
    color: #DC2626;
}
.output-stable {
    background: #F1FFF5;
    border: 2px solid #2E8B57;
    border-radius: 24px;
    padding: 26px;
    color: #0B3D2E;
}
.output-warning {
    background: #FFF8EA;
    border: 2px solid #E5A100;
    border-radius: 24px;
    padding: 26px;
    color: #5C3B00;
}
.output-critical {
    background: #FFF1F1;
    border: 2px solid #D43C3C;
    border-radius: 24px;
    padding: 26px;
    color: #6B1111;
}
.formula-box {
    background: #F7FAFF;
    border-left: 6px solid #0B57D0;
    border-radius: 16px;
    padding: 18px 22px;
    margin: 12px 0;
    color: #102A43;
}
.upload-note {
    background: #FFFFFF;
    border: 1px dashed #9CA3AF;
    border-radius: 20px;
    padding: 18px;
    color: #4B5563;
}
.upload-card {
    background: linear-gradient(135deg, #0B3D2E 0%, #145A3F 100%);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 10px 28px rgba(15, 40, 30, 0.12);
    min-height: 210px;
    margin-bottom: 14px;
}
.upload-card .upload-title {
    color: #FFFFFF !important;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    font-weight: 850;
    margin-bottom: 16px;
}
.upload-card .upload-icon {
    color: #FFFFFF !important;
    font-size: 34px;
    margin-bottom: 14px;
}
.upload-card .upload-description {
    color: #EAF4EC !important;
    font-size: 15px;
    line-height: 1.45;
}
.upload-card .upload-description b {
    color: #FFFFFF !important;
}
.selected-athlete-box {
    background: #F7FAFF;
    border-left: 6px solid #0B57D0;
    border-radius: 16px;
    padding: 16px 18px;
    margin: 14px 0 22px 0;
    color: #07184A;
}
.athlete-card {
    background: #FFFFFF;
    border: 1px solid #DDE6DA;
    border-radius: 24px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(15, 40, 30, 0.08);
    min-height: 320px;
}
.athlete-photo-wrap {
    width: 185px;
    height: 185px;
    border-radius: 50%;
    overflow: hidden;
    margin: 0 auto 18px auto;
    background: #EAF4EC;
    border: 4px solid #EAF4EC;
    display: flex;
    align-items: center;
    justify-content: center;
}

.athlete-photo {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 35%;
    display: block;
    transform: scale(1.32);
}
.athlete-name {
    font-size: 22px;
    font-weight: 850;
    color: #0B3D2E;
    margin-bottom: 8px;
}
.athlete-info {
    font-size: 14px;
    color: #4B5563;
    line-height: 1.45;
}
.athlete-status {
    display: inline-block;
    margin-top: 12px;
    background: #EAF4EC;
    color: #0B3D2E;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 750;
}
.stButton > button {
    background: linear-gradient(135deg, #0B3D2E 0%, #146B4B 100%);
    color: #FFFFFF !important;
    border: none;
    border-radius: 16px;
    padding: 0.75rem 1rem;
    font-weight: 800;
    box-shadow: 0 8px 20px rgba(15, 40, 30, 0.18);
}
/* ============================================================
   DOWNLOAD BUTTON VISIBILITY FIX
   ============================================================ */

div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0B57D0 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.75rem 1.1rem !important;
    font-weight: 850 !important;
    box-shadow: 0 8px 20px rgba(11, 87, 208, 0.22) !important;
}

div[data-testid="stDownloadButton"] > button * {
    color: #FFFFFF !important;
    font-weight: 850 !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #0847A6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}
section[data-testid="stSidebar"] {
    background-color: #0B3D2E;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ============================================================
   FILE UPLOADER VISIBILITY FIX
   ============================================================ */

div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px solid #DDE6DA !important;
    border-radius: 16px !important;
    padding: 14px !important;
}

div[data-testid="stFileUploader"] section {
    background: #F7FAFF !important;
    border: 1.5px dashed #0B57D0 !important;
    border-radius: 16px !important;
}

div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0B57D0 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 850 !important;
    padding: 0.65rem 1rem !important;
}

div[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
    font-weight: 850 !important;
}

div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    color: #07184A !important;
    font-weight: 650 !important;
}

div[data-testid="stFileUploader"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
/* ============================================================
   SIDEBAR NAVIGATION WHITE TEXT FIX
   ============================================================ */

section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}
[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #07184A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 14px !important;
}

/* ============================================================
   LOGIN PAGE TEXT FIX
   ============================================================ */

/* Label principali del login */
div[data-testid="stRadio"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stCheckbox"] label {
    color: #07184A !important;
    font-weight: 700 !important;
}

/* Testo delle opzioni Athlete / Coach */
div[data-testid="stRadio"] p,
div[data-testid="stRadio"] span {
    color: #07184A !important;
    font-weight: 600 !important;
}

/* Testo checkbox */
div[data-testid="stCheckbox"] p,
div[data-testid="stCheckbox"] span {
    color: #07184A !important;
    font-weight: 600 !important;
}

/* Placeholder negli input */
[data-testid="stTextInput"] input::placeholder {
    color: #6B7280 !important;
    opacity: 1 !important;
}

/* Testo scritto negli input */
[data-testid="stTextInput"] input {
    color: #07184A !important;
    background-color: #FFFFFF !important;
}

/* Testo generale dentro il login */
.login-card,
.login-card * {
    color: #07184A !important;
}

/* Mantiene bianchi solo eventuali bottoni */
.stButton > button,
.stButton > button * {
    color: #FFFFFF !important;
}
/* ============================================================
   EXPANDER VISIBILITY FIX
   ============================================================ */

div[data-testid="stExpander"] details summary {
    background-color: #0B3D2E !important;
    color: #FFFFFF !important;
    font-weight: 850 !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 12px 16px !important;
}

div[data-testid="stExpander"] details summary p {
    color: #FFFFFF !important;
    font-weight: 850 !important;
    font-size: 16px !important;
}

div[data-testid="stExpander"] details summary svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #DDE6DA !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-top: 14px !important;
}
</style>
""", unsafe_allow_html=True)



# ============================================================
# LOGIN SYSTEM
# ============================================================

def init_user_state():
    defaults = {
        "logged_in": False,
        "user_role": None,
        "user_name": "",
        "user_club": "",
        "data_consent": False,
        "selected_athlete": None,
        "selected_athlete_name": "",
        "selected_metric": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_name = ""
    st.session_state.user_club = ""
    st.session_state.data_consent = False
    st.session_state.selected_athlete = None
    st.session_state.selected_athlete_name = ""
    st.rerun()


def render_login_page():
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Professional access</div>
        <div class="hero-title">Golf Swing Validation App</div>
        <div class="hero-subtitle">
            A session-based platform for professional golfers and coaches, designed to connect
            swing rhythm, body–club coordination and physiological state.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Create your access profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 0.95], gap="large")

    with col1:
        role = st.radio(
            "I am using the app as:",
            ["Athlete", "Coach"],
            horizontal=True
        )

        name = st.text_input(
            "Name and surname",
            placeholder="Example: Rachele Costantini"
        )

        club = st.text_input(
            "Golf club / training centre",
            placeholder="Example: Milano Golf Club"
        )

        if role == "Athlete":
            consent = st.checkbox(
                "I consent to share my session data with coaches from my same golf club."
            )
        else:
            consent = True

    with col2:
        if role == "Athlete":
            st.markdown("""
            <div class="profile-card">
                <div class="profile-icon">🏌️‍♀️</div>
                <div class="card-title">Athlete profile</div>
                <div class="card-text">
                    Analyse your own swing sessions and monitor rhythm, repeatability,
                    body–club coordination and physiological state after training.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="profile-card">
                <div class="profile-icon">🎯</div>
                <div class="card-title">Coach profile</div>
                <div class="card-text">
                    Review session dashboards for athletes from your same golf club
                    and support training decisions using movement and physiological data.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("Enter the app", use_container_width=True):
        if name.strip() == "":
            st.error("Please enter your name.")
            return

        if club.strip() == "":
            st.error("Please enter your golf club / training centre.")
            return

        st.session_state.logged_in = True
        st.session_state.user_role = role
        st.session_state.user_name = name.strip()
        st.session_state.user_club = club.strip()
        st.session_state.data_consent = consent
        st.rerun()


def require_login():
    init_user_state()

    if not st.session_state["logged_in"]:
        render_login_page()
        st.stop()

def image_to_base64(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        return None

    with open(image_path, "rb") as f:
        data = f.read()

    suffix = image_path.suffix.lower().replace(".", "")

    if suffix == "jpg":
        suffix = "jpeg"

    return f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"

# ============================================================
# DEMO ATHLETE DATABASE
# ============================================================

ATHLETES_DB = [
    {
        "id": "anita",
        "name": "Anita",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "25.05.2026",
        "status": "Data sharing enabled",
        "photo": "assets/anita.jpeg"
    },
    {
        "id": "alessandro",
        "name": "Alessandro",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "15.04.2025",
        "status": "Data sharing enabled",
        "photo": "assets/alessandro.jpeg"
    },
    {
        "id": "stefano",
        "name": "Stefano",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "18.05.2026",
        "status": "Data sharing enabled",
        "photo": "assets/stefano.jpeg"
    }
]


def get_athletes_for_club(club_name):
    club_name = str(club_name).strip().lower()
    same_club = [a for a in ATHLETES_DB if a["club"].strip().lower() == club_name]
    return same_club if len(same_club) > 0 else ATHLETES_DB


def select_athlete(athlete):
    st.session_state.selected_athlete = athlete["id"]
    st.session_state.selected_athlete_name = athlete["name"]
    st.rerun()
def select_metric(metric_name):
    st.session_state.selected_metric = metric_name
    st.rerun()
def go_to_page(page_name):
    st.session_state.pending_page = page_name
    st.rerun()
def render_coach_athlete_roster():
    athletes = get_athletes_for_club(st.session_state.user_club)

    st.markdown('<div class="section-title">Athletes from your club</div>', unsafe_allow_html=True)

    cols = st.columns(len(athletes))

    for col, athlete in zip(cols, athletes):
        with col:

            photo_src = image_to_base64(athlete["photo"])

            if photo_src is None:
                photo_src = f"https://api.dicebear.com/8.x/personas/svg?seed={athlete['name']}"

            st.markdown(f"""
            <div class="athlete-card">
                <div class="athlete-photo-wrap">
                    <img src="{photo_src}" class="athlete-photo">
                </div>
                <div class="athlete-name">{athlete['name']}</div>
                <div class="athlete-info">
                    {athlete['level']}<br>
                    <b>Club:</b> {athlete['club']}<br>
                    <b>Last session:</b> {athlete['last_session']}
                </div>
                <div class="athlete-status">{athlete['status']}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Open {athlete['name']}'s data", key=f"open_{athlete['id']}", use_container_width=True):
                select_athlete(athlete)

    if st.session_state.selected_athlete_name:
        st.markdown(f"""
        <div class="selected-athlete-box">
            <b>Selected athlete:</b> {st.session_state.selected_athlete_name}<br>
            You can now open <b>Session Analysis</b> from the sidebar to review this athlete's data.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# DATA FUNCTIONS
# ============================================================

def parse_movella_number(x):
    """
    Converts normal numbers and Excel-damaged Movella numbers.

    Example:
    845.702.838.897.705 -> 8.45702838897705
    -451.042.556.762.695 -> -4.51042556762695
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == "" or s.lower() == "nan":
        return np.nan

    if "#" in s:
        return np.nan

    s = s.replace(",", ".")

    if s.count(".") > 1:
        sign = -1 if s.startswith("-") else 1
        s_clean = s.replace("-", "").replace(".", "")

        try:
            return sign * float(s_clean) / 1e14
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(
                uploaded_file,
                header=None,
                sep=None,
                engine="python",
                on_bad_lines="skip"
            )

        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(
                uploaded_file,
                header=None
            )

        st.error("Unsupported file format. Please upload CSV or Excel.")
        return None

    except Exception as e:
        st.error(f"Error while reading file: {e}")
        return None


def standardize_imu_columns(df, fs=120):
    if df is None:
        return None

    df = df.copy()

    header_row = None

    for i in range(len(df)):
        row_values = [str(v).strip() for v in df.iloc[i].values]
        row_low = [v.lower() for v in row_values]

        has_acc_original = (
            "acc_x" in row_low and
            "acc_y" in row_low and
            "acc_z" in row_low
        )

        has_acc_short = (
            "ax" in row_low and
            "ay" in row_low and
            "az" in row_low
        )

        if has_acc_original or has_acc_short:
            header_row = i
            break

    if header_row is not None:
        header = [str(v).strip() for v in df.iloc[header_row].values]
        data = df.iloc[header_row + 1:].copy()
        data.columns = header

        data = data.rename(columns={
            "Acc_X": "ax",
            "Acc_Y": "ay",
            "Acc_Z": "az",
            "acc_x": "ax",
            "acc_y": "ay",
            "acc_z": "az",
            "Ax": "ax",
            "Ay": "ay",
            "Az": "az"
        })

        required = ["ax", "ay", "az"]

        for col in required:
            if col not in data.columns:
                st.error(f"Missing IMU column {col}. Columns found: {data.columns.tolist()}")
                return None

        imu_df = pd.DataFrame()
        imu_df["ax"] = data["ax"].apply(parse_movella_number)
        imu_df["ay"] = data["ay"].apply(parse_movella_number)
        imu_df["az"] = data["az"].apply(parse_movella_number)

        imu_df = imu_df.dropna(subset=["ax", "ay", "az"]).reset_index(drop=True)

        if len(imu_df) <= 15:
            st.error(
                f"Too few valid IMU samples after cleaning: {len(imu_df)}. "
                "Check the file format or upload the original Movella CSV."
            )
            return None

        imu_df["time_s"] = np.arange(len(imu_df)) / fs

        imu_df["a_tot"] = np.sqrt(
            imu_df["ax"]**2 +
            imu_df["ay"]**2 +
            imu_df["az"]**2
        )

        return imu_df[["time_s", "ax", "ay", "az", "a_tot"]]

    df.columns = [str(c).strip().lower() for c in df.columns]

    possible_time = ["time", "time_s", "timestamp", "pc_time_s", "t", "seconds"]
    time_col = None

    for c in possible_time:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        df["time_s"] = np.arange(len(df)) / fs
    else:
        df["time_s"] = pd.to_numeric(df[time_col], errors="coerce")

    required = ["ax", "ay", "az"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Missing IMU columns: {missing}. Required columns: ax, ay, az.")
        return None

    df["ax"] = df["ax"].apply(parse_movella_number)
    df["ay"] = df["ay"].apply(parse_movella_number)
    df["az"] = df["az"].apply(parse_movella_number)

    df = df.dropna(subset=["time_s", "ax", "ay", "az"])

    df["a_tot"] = np.sqrt(
        df["ax"]**2 +
        df["ay"]**2 +
        df["az"]**2
    )

    return df[["time_s", "ax", "ay", "az", "a_tot"]]


def read_local_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return None

    name = file_path.name.lower()

    try:
        if name.endswith(".csv") or name.endswith(".cvs"):
            return pd.read_csv(
                file_path,
                header=None,
                sep=None,
                engine="python",
                on_bad_lines="skip"
            )

        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(
                file_path,
                header=None
            )

        st.error("Unsupported local file format. Please use CSV or Excel.")
        return None

    except Exception as e:
        st.error(f"Error while reading local file: {e}")
        return None
def standardize_hr_columns(df):
    if df is None:
        return None

    df = df.copy()

    header_row = None
    for i in range(min(len(df), 20)):
        row = [str(v).strip().lower() for v in df.iloc[i].values]
        if any(v in row for v in ["hr", "heart_rate", "bpm", "hr_bpm"]):
            header_row = i
            break

    if header_row is not None:
        header = [str(v).strip().lower() for v in df.iloc[header_row].values]
        data = df.iloc[header_row + 1:].copy()
        data.columns = header
        df = data
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]

    possible_time = ["time", "time_s", "timestamp", "pc_time_s", "t", "seconds"]
    time_col = None

    for c in possible_time:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        df["time_s"] = np.arange(len(df))
    else:
        df["time_s"] = pd.to_numeric(df[time_col], errors="coerce")

    hr_col = None
    for c in ["hr", "heart_rate", "heart rate", "bpm", "hr_bpm", "heart_rate_bpm"]:
        if c in df.columns:
            hr_col = c
            break

    hrv_col = None
    for c in ["hrv", "rmssd", "hrv_rmssd", "rmssd_ms", "hrv_ms"]:
        if c in df.columns:
            hrv_col = c
            break

    if hr_col is None:
        st.warning("HR column not found. Accepted names: HR, hr, heart_rate, bpm, HR_bpm.")
        df["hr"] = np.nan
    else:
        df["hr"] = pd.to_numeric(df[hr_col], errors="coerce")

    if hrv_col is None:
        df["rmssd"] = np.nan
    else:
        df["rmssd"] = pd.to_numeric(df[hrv_col], errors="coerce")

    df = df.dropna(subset=["time_s"])

    return df[["time_s", "hr", "rmssd"]]


# ============================================================
# ECG / HR FUNCTIONS — .BIN FILE, NO SYNCHRONIZATION
# Pipeline: raw ECG -> band-pass 5–20 Hz -> R-peaks -> RR -> HR
# ============================================================

FS_ECG = 250
ECG_CROP = 1375
ECG_NUM_CHANNELS = 4
ECG_DTYPE = ">i4"


def load_ecg_bin_from_upload(uploaded_file):
    """
    Reads a 4-channel chest-band .bin file.
    Channels:
    0 = Acc_X
    1 = Acc_Y
    2 = Acc_Z
    3 = ECG_Raw
    """

    if uploaded_file is None:
        return None

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()

    raw = np.frombuffer(raw_bytes, dtype=ECG_DTYPE)

    n = len(raw) // ECG_NUM_CHANNELS

    if n <= ECG_CROP:
        st.error("ECG file too short after crop.")
        return None

    mat = raw[:n * ECG_NUM_CHANNELS].reshape(-1, ECG_NUM_CHANNELS)

    mat = mat[ECG_CROP:, :]

    bin_df = pd.DataFrame({
        "time_s": np.arange(len(mat)) / FS_ECG,
        "acc_x": mat[:, 0].astype(float),
        "acc_y": mat[:, 1].astype(float),
        "acc_z": mat[:, 2].astype(float),
        "ecg_raw": mat[:, 3].astype(float)
    })

    return bin_df

def load_ecg_bin_from_path(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        st.error(f"ECG .bin file not found: {file_path}")
        return None

    raw_bytes = file_path.read_bytes()

    raw = np.frombuffer(raw_bytes, dtype=ECG_DTYPE)

    n = len(raw) // ECG_NUM_CHANNELS

    if n <= ECG_CROP:
        st.error("ECG file too short after crop.")
        return None

    mat = raw[:n * ECG_NUM_CHANNELS].reshape(-1, ECG_NUM_CHANNELS)
    mat = mat[ECG_CROP:, :]

    bin_df = pd.DataFrame({
        "time_s": np.arange(len(mat)) / FS_ECG,
        "acc_x": mat[:, 0].astype(float),
        "acc_y": mat[:, 1].astype(float),
        "acc_z": mat[:, 2].astype(float),
        "ecg_raw": mat[:, 3].astype(float)
    })

    return bin_df


def process_ecg_bin_path_to_hr(file_path):
    bin_df = load_ecg_bin_from_path(file_path)

    if bin_df is None:
        return None

    ecg_raw = bin_df["ecg_raw"].values

    ecg_filtered = bandpass_filter_ecg(
        ecg_raw,
        fs=FS_ECG,
        lowcut=5,
        highcut=20,
        order=3
    )

    r_peaks, _ = detect_r_peaks_from_filtered_ecg(
        ecg_filtered,
        fs=FS_ECG
    )

    if len(r_peaks) < 3:
        st.error("Too few ECG R-peaks detected.")
        return None

    hr_df = compute_hr_from_r_peaks(
        r_peaks,
        fs=FS_ECG
    )

    if hr_df is None:
        st.error("Too few valid RR intervals after physiological filtering.")
        return None

    bin_swing_peaks, bin_swing_times_s, acc_resultant, acc_filt, activity = detect_bin_acceleration_swings(
        acc_x=bin_df["acc_x"].values,
        acc_y=bin_df["acc_y"].values,
        acc_z=bin_df["acc_z"].values,
        fs=FS_ECG,
        expected_swings=5
    )

    hr_df.attrs["ecg_raw"] = ecg_raw
    hr_df.attrs["ecg_filtered"] = ecg_filtered
    hr_df.attrs["r_peaks"] = r_peaks

    hr_df.attrs["bin_time_s"] = bin_df["time_s"].values
    hr_df.attrs["bin_acc_resultant"] = acc_resultant
    hr_df.attrs["bin_acc_filt"] = acc_filt
    hr_df.attrs["bin_activity"] = activity
    hr_df.attrs["bin_swing_peaks"] = bin_swing_peaks
    hr_df.attrs["bin_swing_times_s"] = bin_swing_times_s

    return hr_df

def bandpass_filter_ecg(signal, fs=250, lowcut=5, highcut=20, order=3):
    """
    Band-pass filter used to emphasize QRS/R-peaks.
    """

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 30:
        return signal

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks_from_filtered_ecg(ecg_filtered, fs=250):
    """
    R-peak detection from filtered ECG.
    This is the same simple/permissive logic that worked better in Colab.
    """

    ecg_filtered = np.asarray(ecg_filtered, dtype=float)

    distance_samples = int(0.35 * fs)

    height_threshold = np.percentile(ecg_filtered, 85)
    prominence_threshold = np.std(ecg_filtered) * 0.8

    r_peaks, properties = find_peaks(
        ecg_filtered,
        distance=distance_samples,
        height=height_threshold,
        prominence=prominence_threshold
    )

    return r_peaks, properties


def compute_hr_from_r_peaks(r_peaks, fs=250):
    """
    Computes beat-by-beat HR from R-peaks.
    Removes non-physiological RR intervals and applies a light median smoothing.
    """

    if len(r_peaks) < 3:
        return None

    rr_ms_all = np.diff(r_peaks) / fs * 1000.0
    time_hr_s_all = r_peaks[1:] / fs

    # Physiological RR filter:
    # 400 ms = 150 bpm
    # 1200 ms = 50 bpm
    valid_mask = (rr_ms_all >= 400) & (rr_ms_all <= 1200)

    rr_ms = rr_ms_all[valid_mask]
    time_hr_s = time_hr_s_all[valid_mask]

    if len(rr_ms) < 3:
        return None

    hr_bpm = 60000.0 / rr_ms

    # Outlier removal using MAD
    hr_median = np.median(hr_bpm)
    hr_mad = np.median(np.abs(hr_bpm - hr_median))

    if hr_mad == 0:
        clean_mask = np.ones(len(hr_bpm), dtype=bool)
    else:
        clean_mask = np.abs(hr_bpm - hr_median) < 3.5 * hr_mad

    hr_bpm_clean = hr_bpm[clean_mask]
    rr_ms_clean = rr_ms[clean_mask]
    time_hr_s_clean = time_hr_s[clean_mask]

    if len(hr_bpm_clean) < 3:
        return None

    # Light smoothing, same as Colab
    hr_series = pd.Series(hr_bpm_clean)

    hr_smooth = hr_series.rolling(
        window=3,
        center=True,
        min_periods=1
    ).median().values

    hr_mean = np.mean(hr_smooth)
    hr_std = np.std(hr_smooth)

    if len(rr_ms_clean) >= 5:
        rmssd = np.sqrt(np.mean(np.diff(rr_ms_clean) ** 2))
    else:
        rmssd = np.nan

    hr_df = pd.DataFrame({
        "time_s": time_hr_s_clean,
        "rr_ms": rr_ms_clean,
        "hr_raw": hr_bpm_clean,
        "hr": hr_smooth,
        "rmssd": rmssd
    })

    hr_df["hr_mean"] = hr_mean
    hr_df["hr_std"] = hr_std
    hr_df["hr_deviation"] = hr_df["hr"] - hr_mean
    hr_df["outside_1sd"] = np.abs(hr_df["hr_deviation"]) > hr_std

    return hr_df

def compute_local_pre_swing_hr(
    hr_df,
    swing_times_s=None,
    pre_window_s=3.0,
    max_window_before_swing_s=1.5,
    high_hr_threshold=120
):
    """
    Computes local HR before each swing using swing times from the HR .bin file.

    HR_pre_swing_mean = mean HR in the 3 seconds before swing
    HR_pre_swing_max_close = max HR only in the last 0.8 seconds before swing

    The advice is triggered only by HR_pre_swing_max_close.
    """

    rows = []

    if hr_df is None:
        return pd.DataFrame()

    if swing_times_s is None:
        swing_times_s = hr_df.attrs.get("bin_swing_times_s", None)

    if swing_times_s is None or len(swing_times_s) == 0:
        return pd.DataFrame()

    for i, t_swing in enumerate(swing_times_s):

        # Full pre-swing window, useful for context
        pre_mask = (
            (hr_df["time_s"] >= t_swing - pre_window_s) &
            (hr_df["time_s"] < t_swing)
        )

        # Close pre-swing window: only immediately before impact
        close_pre_mask = (
            (hr_df["time_s"] >= t_swing - max_window_before_swing_s) &
            (hr_df["time_s"] < t_swing)
        )

        hr_window = hr_df.loc[pre_mask, "hr"].dropna()
        hr_close_window = hr_df.loc[close_pre_mask, "hr"].dropna()

        if len(hr_window) > 0:
            hr_pre_mean = hr_window.mean()
            hr_pre_max = hr_window.max()
        else:
            hr_pre_mean = np.nan
            hr_pre_max = np.nan

        if len(hr_close_window) > 0:
            hr_pre_max_close = hr_close_window.max()
            high_hr_before_swing = hr_pre_max_close >= high_hr_threshold
        else:
            hr_pre_max_close = np.nan
            high_hr_before_swing = False

        rows.append({
            "swing_id": i + 1,
            "t_swing_bin_s": t_swing,
            "HR_pre_swing_mean_3s": hr_pre_mean,
            "HR_pre_swing_max_3s": hr_pre_max,
            "HR_pre_swing_max_close": hr_pre_max_close,
            "High_HR_before_swing": high_hr_before_swing
        })

    return pd.DataFrame(rows)

def process_ecg_bin_to_hr(uploaded_file):
    """
    Converts raw ECG .bin into beat-by-beat HR.
    Swing markers are detected from the .bin acceleration channels,
    so they are on the same time axis as the HR signal.
    """

    bin_df = load_ecg_bin_from_upload(uploaded_file)

    if bin_df is None:
        return None

    ecg_raw = bin_df["ecg_raw"].values

    ecg_filtered = bandpass_filter_ecg(
        ecg_raw,
        fs=FS_ECG,
        lowcut=5,
        highcut=20,
        order=3
    )

    r_peaks, _ = detect_r_peaks_from_filtered_ecg(
        ecg_filtered,
        fs=FS_ECG
    )

    if len(r_peaks) < 3:
        st.error("Too few ECG R-peaks detected.")
        return None

    hr_df = compute_hr_from_r_peaks(
        r_peaks,
        fs=FS_ECG
    )

    if hr_df is None:
        st.error("Too few valid RR intervals after physiological filtering.")
        return None

    bin_swing_peaks, bin_swing_times_s, acc_resultant, acc_filt, activity = detect_bin_acceleration_swings(
        acc_x=bin_df["acc_x"].values,
        acc_y=bin_df["acc_y"].values,
        acc_z=bin_df["acc_z"].values,
        fs=FS_ECG,
        expected_swings=5
    )

    # Store signals for plot/debug/download
    hr_df.attrs["ecg_raw"] = ecg_raw
    hr_df.attrs["ecg_filtered"] = ecg_filtered
    hr_df.attrs["r_peaks"] = r_peaks

    # Store .bin acceleration and .bin swing markers
    hr_df.attrs["bin_time_s"] = bin_df["time_s"].values
    hr_df.attrs["bin_acc_resultant"] = acc_resultant
    hr_df.attrs["bin_acc_filt"] = acc_filt
    hr_df.attrs["bin_activity"] = activity
    hr_df.attrs["bin_swing_peaks"] = bin_swing_peaks
    hr_df.attrs["bin_swing_times_s"] = bin_swing_times_s

    return hr_df

def detect_bin_acceleration_swings(acc_x, acc_y, acc_z, fs=250, expected_swings=5):
    """
    Detects swing events from the chest-band .bin acceleration channels.

    New logic:
    swing = acceleration direction change with pattern:
    positive -> negative -> positive.

    Important:
    The resultant acceleration cannot change sign because it is always positive.
    Therefore, a signed acceleration signal is reconstructed from Acc_X, Acc_Y, Acc_Z
    using PCA projection.
    """

    acc_x = np.asarray(acc_x, dtype=float)
    acc_y = np.asarray(acc_y, dtype=float)
    acc_z = np.asarray(acc_z, dtype=float)

    # Resultant only for debug / export
    acc_resultant = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Filter each acceleration axis
    acc_x_f = lowpass_filter(acc_x, fs=fs, cutoff=12, order=4)
    acc_y_f = lowpass_filter(acc_y, fs=fs, cutoff=12, order=4)
    acc_z_f = lowpass_filter(acc_z, fs=fs, cutoff=12, order=4)

    # Remove offset from each axis
    X = np.vstack([
        acc_x_f - np.median(acc_x_f),
        acc_y_f - np.median(acc_y_f),
        acc_z_f - np.median(acc_z_f)
    ]).T

    # PCA projection to obtain one signed acceleration signal
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        signed_acc = X @ vh[0]
    except Exception:
        signed_acc = acc_x_f - np.median(acc_x_f)

    # Smooth signed acceleration
    acc_filt = lowpass_filter(
        signed_acc,
        fs=fs,
        cutoff=8,
        order=4
    )

    acc_filt = acc_filt - np.median(acc_filt)

    # Activity used only for plotting / debug
    activity = np.abs(acc_filt)

    def find_pos_neg_pos_candidates(signal):
        signal = np.asarray(signal, dtype=float)

        # Deadband to avoid noise-related sign changes
        deadband = max(
            0.10 * np.std(signal),
            0.05 * np.percentile(np.abs(signal), 95)
        )

        sign_signal = np.zeros(len(signal))
        sign_signal[signal > deadband] = 1
        sign_signal[signal < -deadband] = -1

        # Fill zeros with previous valid sign
        for i in range(1, len(sign_signal)):
            if sign_signal[i] == 0:
                sign_signal[i] = sign_signal[i - 1]

        candidates = []

        for i in range(1, len(sign_signal)):

            # positive -> negative
            if sign_signal[i - 1] == 1 and sign_signal[i] == -1:

                start_cross = i
                end_cross = None

                # search next negative -> positive
                for j in range(i + 1, len(sign_signal)):
                    if sign_signal[j - 1] == -1 and sign_signal[j] == 1:
                        end_cross = j
                        break

                if end_cross is None:
                    continue

                duration_s = (end_cross - start_cross) / fs

                # Plausible duration of the acceleration reversal
                if duration_s < 0.12 or duration_s > 2.00:
                    continue

                margin = int(0.35 * fs)

                win_start = max(0, start_cross - margin)
                win_end = min(len(signal), end_cross + margin)

                if win_end <= win_start + 10:
                    continue

                local_signal = signal[win_start:win_end]

                # Representative swing instant:
                # maximum absolute signed acceleration inside the pattern window
                local_peak_relative = np.argmax(np.abs(local_signal))
                peak_index = win_start + local_peak_relative

                positive_before = (
                    np.max(signal[win_start:start_cross])
                    if start_cross > win_start else 0
                )

                negative_middle = (
                    np.min(signal[start_cross:end_cross])
                    if end_cross > start_cross else 0
                )

                positive_after = (
                    np.max(signal[end_cross:win_end])
                    if win_end > end_cross else 0
                )

                # Require a true + / - / + sequence
                if positive_before <= deadband:
                    continue

                if negative_middle >= -deadband:
                    continue

                if positive_after <= deadband:
                    continue

                score = (
                    abs(positive_before)
                    + abs(negative_middle)
                    + abs(positive_after)
                )

                candidates.append({
                    "peak": peak_index,
                    "start_cross": start_cross,
                    "end_cross": end_cross,
                    "duration_s": duration_s,
                    "score": score
                })

        return candidates

    # PCA sign can be arbitrary, so check both directions
    candidates_original = find_pos_neg_pos_candidates(acc_filt)
    candidates_inverted = find_pos_neg_pos_candidates(-acc_filt)

    candidates = candidates_original + candidates_inverted

    if len(candidates) == 0:
        st.warning(
            "No swing detected from positive-negative-positive acceleration reversal. "
            "The app could not identify clear swing events in the .bin acceleration signal."
        )

        peaks = np.array([], dtype=int)
        swing_times_s = np.array([], dtype=float)

        return peaks, swing_times_s, acc_resultant, acc_filt, activity

    # Sort by intensity
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Remove events too close to each other
    selected = []
    min_distance = int(1.0 * fs)

    for cand in candidates:
        p = cand["peak"]

        too_close = False

        for sel in selected:
            if abs(p - sel["peak"]) < min_distance:
                too_close = True
                break

        if not too_close:
            selected.append(cand)

        if len(selected) == expected_swings:
            break

    selected = sorted(selected, key=lambda x: x["peak"])

    peaks = np.array([cand["peak"] for cand in selected], dtype=int)
    swing_times_s = peaks / fs

    if len(peaks) < expected_swings:
        st.warning(
            f"Detected only {len(peaks)} swing(s) from the .bin acceleration signal "
            f"instead of {expected_swings}."
        )

    return peaks, swing_times_s, acc_resultant, acc_filt, activity
# ============================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================

def lowpass_filter(signal, fs=120, cutoff=10, order=4):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 30:
        return signal

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if normal_cutoff >= 1:
        return signal

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return np.nan

    mean_value = np.mean(values)

    if mean_value == 0:
        return np.nan

    return np.std(values, ddof=1) / abs(mean_value) * 100


def find_start_after_flat(signal, peak_idx, left_bound, fs,
                          pre_flat_window_s=2.0,
                          baseline_margin=1.0,
                          confirm_samples=12):
    """
    Start = first point where the signal leaves the flat phase
    within the 2 seconds before the main club acceleration peak.
    """
    search_samples = int(pre_flat_window_s * fs)

    search_start = max(left_bound, peak_idx - search_samples)
    search_end = peak_idx

    segment = signal[search_start:search_end]

    if len(segment) < confirm_samples + 5:
        return search_start, np.nan

    low_values = segment[segment <= np.percentile(segment, 35)]

    if len(low_values) > 0:
        flat_level = np.median(low_values)
    else:
        flat_level = np.median(segment)

    threshold = flat_level + baseline_margin

    start_idx = search_start

    for idx in range(search_start, search_end - confirm_samples):
        window = signal[idx:idx + confirm_samples]

        if np.all(window > threshold):
            start_idx = idx
            break

    return start_idx, threshold


def find_end_after_peak(signal, peak_idx, right_bound, fs,
                        post_peak_window_s=1.0,
                        return_margin=2.0):
    """
    End = first point after the peak where the signal returns close to the flat level.
    """
    search_samples = int(post_peak_window_s * fs)
    search_end = min(right_bound, peak_idx + search_samples)

    if search_end <= peak_idx + 5:
        return right_bound

    post_segment = signal[peak_idx:search_end]

    low_values = post_segment[post_segment <= np.percentile(post_segment, 35)]

    if len(low_values) > 0:
        flat_level = np.median(low_values)
    else:
        flat_level = np.median(post_segment)

    threshold = flat_level + return_margin

    for idx in range(peak_idx, search_end):
        if signal[idx] <= threshold:
            return idx

    return search_end


def analyze_five_swings(
    club_df,
    sacral_df,
    fs=120,
    club_cutoff=10,
    sacral_cutoff=6,
    expected_swings=5,
    min_distance_s=3.0,
    club_prominence=20,
    search_window_s=1.5,
    t_start_analysis=5.0,
    pre_flat_window_s=2.0,
    baseline_margin=1.0,
    post_peak_window_s=1.0,
    return_margin=2.0
):
    club = club_df.copy()
    sacral = sacral_df.copy()

    club["a_filt"] = lowpass_filter(
        club["a_tot"],
        fs=fs,
        cutoff=club_cutoff,
        order=4
    )

    sacral["a_filt"] = lowpass_filter(
        sacral["a_tot"],
        fs=fs,
        cutoff=sacral_cutoff,
        order=4
    )

    club_analysis = club[club["time_s"] >= t_start_analysis].copy().reset_index(drop=True)
    sacral_analysis = sacral[sacral["time_s"] >= t_start_analysis].copy().reset_index(drop=True)

    club_signal = club_analysis["a_filt"].values
    sacral_signal = sacral_analysis["a_filt"].values

    club_time = club_analysis["time_s"].values
    sacral_time = sacral_analysis["time_s"].values

    distance_samples = int(min_distance_s * fs)

    club_peaks, _ = find_peaks(
        club_signal,
        distance=distance_samples,
        prominence=club_prominence
    )

    if len(club_peaks) > expected_swings:
        top_idx = np.argsort(club_signal[club_peaks])[-expected_swings:]
        club_peaks = np.sort(club_peaks[top_idx])

    club_peaks = np.sort(club_peaks)
    n_swings = min(len(club_peaks), expected_swings)

    rows = []
    confirm_samples = int(0.10 * fs)

    for i in range(n_swings):

        peak_idx = club_peaks[i]

        t_peak_club = club_time[peak_idx]
        club_peak_acc = club_signal[peak_idx]

        if i == 0:
            left_bound = 0
        else:
            left_bound = int((club_peaks[i - 1] + peak_idx) / 2)

        if i == n_swings - 1:
            right_bound = len(club_signal) - 1
        else:
            right_bound = int((peak_idx + club_peaks[i + 1]) / 2)

        start_idx, start_threshold = find_start_after_flat(
            signal=club_signal,
            peak_idx=peak_idx,
            left_bound=left_bound,
            fs=fs,
            pre_flat_window_s=pre_flat_window_s,
            baseline_margin=baseline_margin,
            confirm_samples=confirm_samples
        )

        end_idx = find_end_after_peak(
            signal=club_signal,
            peak_idx=peak_idx,
            right_bound=right_bound,
            fs=fs,
            post_peak_window_s=post_peak_window_s,
            return_margin=return_margin
        )

        t_start = club_time[start_idx]
        t_end = club_time[end_idx]

        t_backswing = t_peak_club - t_start
        t_downswing = t_end - t_peak_club

        if t_backswing > 0 and t_downswing > 0:
            tempo_ratio = t_backswing / t_downswing
        else:
            tempo_ratio = np.nan

        sacral_window_mask = (
            (sacral_time >= t_peak_club - search_window_s) &
            (sacral_time <= t_peak_club + search_window_s)
        )

        sacral_window_idx = np.where(sacral_window_mask)[0]

        if len(sacral_window_idx) > 0:
            local_sacral_idx = sacral_window_idx[
                np.argmax(sacral_signal[sacral_window_idx])
            ]

            t_peak_sacral = sacral_time[local_sacral_idx]
            sacral_peak_acc = sacral_signal[local_sacral_idx]
        else:
            t_peak_sacral = np.nan
            sacral_peak_acc = np.nan

        # Colab definition:
        # Δt = t_peak,A5 - t_peak,A4 = sacral - club
        delta_t = t_peak_sacral - t_peak_club

        rows.append({
            "swing_id": i + 1,
            "t_start": t_start,
            "t_peak_club": t_peak_club,
            "t_end": t_end,
            "t_peak_sacral": t_peak_sacral,
            "t_backswing": t_backswing,
            "t_downswing": t_downswing,
            "tempo_ratio": tempo_ratio,
            "delta_t": delta_t,
            "club_peak_acc": club_peak_acc,
            "sacral_peak_acc": sacral_peak_acc,
            "start_threshold": start_threshold
        })

    expected_columns = [
        "swing_id",
        "t_start",
        "t_peak_club",
        "t_end",
        "t_peak_sacral",
        "t_backswing",
        "t_downswing",
        "tempo_ratio",
        "delta_t",
        "club_peak_acc",
        "sacral_peak_acc",
        "start_threshold"
    ]

    swing_analysis_df = pd.DataFrame(rows, columns=expected_columns)

    return swing_analysis_df, club, sacral

# ============================================================
# INTERPRETATION FUNCTIONS
# ============================================================

def get_hr_status(hr_mean, rmssd_mean):
    hr_altered = False
    hrv_altered = False

    if not np.isnan(hr_mean) and hr_mean > 120:
        hr_altered = True

    if not np.isnan(rmssd_mean) and rmssd_mean < 30:
        hrv_altered = True

    if hr_altered or hrv_altered:
        return "Altered", "🔴", "status-altered"

    return "Normal", "🟢", "status-normal"


def classify_interpretation(
    tempo_ratio,
    delta_t,
    hr_mean,
    rmssd_mean,
    pre_swing_high_hr=False,
    pre_swing_hr_mean=np.nan,
    swing_df=None
):
    """
    Integrated interpretation of the session.

    The interpretation combines:
    1. Technical rhythm from Tempo Ratio
    2. Body–club coordination from delta_t
    3. Physiological activation from HR / HRV
    4. Local pre-swing HR activation
    """

    # ============================================================
    # TECHNICAL THRESHOLDS
    # ============================================================

    rhythm_alert = False
    coordination_alert = False

    rhythm_message = ""
    coordination_message = ""
    physiology_message = ""
    advice_message = ""

    # Tempo Ratio:
    # good rhythm is close to 3:1, not simply "the higher the better"
    if np.isnan(tempo_ratio):
        rhythm_message = "Tempo Ratio could not be evaluated."
    elif 2.5 <= tempo_ratio <= 3.5:
        rhythm_message = (
            f"The mean Tempo Ratio is {tempo_ratio:.2f}, which is close to the expected 3:1 rhythm. "
            "This suggests that the temporal structure between backswing and downswing is well preserved."
        )
    elif 2.0 <= tempo_ratio < 2.5 or 3.5 < tempo_ratio <= 4.5:
        rhythm_alert = True
        rhythm_message = (
            f"The mean Tempo Ratio is {tempo_ratio:.2f}. "
            "This is not strongly altered, but it is outside the optimal reference zone around 3:1. "
            "The athlete should monitor rhythm consistency across repeated swings."
        )
    else:
        rhythm_alert = True
        rhythm_message = (
            f"The mean Tempo Ratio is {tempo_ratio:.2f}, which indicates an altered swing rhythm. "
            "This may reflect an imbalance between the backswing and downswing phases."
        )

    # Delta t:
    # lower absolute delta_t = better body-club timing transfer
    if np.isnan(delta_t):
        coordination_message = "Body–club timing delay could not be evaluated."
    elif abs(delta_t) <= 0.30:
        coordination_message = (
            f"The mean body–club delay is Δt = {delta_t:.3f} s. "
            "This low delay suggests that the movement transfer from the body to the club is well coordinated."
        )
    elif abs(delta_t) <= 0.50:
        coordination_alert = True
        coordination_message = (
            f"The mean body–club delay is Δt = {delta_t:.3f} s. "
            "This indicates a moderate timing delay between sacral motion and club motion."
        )
    else:
        coordination_alert = True
        coordination_message = (
            f"The mean body–club delay is Δt = {delta_t:.3f} s. "
            "This suggests that the transfer from body motion to club motion may not be optimal."
        )

    # ============================================================
    # PHYSIOLOGICAL STATUS
    # ============================================================

    physiological_status, _, _ = get_hr_status(hr_mean, rmssd_mean)
    physiology_alert = physiological_status == "Altered"

    if np.isnan(hr_mean):
        physiology_message = "HR was not available for this session."
    elif physiology_alert:
        physiology_message = (
            f"The mean HR is {hr_mean:.0f} bpm, indicating increased physiological activation. "
            "This may influence rhythm, decision timing and movement control."
        )
    else:
        physiology_message = (
            f"The mean HR is {hr_mean:.0f} bpm, suggesting that the physiological state remained within the normal prototype range."
        )

    # Local pre-swing HR
    if pre_swing_high_hr:
        advice_message = (
            f"Local HR before the swing reached approximately {pre_swing_hr_mean:.0f} bpm. "
            "Before the next shot, the athlete should pause for a few seconds, relax the grip and shoulders, "
            "take 2–3 slow breaths, and restart the pre-shot routine only when the activation level feels more stable."
        )
    else:
        advice_message = (
            "No clear excessive HR peak was detected immediately before the swings. "
            "The athlete can maintain the same pre-shot routine, focusing on repeating the same rhythm and timing."
        )

    # ============================================================
    # SWING-BY-SWING COMMENT
    # ============================================================

    swing_comment = ""

    if swing_df is not None and not swing_df.empty:
        best_swing = None
        critical_swing = None

        temp = swing_df.copy()

        temp["tempo_distance_from_3"] = np.abs(temp["tempo_ratio"] - 3.0)
        temp["abs_delta_t"] = np.abs(temp["delta_t"])

        # Best swing = closest to 3:1 and lowest delta_t
        temp["technical_score"] = temp["tempo_distance_from_3"] + temp["abs_delta_t"]

        best_row = temp.loc[temp["technical_score"].idxmin()]
        worst_row = temp.loc[temp["technical_score"].idxmax()]

        best_swing = int(best_row["swing_id"])
        critical_swing = int(worst_row["swing_id"])

        swing_comment = (
            f"From the swing-by-swing analysis, Swing {best_swing} appears to be the most technically balanced, "
            f"because it combines a Tempo Ratio close to 3:1 with a relatively low Δt. "
            f"Swing {critical_swing} is the one that should be reviewed more carefully, because its rhythm and/or body–club timing "
            "is less consistent compared with the rest of the session."
        )

    # ============================================================
    # FINAL CLASSIFICATION
    # ============================================================

    technical_alert = rhythm_alert or coordination_alert

    if not technical_alert and not physiology_alert and not pre_swing_high_hr:
        title = "Stable session"
        output_class = "output-stable"
        global_message = (
            "The session appears stable. The technical indicators and physiological state are both within the expected range."
        )

    elif technical_alert and not physiology_alert and not pre_swing_high_hr:
        title = "Technical aspect to monitor"
        output_class = "output-warning"
        global_message = (
            "The session seems mainly influenced by technical factors. "
            "The athlete should focus on keeping a repeatable 3:1 rhythm and reducing the delay between body and club motion."
        )

    elif not technical_alert and (physiology_alert or pre_swing_high_hr):
        title = "Physiological influence to monitor"
        output_class = "output-warning"
        global_message = (
            "The technical execution is not clearly altered, but the physiological activation appears elevated. "
            "This may affect preparation before the shot even if the movement pattern remains acceptable."
        )

    else:
        title = "Combined technical and physiological influence"
        output_class = "output-critical"
        global_message = (
            "The session appears influenced by both technical and physiological factors. "
            "Rhythm or body–club coordination is altered together with increased physiological activation."
        )

    explanation = (
        f"{global_message}<br><br>"
        f"<b>Technical rhythm:</b> {rhythm_message}<br><br>"
        f"<b>Body–club coordination:</b> {coordination_message}<br><br>"
        f"<b>Physiological state:</b> {physiology_message}<br><br>"
    )

    if swing_comment != "":
        explanation += f"<b>Swing-by-swing insight:</b> {swing_comment}<br><br>"

    explanation += f"<b>Practical advice:</b> {advice_message}"

    return title, explanation, output_class
# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_filtered_preview(club_df, sacral_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=club_df["time_s"],
        y=club_df["a_filt"],
        mode="lines",
        name="Club IMU - filtered",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=sacral_df["time_s"],
        y=sacral_df["a_filt"],
        mode="lines",
        name="Sacral IMU - filtered",
        line=dict(width=3)
    ))

    fig.update_layout(
        title="Filtered resultant acceleration - Club IMU vs Sacral IMU",
        xaxis_title="Time [s]",
        yaxis_title="Resultant acceleration",
        height=500,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#07184A"),
        title_font=dict(color="#07184A"),
        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(color="#07184A")
        ),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    return fig


def plot_acceleration(df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time_s"],
        y=df["a_tot"],
        mode="lines",
        name="Raw resultant acceleration",
        line=dict(width=1)
    ))

    if "a_filt" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time_s"],
            y=df["a_filt"],
            mode="lines",
            name="Filtered acceleration",
            line=dict(width=3)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Resultant acceleration",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=60)
    )

    return fig


def plot_combined_club_sacral(club_df, sacral_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=club_df["time_s"],
        y=club_df["a_filt"],
        mode="lines",
        name="Club IMU",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=sacral_df["time_s"],
        y=sacral_df["a_filt"],
        mode="lines",
        name="Sacral IMU",
        line=dict(width=3)
    ))

    fig.update_layout(
        title="Body–Club Coordination: Club vs Sacral Acceleration",
        xaxis_title="Time (s)",
        yaxis_title="Filtered resultant acceleration",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=60)
    )

    return fig


def plot_club_events_five_swings(club_df, swing_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=club_df["time_s"],
        y=club_df["a_filt"],
        mode="lines",
        name="Club filtered acceleration",
        line=dict(width=3)
    ))

    for _, row in swing_df.iterrows():
        fig.add_vline(
            x=row["t_start"],
            line_dash="dot",
            line_width=1,
            annotation_text=f"S{int(row['swing_id'])} start",
            annotation_position="top"
        )

        fig.add_vline(
            x=row["t_peak_club"],
            line_dash="solid",
            line_width=2,
            annotation_text=f"S{int(row['swing_id'])} peak",
            annotation_position="top"
        )

        fig.add_vline(
            x=row["t_end"],
            line_dash="dash",
            line_width=1,
            annotation_text=f"S{int(row['swing_id'])} end",
            annotation_position="top"
        )

    fig.update_layout(
        title="Club IMU — five-swing Tempo Ratio detection",
        xaxis_title="Time (s)",
        yaxis_title="Filtered resultant acceleration",
        height=460,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    return fig


def plot_swing_metric_variation(swing_df, y_col, title, y_title):
    fig = go.Figure()

    mean_value = swing_df[y_col].mean()

    fig.add_trace(go.Scatter(
        x=swing_df["swing_id"],
        y=swing_df[y_col],
        mode="lines+markers",
        name=y_title,
        marker=dict(size=11),
        line=dict(width=3)
    ))

    fig.add_hline(
        y=mean_value,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Mean = {mean_value:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Swing Number",
        yaxis_title=y_title,
        height=420,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#07184A"),
        title_font=dict(color="#07184A"),
        margin=dict(l=20, r=20, t=60, b=60)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    return fig


def plot_tempo_ratio_vs_delta_t(swing_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=swing_df["swing_id"],
        y=swing_df["tempo_ratio"],
        mode="lines+markers",
        name="Tempo Ratio",
        yaxis="y1",
        marker=dict(size=10),
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=swing_df["swing_id"],
        y=swing_df["delta_t"],
        mode="lines+markers",
        name="Δt = t_peak,sacral − t_peak,club",
        yaxis="y2",
        marker=dict(size=10),
        line=dict(width=3, dash="dash")
    ))

    fig.update_layout(
        title="Tempo Ratio vs Body–Club Peak Timing Delay",
        xaxis=dict(
            title="Swing Number",
            color="#07184A",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=False
        ),
        yaxis=dict(
            title="Tempo Ratio",
            side="left",
            color="#07184A",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=False
        ),
        yaxis2=dict(
            title="Δt A5-A4 [s]",
            overlaying="y",
            side="right",
            color="#07184A",
            showgrid=False,
            zeroline=False
        ),
        height=450,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#07184A"),
        title_font=dict(color="#07184A"),
        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(color="#07184A")
        ),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    return fig

def plot_motion_physiology_overlay(club_df, sacral_df, hr_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=club_df["time_s"],
        y=club_df["a_filt"],
        mode="lines",
        name="Club acceleration",
        yaxis="y1",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=sacral_df["time_s"],
        y=sacral_df["a_filt"],
        mode="lines",
        name="Sacral acceleration",
        yaxis="y1",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=hr_df["time_s"],
        y=hr_df["hr"],
        mode="lines+markers",
        name="Heart Rate",
        yaxis="y2",
        line=dict(width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Integrated Motion–Physiology Overlay",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Filtered resultant acceleration", side="left"),
        yaxis2=dict(title="Heart Rate (bpm)", overlaying="y", side="right"),
        height=470,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    return fig


def plot_hr_hrv(hr_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hr_df["time_s"],
        y=hr_df["hr"],
        mode="lines+markers",
        name="HR (bpm)",
        yaxis="y1",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=hr_df["time_s"],
        y=hr_df["rmssd"],
        mode="lines+markers",
        name="RMSSD (ms)",
        yaxis="y2",
        line=dict(width=3)
    ))

    fig.update_layout(
        title="Physiological State During Session",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="HR (bpm)", side="left"),
        yaxis2=dict(title="RMSSD (ms)", overlaying="y", side="right"),
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=60)
    )

    return fig
def plot_hr_trend_deviation(hr_df, swing_times_s=None, pre_window_s=3.0, close_window_s=0.8):
    fig = go.Figure()

    hr_mean = hr_df["hr"].mean()
    hr_std = hr_df["hr"].std()

    if swing_times_s is None:
        swing_times_s = hr_df.attrs.get("bin_swing_times_s", None)

    fig.add_trace(go.Scatter(
        x=hr_df["time_s"],
        y=hr_df["hr"],
        mode="lines+markers",
        name="HR trend",
        line=dict(width=2),
        marker=dict(size=6)
    ))

    fig.add_hrect(
        y0=hr_mean - hr_std,
        y1=hr_mean + hr_std,
        fillcolor="rgba(42, 157, 143, 0.15)",
        line_width=0,
        annotation_text="Mean ± 1 SD",
        annotation_position="top left"
    )

    fig.add_hline(
        y=hr_mean,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Mean HR = {hr_mean:.0f} bpm",
        annotation_position="top left"
    )

    outliers = hr_df[hr_df["outside_1sd"] == True]

    fig.add_trace(go.Scatter(
        x=outliers["time_s"],
        y=outliers["hr"],
        mode="markers",
        name="Deviation > 1 SD",
        marker=dict(size=12, symbol="diamond")
    ))

    if swing_times_s is not None and len(swing_times_s) > 0:
        y_top = hr_df["hr"].max()

        for i, t_swing in enumerate(swing_times_s):
            fig.add_vrect(
                x0=t_swing - pre_window_s,
                x1=t_swing,
                fillcolor="rgba(255, 193, 7, 0.10)",
                line_width=0,
                layer="below"
            )

            fig.add_vrect(
                x0=t_swing - close_window_s,
                x1=t_swing,
                fillcolor="rgba(255, 193, 7, 0.28)",
                line_width=0,
                layer="below"
            )

            fig.add_vline(
                x=t_swing,
                line_dash="dash",
                line_width=2,
                line_color="rgba(220, 38, 38, 0.85)",
                annotation_text=f"S{i+1}",
                annotation_position="top"
            )

            fig.add_trace(go.Scatter(
                x=[t_swing],
                y=[y_top],
                mode="markers",
                name=f"S{i+1}",
                marker=dict(size=10, symbol="triangle-down"),
                showlegend=False
            ))

    fig.update_layout(
        title="Heart Rate Trend and Deviation During the Session",
        xaxis_title="Time [s]",
        yaxis_title="Heart Rate [bpm]",
        height=470,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#07184A"),
        title_font=dict(color="#07184A"),
        legend=dict(
            orientation="h",
            y=-0.25,
            font=dict(color="#07184A")
        ),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        color="#07184A"
    )

    return fig
    # ============================================================
    # SWING MARKERS ON HR GRAPH
    # ============================================================

    if swing_analysis_df is not None and not swing_analysis_df.empty:
        y_top = hr_df["hr"].max()

        for _, row in swing_analysis_df.iterrows():
            swing_id = int(row["swing_id"])
            t_swing = row["t_peak_club"]

            # pre-swing window
            fig.add_vrect(
                x0=t_swing - pre_window_s,
                x1=t_swing,
                fillcolor="rgba(255, 193, 7, 0.16)",
                line_width=0,
                layer="below"
            )

            # swing / impact marker
            fig.add_vline(
                x=t_swing,
                line_dash="dash",
                line_width=2,
                line_color="rgba(220, 38, 38, 0.85)",
                annotation_text=f"S{swing_id}",
                annotation_position="top"
            )

            fig.add_trace(go.Scatter(
                x=[t_swing],
                y=[y_top],
                mode="markers",
                name=f"S{swing_id} impact",
                marker=dict(size=10, symbol="triangle-down"),
                showlegend=False
            ))

    fig.update_layout(
        title="Heart Rate Trend and Deviation During the Session",
        xaxis_title="Time [s]",
        yaxis_title="Heart Rate [bpm]",
        height=470,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70)
    )

    return fig
# ============================================================
# UI COMPONENTS
# ============================================================

def metric_card(title, value, description):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-name">{title}</div>
        <div class="metric-main">{value}</div>
        <div class="metric-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def physiological_metric_card(title, value, description, status, icon, status_class):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-name">{title}</div>
        <div class="metric-main">{value}</div>
        <div class="metric-desc">{description}</div>
        <div class="{status_class}">
            {icon} {status}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOGIN GATE
# ============================================================

require_login()

if st.session_state.user_role == "Athlete" and not st.session_state.selected_athlete_name:
    st.session_state.selected_athlete_name = st.session_state.user_name
    st.session_state.selected_athlete = "self"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⛳ Golf App")
st.sidebar.markdown("Professional session validation")

st.sidebar.markdown("---")
st.sidebar.markdown("### User profile")
st.sidebar.markdown(f"**Name:** {st.session_state.user_name}")
st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
st.sidebar.markdown(f"**Club:** {st.session_state.user_club}")

if st.session_state.selected_athlete_name:
    st.sidebar.markdown(f"**Selected athlete:** {st.session_state.selected_athlete_name}")

if st.session_state.user_role == "Athlete":
    if st.session_state.data_consent:
        st.sidebar.success("Data sharing enabled")
    else:
        st.sidebar.warning("Data sharing disabled")

if st.sidebar.button("Logout"):
    logout()

st.sidebar.markdown("---")

PAGES = ["Home", "Session Analysis", "Metrics Explanation"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# If a button asked to change page, apply it BEFORE creating the sidebar widget
if "pending_page" in st.session_state:
    st.session_state.page = st.session_state.pending_page
    del st.session_state.pending_page

# Keep sidebar radio synchronized BEFORE the widget is created
if "sidebar_navigation" not in st.session_state:
    st.session_state.sidebar_navigation = st.session_state.page
else:
    st.session_state.sidebar_navigation = st.session_state.page

st.sidebar.radio(
    "Navigation",
    PAGES,
    key="sidebar_navigation"
)

st.session_state.page = st.session_state.sidebar_navigation
st.sidebar.markdown("---")

fs = st.sidebar.selectbox(
    "IMU sampling frequency",
    [100, 120, 200],
    index=1
)

club_cutoff = st.sidebar.slider(
    "Club IMU low-pass cut-off frequency",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

sacral_cutoff = st.sidebar.slider(
    "Sacral IMU low-pass cut-off frequency",
    min_value=3,
    max_value=20,
    value=6,
    step=1
)

club_prominence = st.sidebar.slider(
    "Club peak prominence",
    min_value=1,
    max_value=80,
    value=20,
    step=1
)

st.sidebar.caption("Suggested setup: 120 Hz, club 10 Hz, sacral 6 Hz.")

# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Prototype validation dashboard</div>
        <div class="hero-title">Golf Swing Validation App</div>
        <div class="hero-subtitle">
            A session-based platform for professional golfers, designed to connect swing rhythm,
            body–club coordination and physiological state.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_role == "Coach":
        st.markdown(f"""
        <div class="formula-box">
            <b>Coach access:</b> you are logged in as a coach of 
            <b>{st.session_state.user_club}</b>. 
            Select one athlete from your club to review their session data.
        </div>
        """, unsafe_allow_html=True)

        render_coach_athlete_roster()

    else:
        if st.session_state.data_consent:
            consent_text = "Your session data can be shared with coaches from your same golf club."
        else:
            consent_text = "Your session data will remain private and will not be shared with coaches."

        st.markdown(f"""
        <div class="formula-box">
            <b>Athlete access:</b> you are logged in as an athlete of 
            <b>{st.session_state.user_club}</b>.<br>
            {consent_text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Training Session Workflow</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-subtitle">
    The app follows a simple three-step workflow: upload the sensor data, analyse the swing performance,
    and receive an integrated interpretation with practical feedback.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
    <div class="metric-selection-card">
    <div class="metric-icon">📤</div>
    <div class="metric-selection-title">1. Upload Data</div>
    <div class="metric-selection-text">
    Import the data collected during the training session from the club IMU, body IMU and HR/HRV sensor.
    This allows the app to combine technical movement data with physiological information.
    </div>
    <div class="metric-status-ready">Input phase</div>
    </div>
    """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
    <div class="metric-selection-card">
    <div class="metric-icon">📊</div>
    <div class="metric-selection-title">2. Analyse Performance</div>
    <div class="metric-selection-text">
    The app calculates key indicators such as Tempo Ratio, body–club delay Δt, repeatability through CV,
    heart rate and HRV/RMSSD.
    </div>
    <div class="metric-status-ready">Analysis phase</div>
    </div>
    """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
    <div class="metric-selection-card">
    <div class="metric-icon">💡</div>
    <div class="metric-selection-title">3. Receive Interpretation</div>
    <div class="metric-selection-text">
    The final output combines technical and physiological information to explain how the session went
    and provide swing-by-swing recommendations.
    </div>
    <div class="metric-status-ready">Feedback phase</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">Quick access</div>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)

    with q1:
        if st.button("Start Session Analysis", use_container_width=True):
            st.session_state.pending_page = "Session Analysis"
            st.rerun()

    with q2:
        if st.button("Open Metrics Explanation", use_container_width=True):
            st.session_state.pending_page = "Metrics Explanation"
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-title">What the app adds</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Club IMU</div>
            <div class="card-text">
            Measures the temporal structure of the swing and estimates backswing, downswing and impact timing.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <div class="card-title">Sacral IMU</div>
            <div class="card-text">
            Evaluates body motion and the timing transfer from the body to the club through Δt.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
            <div class="card-title">HR / HRV</div>
            <div class="card-text">
            Provides the physiological context, helping understand whether movement quality changes
            under different athlete conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SESSION ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Session Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Session analysis</div>
        <div class="hero-title">Integrated Swing Interpretation</div>
        <div class="hero-subtitle">
            Upload club and sacral IMU data to calculate Tempo Ratio, body–club timing delay and repeatability.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_home_from_session"):
        go_to_page("Home")
    if st.session_state.user_role == "Coach" and not st.session_state.selected_athlete_name:
        st.warning("Please select an athlete from the Home page before opening the session analysis.")
        st.stop()

    st.markdown(f"""
    <div class="selected-athlete-box">
        <b>Current athlete:</b> {st.session_state.selected_athlete_name}
    </div>
    """, unsafe_allow_html=True)
    


    st.markdown('<div class="section-title">Choose the performance area to analyse</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="section-subtitle">
Select the training dimension you want to explore. Each area focuses on a different aspect of the swing and athlete performance.
</div>
""", unsafe_allow_html=True)

    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.markdown("""
<div class="metric-selection-card">
<div class="metric-icon">⏱️</div>
<div class="metric-selection-title">Rhythm</div>
<div class="metric-selection-text">
Analyse swing timing, Tempo Ratio, repeatability across the five swings, body–club timing delay and physiological context.
</div>
<div class="metric-status-ready">Available now</div>
</div>
""", unsafe_allow_html=True)

        if st.button("Open Rhythm analysis", use_container_width=True):
            select_metric("Ritmo")

    with mcol2:
        st.markdown("""
<div class="metric-selection-card">
<div class="metric-icon">🧍</div>
<div class="metric-selection-title">Stability</div>
<div class="metric-selection-text">
Future section dedicated to trunk stability, body control, postural consistency and movement repeatability during the swing.
</div>
<div class="metric-status-coming">Coming soon</div>
</div>
""", unsafe_allow_html=True)

        if st.button("Open Stability analysis", use_container_width=True):
            select_metric("Stabilità")

    with mcol3:
        st.markdown("""
<div class="metric-selection-card">
<div class="metric-icon">⚡</div>
<div class="metric-selection-title">Power</div>
<div class="metric-selection-text">
Future section dedicated to acceleration intensity, force-related indicators and power expression during the swing.
</div>
<div class="metric-status-coming">Coming soon</div>
</div>
""", unsafe_allow_html=True)

        if st.button("Open Power analysis", use_container_width=True):
            select_metric("Forza")

    st.markdown("---")

    if st.session_state.selected_metric is None:
        st.info("Select a metric to open the corresponding analysis.")
        st.stop()

    if st.session_state.selected_metric != "Ritmo":
        st.markdown(f"""
        <div class="formula-box">
            <b>{st.session_state.selected_metric}</b> section is currently a placeholder.
            This part can be developed later with dedicated indicators.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Back to metric selection", use_container_width=True):
            st.session_state.selected_metric = None
            st.rerun()

        st.stop()

    if st.button("← Back to performance areas", key="back_to_metric_selection"):
        st.session_state.selected_metric = None
        st.rerun()

    st.markdown('<div class="section-title">Ritmo Analysis</div>', unsafe_allow_html=True)


    session_name = st.selectbox(
        "Select session",
        ["Session 1", "Session 2", "Session 3"],
        key="session_selectbox_rhythm"
    )

    data_source = st.radio(
        "Select data source",
        ["Upload files manually", "Use preloaded data for selected session"],
        horizontal=True,
        key="data_source_selector"
    )

    club_file = None
    sacral_file = None
    hr_file = None

    if data_source == "Upload files manually":

        u1, u2, u3 = st.columns(3)

        with u1:
            st.markdown("""
<div class="upload-card">
<div class="upload-title">Club Motion</div>
<div class="upload-icon">🏌️</div>
<div class="upload-description">
Upload the <b>club-mounted IMU</b> file, for example <b>A4_prima.csv</b>.
</div>
</div>
""", unsafe_allow_html=True)

            club_file = st.file_uploader(
                "Upload Club IMU file",
                type=["csv", "xlsx", "xls", "bin"],
                key="club_file"
            )

        with u2:
            st.markdown("""
<div class="upload-card">
<div class="upload-title">Body Motion</div>
<div class="upload-icon">🧍</div>
<div class="upload-description">
Upload the <b>sacral IMU</b> file, for example <b>A5_prima.csv</b>.
</div>
</div>
""", unsafe_allow_html=True)

            sacral_file = st.file_uploader(
                "Upload Sacral IMU file",
                type=["csv", "xlsx", "xls"],
                key="sacral_file"
            )

        with u3:
            st.markdown("""
<div class="upload-card">
<div class="upload-title">Physiological State</div>
<div class="upload-icon">❤️</div>
<div class="upload-description">
Optional: upload an <b>HR/HRV</b> file to add physiological context.
</div>
</div>
""", unsafe_allow_html=True)

            hr_file = st.file_uploader(
                "Upload HR/HRV file",
                type=["csv", "xlsx", "xls", "bin"],
                key="hr_file"
            )

    else:
        if session_name == "Session 1":
            club_file = "data/A4_prima.csv"
            sacral_file = "data/A5_prima.csv"
            hr_file = "data/SignalsSave_051358.bin"

        elif session_name == "Session 2":
            club_file = "data/A4_dopo.csv"
            sacral_file = "data/A5_dopo.csv"
            hr_file = "data/SignalsSave_052232.bin"

        else:
            club_file = None
            sacral_file = None
            hr_file = None

            st.warning(
                "Preloaded data are currently available only for Session 1 and Session 2."
            )

        if club_file is not None and sacral_file is not None:
            st.markdown(f"""
<div class="formula-box">
<b>Preloaded {session_name} data selected.</b><br>
The app is using the local files:<br>
Club IMU: <b>{club_file}</b><br>
Body IMU: <b>{sacral_file}</b><br>
HR/ECG: <b>{hr_file}</b>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    if club_file is not None and sacral_file is not None:

        if data_source == "Upload files manually":
            club_raw = read_uploaded_file(club_file)
            sacral_raw = read_uploaded_file(sacral_file)
        else:
            club_raw = read_local_file(club_file)
            sacral_raw = read_local_file(sacral_file)

        club_df = standardize_imu_columns(club_raw, fs=fs)
        sacral_df = standardize_imu_columns(sacral_raw, fs=fs)

        hr_df = None

        if club_df is not None and sacral_df is not None:

            swing_analysis_df, club_peak_df, sacral_peak_df = analyze_five_swings(
                club_df,
                sacral_df,
                fs=fs,
                club_cutoff=club_cutoff,
                sacral_cutoff=sacral_cutoff,
                expected_swings=5,
                min_distance_s=3.0,
                club_prominence=club_prominence,
                search_window_s=1.5,
                t_start_analysis=5.0,
                pre_flat_window_s=2.0,
                baseline_margin=1.0,
                post_peak_window_s=1.0,
                return_margin=2.0
            )

            if swing_analysis_df.empty:
                st.error(
                    "No swings were detected. Try lowering Club peak prominence "
                    "or check that the club IMU file is loaded correctly."
                )
                st.stop()


            hr_df = None

            if hr_file is not None:

                if data_source == "Upload files manually":

                    if hr_file.name.lower().endswith(".bin"):
                        hr_df = process_ecg_bin_to_hr(hr_file)
                    else:
                        hr_raw = read_uploaded_file(hr_file)
                        hr_df = standardize_hr_columns(hr_raw)

                else:

                    if str(hr_file).lower().endswith(".bin"):
                        hr_df = process_ecg_bin_path_to_hr(hr_file)
                    else:
                        hr_raw = read_local_file(hr_file)
                        hr_df = standardize_hr_columns(hr_raw)
            tempo_ratio = swing_analysis_df["tempo_ratio"].mean()
            delta_t = swing_analysis_df["delta_t"].mean()

            t_peak_club = swing_analysis_df["t_peak_club"].mean()
            t_peak_sacral = swing_analysis_df["t_peak_sacral"].mean()

            cv_tempo_ratio = coefficient_of_variation(swing_analysis_df["tempo_ratio"])
            cv_delta_t = coefficient_of_variation(swing_analysis_df["delta_t"])

            hr_mean = np.nan
            rmssd_mean = np.nan

            if hr_df is not None:
                hr_mean = hr_df["hr"].mean()
                rmssd_mean = hr_df["rmssd"].mean()

            hr_status, hr_icon, hr_status_class = get_hr_status(hr_mean, rmssd_mean)

            pre_swing_hr_df = pd.DataFrame()
            pre_swing_high_hr = False
            pre_swing_hr_mean = np.nan

            if hr_df is not None:
                bin_swing_times_s = hr_df.attrs.get("bin_swing_times_s", None)

                pre_swing_hr_df = compute_local_pre_swing_hr(
                    hr_df=hr_df,
                    swing_times_s=bin_swing_times_s,
                    pre_window_s=3.0,
                     max_window_before_swing_s=1.5,
                    high_hr_threshold=120
                )

                if not pre_swing_hr_df.empty:
                    pre_swing_high_hr = pre_swing_hr_df["High_HR_before_swing"].any()
                    pre_swing_hr_mean = pre_swing_hr_df["HR_pre_swing_max_close"].max()


            output, explanation, output_class = classify_interpretation(
                tempo_ratio=tempo_ratio,
                delta_t=delta_t,
                hr_mean=hr_mean,
                rmssd_mean=rmssd_mean,
                pre_swing_high_hr=pre_swing_high_hr,
                pre_swing_hr_mean=pre_swing_hr_mean,
                swing_df=swing_analysis_df
            )

            # ============================================================
            # SIGNAL PREVIEW
            # ============================================================

            st.markdown('<div class="section-title">Signal Preview</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="formula-box">
                This plot shows the filtered resultant acceleration of the two synchronized IMU sensors.
                The club IMU is filtered at 10 Hz, while the sacral IMU is filtered at 6 Hz.
                The first 5 seconds are excluded from the swing detection, as in the Colab processing.
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(
                plot_filtered_preview(club_peak_df, sacral_peak_df),
                use_container_width=True,
                theme=None,
                key="signal_preview_filtered_imu"
            )

            # ============================================================
            # KEY RESULTS
            # ============================================================

            st.markdown(f'<div class="section-title">{session_name} — Key Results</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                value = "N/A" if np.isnan(tempo_ratio) else f"{tempo_ratio:.2f}"
                metric_card("Mean Tempo Ratio", value, "Mean across 5 swings")

            with m2:
                value = "N/A" if np.isnan(delta_t) else f"{delta_t:.3f} s"
                metric_card("Mean Δt body–club", value, "Mean t_peak,sacral − t_peak,club")

            with m3:
                value = "N/A" if np.isnan(hr_mean) else f"{hr_mean:.0f} bpm"
                physiological_metric_card(
                    "Mean HR",
                    value,
                    "Physiological activation",
                    hr_status,
                    hr_icon,
                    hr_status_class
                )

            with m4:
                value = "N/A" if np.isnan(rmssd_mean) else f"{rmssd_mean:.1f} ms"
                physiological_metric_card(
                    "Mean HRV / RMSSD",
                    value,
                    "Physiological readiness",
                    hr_status,
                    hr_icon,
                    hr_status_class
                )

            # ============================================================
            # SWING BY SWING ANALYSIS
            # ============================================================

            st.markdown("---")
            st.markdown('<div class="section-title">Swing-by-Swing Analysis</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="formula-box">
                For each swing, the app computes:
                <b>T<sub>backswing</sub></b> = t<sub>peak,club</sub> − t<sub>start</sub>,
                <b>T<sub>downswing</sub></b> = t<sub>end</sub> − t<sub>peak,club</sub>,
                <b>Tempo Ratio</b> = T<sub>backswing</sub> / T<sub>downswing</sub>,
                and <b>Δt</b> = t<sub>peak,sacral</sub> − t<sub>peak,club</sub>.
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(swing_analysis_df, use_container_width=True)

            csv_swing = swing_analysis_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download swing-by-swing results",
                data=csv_swing,
                file_name="swing_by_swing_results.csv",
                mime="text/csv"
            )

            c_var1, c_var2 = st.columns(2)

            with c_var1:
                st.plotly_chart(
                    plot_swing_metric_variation(
                        swing_analysis_df,
                        "tempo_ratio",
                        "Tempo Ratio variation across 5 swings",
                        "Tempo Ratio"
                    ),
                    theme=None,
                    use_container_width=True
                )

            with c_var2:
                st.plotly_chart(
                    plot_swing_metric_variation(
                        swing_analysis_df,
                        "delta_t",
                        "Body–Club Peak Delay variation across 5 swings",
                        "Δt [s]"
                    ),
                    theme=None,
                    use_container_width=True
                )

            st.plotly_chart(
                plot_tempo_ratio_vs_delta_t(swing_analysis_df),
                theme=None,
                use_container_width=True
            )

            st.markdown(f"""
            <div class="formula-box">
                <b>CV(Tempo Ratio):</b> {cv_tempo_ratio:.2f}%<br>
                <b>CV(Δt body–club):</b> {cv_delta_t:.2f}%<br>
                <b>Interpretation:</b> a lower CV indicates higher repeatability across the five swings.
            </div>
            """, unsafe_allow_html=True)



            

           

            if hr_df is not None:
                st.markdown("---")
                st.markdown('<div class="section-title">Heart Rate Trend During Session</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="formula-box">
                    The ECG .bin file is processed independently from the IMU signals.
                    Swing markers on this graph are detected from the acceleration channels of the same .bin file,
                    using the acceleration direction change pattern positive → negative → positive.
                    The graph shows the beat-by-beat HR trend during the session, the mean HR,
                    the ±1 SD variability band and the HR points that deviate more than one standard deviation.
                </div>
                """, unsafe_allow_html=True)


                bin_swing_times_s = hr_df.attrs.get("bin_swing_times_s", None)

                st.plotly_chart(
                    plot_hr_trend_deviation(
                        hr_df,
                        swing_times_s=bin_swing_times_s,
                        pre_window_s=3.0
                    ),
                    use_container_width=True,
                    theme=None,
                    key="hr_trend_with_bin_swing_markers"
                )

                if not pre_swing_hr_df.empty:
                    st.markdown("---")
                    st.markdown('<div class="section-title">Local Pre-Swing HR Check</div>', unsafe_allow_html=True)

                    st.markdown("""
                    <div class="formula-box">
                        The yellow areas in the HR graph represent the 3 seconds before each detected swing.
                        The table below shows the local HR used by the app to decide whether to display the
                        relax-and-breathe advice.
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(pre_swing_hr_df, use_container_width=True)

                    csv_pre_swing_hr = pre_swing_hr_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download local pre-swing HR",
                        data=csv_pre_swing_hr,
                        file_name="local_pre_swing_hr.csv",
                        mime="text/csv"
                    )

                hr_mean_session = hr_df["hr"].mean()
                hr_std_session = hr_df["hr"].std()
                rmssd_session = hr_df["rmssd"].mean()

                csv_hr = hr_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download HR results",
                    data=csv_hr,
                    file_name="hr_trend_results.csv",
                    mime="text/csv"
                )
                if "ecg_raw" in hr_df.attrs and "ecg_filtered" in hr_df.attrs and "r_peaks" in hr_df.attrs:

                    ecg_raw = hr_df.attrs["ecg_raw"]
                    ecg_filtered = hr_df.attrs["ecg_filtered"]
                    r_peaks = hr_df.attrs["r_peaks"]

                    ecg_export_df = pd.DataFrame({
                        "sample": np.arange(len(ecg_raw)),
                        "time_s": np.arange(len(ecg_raw)) / FS_ECG,
                        "ecg_raw": ecg_raw,
                        "ecg_filtered_5_20": ecg_filtered
                    })

                    ecg_export_df["r_peak"] = 0
                    ecg_export_df.loc[r_peaks, "r_peak"] = 1

                    csv_ecg = ecg_export_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download ECG raw + filtered + R-peaks",
                        data=csv_ecg,
                        file_name="ecg_raw_filtered_rpeaks.csv",
                        mime="text/csv"
                    )
                st.markdown(f"""
                <div class="formula-box">
                    <b>Mean HR:</b> {hr_mean_session:.1f} bpm<br>
                    <b>HR standard deviation:</b> {hr_std_session:.1f} bpm<br>
                    <b>Global RMSSD:</b> {rmssd_session:.1f} ms<br>
                    <b>Note:</b> HR is displayed on its own ECG time axis and is not synchronized with the IMU signals.
                </div>
                """, unsafe_allow_html=True)



            # ============================================================
            # INTEGRATED INTERPRETATION
            # ============================================================

            st.markdown("---")
            st.markdown('<div class="section-title">Integrated Interpretation</div>', unsafe_allow_html=True)

            # Header class based on output severity
            if output_class == "output-stable":
                header_class = "interpretation-header-stable"
            elif output_class == "output-warning":
                header_class = "interpretation-header-warning"
            else:
                header_class = "interpretation-header-critical"

            # ============================================================
            # TECHNICAL RHYTHM CARD
            # ============================================================

            if np.isnan(tempo_ratio):
                rhythm_label = "N/A"
                rhythm_status = "Not available"
                rhythm_text = "Tempo Ratio could not be evaluated for this session."
            elif 2.5 <= tempo_ratio <= 3.5:
                rhythm_label = f"{tempo_ratio:.2f}"
                rhythm_status = "Good rhythm"
                rhythm_text = (
                    "Tempo Ratio is close to the 3:1 reference, suggesting a balanced "
                    "relationship between backswing and downswing."
                )
            elif 2.0 <= tempo_ratio < 2.5 or 3.5 < tempo_ratio <= 4.5:
                rhythm_label = f"{tempo_ratio:.2f}"
                rhythm_status = "To monitor"
                rhythm_text = (
                    "Tempo Ratio is outside the ideal 3:1 zone. Rhythm is not severely altered, "
                    "but consistency should be monitored across swings."
                )
            else:
                rhythm_label = f"{tempo_ratio:.2f}"
                rhythm_status = "Altered rhythm"
                rhythm_text = (
                    "Tempo Ratio is far from the 3:1 reference, suggesting an imbalance "
                    "between backswing and downswing."
                )

            # ============================================================
            # BODY–CLUB TRANSFER CARD
            # ============================================================

            if np.isnan(delta_t):
                transfer_label = "N/A"
                transfer_status = "Not available"
                transfer_text = "Body–club timing could not be evaluated."
            elif abs(delta_t) <= 0.30:
                transfer_label = f"{delta_t:.3f} s"
                transfer_status = "Good transfer"
                transfer_text = (
                    "Low Δt suggests that body motion and club motion are well coordinated, "
                    "with an effective transfer from the body to the club."
                )
            elif abs(delta_t) <= 0.50:
                transfer_label = f"{delta_t:.3f} s"
                transfer_status = "Moderate delay"
                transfer_text = (
                    "Δt indicates a moderate delay between sacral motion and club motion. "
                    "The transfer should be monitored."
                )
            else:
                transfer_label = f"{delta_t:.3f} s"
                transfer_status = "High delay"
                transfer_text = (
                    "High Δt suggests that the transfer from body motion to club motion "
                    "may not be optimal."
                )

            # ============================================================
            # PHYSIOLOGICAL STATE CARD
            # ============================================================

            if np.isnan(hr_mean):
                physio_label = "N/A"
                physio_status = "Not available"
                physio_text = "HR was not available for this session."
            elif hr_mean > 120 or pre_swing_high_hr:
                physio_label = f"{hr_mean:.0f} bpm"
                physio_status = "High activation"
                physio_text = (
                    "Physiological activation appears elevated and may influence preparation, "
                    "rhythm and movement control."
                )
            else:
                physio_label = f"{hr_mean:.0f} bpm"
                physio_status = "Normal activation"
                physio_text = "Mean HR remained within the normal prototype range."

            # ============================================================
            # SWING-BY-SWING INSIGHT
            # ============================================================

            best_swing_text = "Swing-by-swing technical insight was not available."

            if swing_analysis_df is not None and not swing_analysis_df.empty:
                temp_df = swing_analysis_df.copy()

                temp_df["tempo_distance_from_3"] = np.abs(temp_df["tempo_ratio"] - 3.0)
                temp_df["abs_delta_t"] = np.abs(temp_df["delta_t"])

                temp_df["technical_score"] = (
                    temp_df["tempo_distance_from_3"] +
                    temp_df["abs_delta_t"]
                )

                best_row = temp_df.loc[temp_df["technical_score"].idxmin()]
                worst_row = temp_df.loc[temp_df["technical_score"].idxmax()]

                best_swing = int(best_row["swing_id"])
                worst_swing = int(worst_row["swing_id"])

                best_swing_text = (
                    f"Swing {best_swing} appears to be the most balanced because it combines "
                    f"a Tempo Ratio closer to 3:1 with a lower body–club delay. "
                    f"Swing {worst_swing} should be reviewed more carefully because rhythm "
                    f"and/or transfer timing were less consistent."
                )

            # ============================================================
            # PRACTICAL ADVICE
            # ============================================================

            if pre_swing_high_hr:
                advice_text = (
                    f"Pre-swing HR reached approximately {pre_swing_hr_mean:.0f} bpm. "
                    "Before the next shot, the athlete should wait a few seconds, relax grip "
                    "and shoulders, take 2–3 slow breaths, and restart the pre-shot routine "
                    "only when activation feels stable."
                )
            elif not np.isnan(tempo_ratio) and (tempo_ratio < 2.5 or tempo_ratio > 3.5):
                advice_text = (
                    "Focus on repeating a more stable swing tempo. The goal is not to swing "
                    "slower or faster, but to keep the backswing-to-downswing relationship "
                    "close to a repeatable 3:1 pattern."
                )
            elif not np.isnan(delta_t) and abs(delta_t) > 0.30:
                advice_text = (
                    "Focus on body–club coordination. The athlete should work on transferring "
                    "motion smoothly from the body to the club, reducing the timing delay "
                    "between sacral and club peaks."
                )
            else:
                advice_text = (
                    "The session appears stable. The athlete can maintain the same routine, "
                    "focusing on repeating the same rhythm and body–club timing across the next swings."
                )

            # ============================================================
            # MAIN INTERPRETATION HEADER
            # IMPORTANT: HTML STARTS AT BEGINNING OF LINE
            # ============================================================

            st.markdown(f"""
            <div class="interpretation-dashboard">
            <div class="{header_class}">
            <div class="interpretation-title">{output}</div>
            <div class="interpretation-subtitle">
            This is the main output of the app. It combines technical swing indicators and physiological state
            to understand whether the session was influenced mainly by technique, physiology, both, or neither.
            </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # THREE SUMMARY CARDS
            # ============================================================

            i1, i2, i3 = st.columns(3)

            with i1:
                st.markdown(f"""
            <div class="insight-card">
            <div class="insight-card-title">Technical Rhythm</div>
            <div class="insight-card-value">{rhythm_label}</div>
            <div class="insight-card-text">
            <b>{rhythm_status}</b><br>
            {rhythm_text}
            </div>
            </div>
            """, unsafe_allow_html=True)

            with i2:
                st.markdown(f"""
            <div class="insight-card">
            <div class="insight-card-title">Body–Club Transfer</div>
            <div class="insight-card-value">{transfer_label}</div>
            <div class="insight-card-text">
            <b>{transfer_status}</b><br>
            {transfer_text}
            </div>
            </div>
            """, unsafe_allow_html=True)

            with i3:
                st.markdown(f"""
            <div class="insight-card">
            <div class="insight-card-title">Physiological State</div>
            <div class="insight-card-value">{physio_label}</div>
            <div class="insight-card-text">
            <b>{physio_status}</b><br>
            {physio_text}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # INTERACTIVE EXPANDERS
            # ============================================================

            with st.expander("Swing-by-swing interpretation", expanded=True):

                st.markdown("""
                This section evaluates each swing individually. 
                A good technical swing should combine a Tempo Ratio close to the 3:1 reference 
                with a low body–club timing delay Δt. If available, pre-swing HR is also considered 
                to understand whether the shot was performed under high physiological activation.
                """)

                swing_detail_df = swing_analysis_df.copy()

                # Add HR information swing-by-swing if available
                if pre_swing_hr_df is not None and not pre_swing_hr_df.empty:
                    swing_detail_df = swing_detail_df.merge(
                        pre_swing_hr_df[[
                            "swing_id",
                            "HR_pre_swing_max_close",
                            "High_HR_before_swing"
                        ]],
                        on="swing_id",
                        how="left"
                    )
                else:
                    swing_detail_df["HR_pre_swing_max_close"] = np.nan
                    swing_detail_df["High_HR_before_swing"] = False

                # Technical interpretation for each swing
                interpretation_rows = []

                for _, row in swing_detail_df.iterrows():

                    swing_id = int(row["swing_id"])
                    tr = row["tempo_ratio"]
                    dt = row["delta_t"]
                    hr_pre = row["HR_pre_swing_max_close"]
                    high_hr = row["High_HR_before_swing"]

                    # Tempo Ratio interpretation
                    if np.isnan(tr):
                        rhythm_status = "Not available"
                        rhythm_comment = "Tempo Ratio could not be evaluated."
                    elif 2.5 <= tr <= 3.5:
                        rhythm_status = "Good rhythm"
                        rhythm_comment = "Tempo Ratio is close to the 3:1 reference."
                    elif 2.0 <= tr < 2.5 or 3.5 < tr <= 4.5:
                        rhythm_status = "Rhythm to monitor"
                        rhythm_comment = "Tempo Ratio is outside the optimal 3:1 zone but not severely altered."
                    else:
                        rhythm_status = "Altered rhythm"
                        rhythm_comment = "Tempo Ratio is far from the 3:1 reference."

                    # Body–club transfer interpretation
                    if np.isnan(dt):
                        transfer_status = "Not available"
                        transfer_comment = "Body–club timing delay could not be evaluated."
                    elif abs(dt) <= 0.30:
                        transfer_status = "Good transfer"
                        transfer_comment = "Low Δt suggests good transfer from body motion to club motion."
                    elif abs(dt) <= 0.50:
                        transfer_status = "Moderate delay"
                        transfer_comment = "Moderate Δt suggests body–club timing should be monitored."
                    else:
                        transfer_status = "High delay"
                        transfer_comment = "High Δt suggests that body–club coordination may be altered."

                    # HR interpretation
                    if np.isnan(hr_pre):
                        hr_status = "HR not available"
                        hr_comment = "No local pre-swing HR value was available."
                    elif high_hr:
                        hr_status = "High pre-swing activation"
                        hr_comment = (
                            f"Pre-swing HR reached {hr_pre:.0f} bpm. "
                            "The athlete should pause, breathe slowly and restart the routine calmly."
                        )
                    else:
                        hr_status = "Normal pre-swing activation"
                        hr_comment = f"Pre-swing HR was {hr_pre:.0f} bpm."

                    # Final swing advice considering rhythm, body–club transfer and physiological state
                    if rhythm_status == "Good rhythm" and transfer_status == "Good transfer" and not high_hr:
                        final_advice = (
                            "This swing appears technically stable. Tempo Ratio is close to the 3:1 reference, "
                            "body–club transfer is well coordinated, and pre-swing HR does not appear elevated. "
                            "The athlete should try to repeat this rhythm, timing and pre-shot routine."
                        )

                    elif rhythm_status == "Good rhythm" and transfer_status == "Good transfer" and high_hr:
                        final_advice = (
                            "Technically, this swing is acceptable because rhythm and body–club transfer are both good. "
                            "However, pre-swing HR is elevated, suggesting high physiological activation before the shot. "
                            "Advice: before the next swing, wait a few seconds, relax grip and shoulders, take 2–3 slow breaths, "
                            "and restart the pre-shot routine when activation feels more stable."
                        )

                    elif rhythm_status != "Good rhythm" and transfer_status == "Good transfer" and not high_hr:
                        final_advice = (
                            "Body–club transfer is acceptable, meaning the movement is transmitted well from the body to the club. "
                            "However, the swing rhythm is not optimal. The athlete should focus on making the backswing-to-downswing "
                            "relationship more repeatable and closer to the 3:1 reference."
                        )

                    elif rhythm_status != "Good rhythm" and transfer_status == "Good transfer" and high_hr:
                        final_advice = (
                            "Body–club transfer is acceptable, but rhythm is altered and pre-swing HR is elevated. "
                            "This suggests that physiological activation may be affecting the temporal structure of the swing. "
                            "Advice: pause before the next shot, breathe slowly, relax the grip, and then focus on reproducing "
                            "a smoother and more repeatable 3:1 rhythm."
                        )

                    elif rhythm_status == "Good rhythm" and transfer_status != "Good transfer" and not high_hr:
                        final_advice = (
                            "The swing rhythm is acceptable, but body–club transfer is not optimal. "
                            "The athlete keeps a good backswing-to-downswing structure, but the timing between sacral motion "
                            "and club motion should be improved. Advice: work on transferring motion smoothly from the body to the club."
                        )

                    elif rhythm_status == "Good rhythm" and transfer_status != "Good transfer" and high_hr:
                        final_advice = (
                            "The rhythm is acceptable, but body–club transfer is not optimal and pre-swing HR is elevated. "
                            "This may indicate that high activation is interfering with the quality of movement transfer. "
                            "Advice: take a short pause, breathe slowly, relax the shoulders and grip, then repeat the swing "
                            "with attention to smooth body-to-club coordination."
                        )

                    elif rhythm_status != "Good rhythm" and transfer_status != "Good transfer" and not high_hr:
                        final_advice = (
                            "This swing should be reviewed carefully because both rhythm and body–club transfer are not optimal. "
                            "The issue appears mainly technical rather than physiological. Advice: work first on stabilizing "
                            "the Tempo Ratio close to 3:1, then reduce the body–club timing delay."
                        )

                    elif rhythm_status != "Good rhythm" and transfer_status != "Good transfer" and high_hr:
                        final_advice = (
                            "This swing shows a combined alteration: rhythm is not optimal, body–club transfer is not optimal, "
                            "and pre-swing HR is elevated. The execution may be influenced by both technical and physiological factors. "
                            "Advice: stop the sequence briefly, recover breathing, relax grip and shoulders, and restart the pre-shot routine "
                            "before attempting to reproduce a smoother rhythm and better body–club timing."
                        )

                    else:
                        final_advice = (
                            "This swing should be reviewed carefully because one or more indicators could not be fully evaluated."
                        )

                    interpretation_rows.append({
                        "Swing": swing_id,
                        "Tempo Ratio": np.nan if np.isnan(tr) else round(tr, 2),
                        "Rhythm status": rhythm_status,
                        "Δt body–club [s]": np.nan if np.isnan(dt) else round(dt, 3),
                        "Transfer status": transfer_status,
                        "Pre-swing HR max [bpm]": np.nan if np.isnan(hr_pre) else round(hr_pre, 0),
                        "HR status": hr_status,
                        "Interpretation": final_advice
                    })

                interpretation_df = pd.DataFrame(interpretation_rows)

                st.dataframe(
                    interpretation_df,
                    use_container_width=True,
                    hide_index=True
                )

                selected_swing = st.selectbox(
                    "Select one swing to read the detailed interpretation",
                    interpretation_df["Swing"].tolist()
                )

                selected_row = interpretation_df[
                    interpretation_df["Swing"] == selected_swing
                ].iloc[0]

                st.markdown(f"""
            <div class="coach-advice-box">
            <div class="coach-advice-title">Swing {int(selected_swing)} detailed interpretation</div>
            <div class="coach-advice-text">
            <b>Rhythm:</b> {selected_row["Rhythm status"]}<br>
            <b>Body–club transfer:</b> {selected_row["Transfer status"]}<br>
            <b>Physiological state:</b> {selected_row["HR status"]}<br><br>
            <b>Advice:</b> {selected_row["Interpretation"]}
            </div>
            </div>
            """, unsafe_allow_html=True)
            # ============================================================
            # SAMPLING AND FILTERING
            # ============================================================

            st.markdown("---")
            st.markdown('<div class="section-title">Sampling & Filtering</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="formula-box">
                <b>IMU sampling frequency:</b> {fs} Hz<br>
                <b>Signal used:</b> resultant acceleration 
                a<sub>tot</sub> = √(a<sub>x</sub>² + a<sub>y</sub>² + a<sub>z</sub>²)<br>
                <b>Filtering:</b> fourth-order zero-phase Butterworth low-pass filter<br>
                <b>Club IMU cut-off frequency:</b> {club_cutoff} Hz<br>
                <b>Sacral IMU cut-off frequency:</b> {sacral_cutoff} Hz<br>
                <b>Analysis start:</b> first 5 seconds removed<br>
                <b>Δt definition:</b> t<sub>peak,sacral</sub> − t<sub>peak,club</sub>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="upload-note">
            <h3>Upload data to start the session analysis</h3>
            <p><b>Club IMU and Sacral IMU files are required.</b> HR/HRV is optional.</p>
            <p>Accepted IMU formats:</p>
            <ul>
                <li>Raw Movella CSV with <b>Acc_X, Acc_Y, Acc_Z</b></li>
                <li>Modified CSV with <b>ax, ay, az</b></li>
                <li>Clean file with <b>time_s, ax, ay, az</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# METRICS EXPLANATION PAGE
# ============================================================

elif st.session_state.page == "Metrics Explanation":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Validation metrics</div>
        <div class="hero-title">How the App Interprets the Swing</div>
        <div class="hero-subtitle">
            The prototype connects swing rhythm, body–club coordination and physiological condition.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_home_from_metrics"):
        go_to_page("Home")

    st.markdown('<div class="section-title">1. Tempo Ratio</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
    Tempo Ratio = T<sub>backswing</sub> / T<sub>downswing</sub><br>
    T<sub>backswing</sub> = t<sub>peak,club</sub> − t<sub>start</sub><br>
    T<sub>downswing</sub> = t<sub>end</sub> − t<sub>peak,club</sub>
    </div>
    """, unsafe_allow_html=True)

    st.write("This describes the temporal structure of each swing, extracted from the club-mounted IMU.")

    st.markdown('<div class="section-title">2. Repeatability Across Swings</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
    CV(Tempo Ratio) = SD(Tempo Ratio) / Mean(Tempo Ratio) × 100
    </div>
    """, unsafe_allow_html=True)

    st.write("This shows how stable the swing rhythm is across the five repeated shots in the same session.")

    st.markdown('<div class="section-title">3. Body–Club Coordination</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="formula-box">
    Δt = t<sub>peak,sacral</sub> − t<sub>peak,club</sub>
    </div>
    """, unsafe_allow_html=True)

    st.write("This shows the timing relationship between body motion and club motion.")

    st.markdown('<div class="section-title">4. Physiological State</div>', unsafe_allow_html=True)

    st.write("HR describes physiological activation, while HRV/RMSSD provides information about physiological readiness.")

    st.markdown("""
    <div class="formula-box">
    🟢 Normal = HR and HRV are within the prototype range<br>
    🔴 Altered = HR is high or HRV/RMSSD is low
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">5. Added Value</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="output-stable">
    <h3>Integrated analysis</h3>
    <p>
    By combining club IMU, sacral IMU and HR/HRV, the app helps understand whether rhythm,
    repeatability and coordination are maintained or altered under different physiological conditions.
    </p>
    </div>
    """, unsafe_allow_html=True)