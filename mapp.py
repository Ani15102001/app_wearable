import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import plotly.graph_objects as go


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
    height: 100%;
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
.athlete-photo {
    width: 105px;
    height: 105px;
    border-radius: 50%;
    margin-bottom: 12px;
    background: #EAF4EC;
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
section[data-testid="stSidebar"] {
    background-color: #0B3D2E;
}
section[data-testid="stSidebar"] * {
    color: white !important;
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
        "selected_athlete_name": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    if not st.session_state.logged_in:
        render_login_page()
        st.stop()


# ============================================================
# DEMO ATHLETE DATABASE
# ============================================================

ATHLETES_DB = [
    {
        "id": "anita",
        "name": "Anita",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "Session 03",
        "status": "Data sharing enabled",
        "photo": "https://api.dicebear.com/8.x/personas/svg?seed=Anita"
    },
    {
        "id": "alessandro",
        "name": "Alessandro",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "Session 02",
        "status": "Data sharing enabled",
        "photo": "https://api.dicebear.com/8.x/personas/svg?seed=Alessandro"
    },
    {
        "id": "stefano",
        "name": "Stefano",
        "club": "Milano Golf Club",
        "level": "Professional athlete",
        "last_session": "Session 01",
        "status": "Data sharing enabled",
        "photo": "https://api.dicebear.com/8.x/personas/svg?seed=Stefano"
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


def render_coach_athlete_roster():
    athletes = get_athletes_for_club(st.session_state.user_club)

    st.markdown('<div class="section-title">Athletes from your club</div>', unsafe_allow_html=True)

    cols = st.columns(len(athletes))

    for col, athlete in zip(cols, athletes):
        with col:
            st.markdown(f"""
            <div class="athlete-card">
                <img src="{athlete['photo']}" class="athlete-photo">
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
    Detects swing/impact events from the chest-band .bin acceleration channels.
    These markers are used only on the HR graph because they share the same time axis as ECG.
    """

    acc_resultant = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    acc_filt = lowpass_filter(
        acc_resultant,
        fs=fs,
        cutoff=10,
        order=4
    )

    activity = np.abs(acc_filt - np.median(acc_filt))

    distance_samples = int(2.0 * fs)
    prominence_value = np.std(activity) * 1.2

    peaks, _ = find_peaks(
        activity,
        distance=distance_samples,
        prominence=prominence_value
    )

    if len(peaks) > expected_swings:
        top_idx = np.argsort(activity[peaks])[-expected_swings:]
        peaks = np.sort(peaks[top_idx])

    peaks = np.sort(peaks)

    swing_times_s = peaks / fs

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
    pre_swing_hr_mean=np.nan
):
    rhythm_alert = False
    body_motion_alert = False

    if not np.isnan(tempo_ratio) and (tempo_ratio < 2.0 or tempo_ratio > 4.5):
        rhythm_alert = True

    if not np.isnan(delta_t) and abs(delta_t) > 0.50:
        body_motion_alert = True

    physiological_status, _, _ = get_hr_status(hr_mean, rmssd_mean)
    physiology_alert = physiological_status == "Altered"

    # ============================================================
    # LOCAL PRE-SWING HR ADVICE
    # ============================================================

    if pre_swing_high_hr and not rhythm_alert and not body_motion_alert:
        return (
            "High pre-swing activation",
            f"Local heart rate before the swing sequence appears elevated "
            f"({pre_swing_hr_mean:.0f} bpm on average before the detected swings). "
            "Swing rhythm and body–club coordination are not clearly altered, but the athlete may be starting the shot with high physiological activation. "
            "Advice: take a short pause before the next shot, relax the shoulders and grip, use 2–3 slow breaths, and restart the pre-shot routine calmly.",
            "output-warning"
        )

    if pre_swing_high_hr and (rhythm_alert or body_motion_alert):
        return (
            "Swing execution may be affected by high pre-swing activation",
            f"Local heart rate before the swing sequence appears elevated "
            f"({pre_swing_hr_mean:.0f} bpm on average before the detected swings), together with changes in rhythm or body–club coordination. "
            "Advice: pause briefly, breathe slowly, relax the grip and shoulders, and repeat the pre-shot routine before executing the next swing.",
            "output-critical"
        )

    # ============================================================
    # EXISTING LOGIC
    # ============================================================

    if body_motion_alert and not rhythm_alert and not physiology_alert:
        return (
            "Body motion issue likely",
            "The sacral IMU indicates altered body–club coordination, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if rhythm_alert and not body_motion_alert and not physiology_alert:
        return (
            "Rhythm issue likely",
            "The club IMU indicates altered swing rhythm, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if rhythm_alert and body_motion_alert and not physiology_alert:
        return (
            "Technical issue likely",
            "Both swing rhythm and body–club coordination appear altered, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if not rhythm_alert and not body_motion_alert and physiology_alert:
        return (
            "Physiological influence to monitor",
            "HR or HRV appears altered, but rhythm and body–club coordination remain stable.",
            "output-warning"
        )

    if rhythm_alert and not body_motion_alert and physiology_alert:
        return (
            "Rhythm affected by physiological state",
            "Swing rhythm appears altered together with an altered physiological state.",
            "output-critical"
        )

    if body_motion_alert and not rhythm_alert and physiology_alert:
        return (
            "Body motion affected by physiological state",
            "Body–club coordination appears altered together with an altered physiological state.",
            "output-critical"
        )

    if rhythm_alert and body_motion_alert and physiology_alert:
        return (
            "Combined effect likely",
            "Swing rhythm, body–club coordination and physiological state are all altered.",
            "output-critical"
        )

    return (
        "Stable execution",
        "Rhythm, body–club coordination and physiological state appear stable in this session.",
        "output-stable"
    )

    # ============================================================
    # NEW: high HR advice
    # ============================================================

    if high_hr_alert and not rhythm_alert and not body_motion_alert:
        return (
            "High physiological activation before the shot",
            "Heart rate appears elevated before or during the swing sequence, while swing rhythm and body–club coordination remain stable. "
            "Advice: take a short pause before the next shot, relax the shoulders, use slow breathing, and restart the routine only when you feel settled.",
            "output-warning"
        )

    if high_hr_alert and (rhythm_alert or body_motion_alert):
        return (
            "Swing execution may be affected by high activation",
            "Heart rate appears elevated together with changes in rhythm or body–club coordination. "
            "Advice: before the next shot, take 2–3 slow breaths, relax the grip and shoulders, and repeat the pre-shot routine calmly.",
            "output-critical"
        )

    # ============================================================
    # Existing logic
    # ============================================================

    if body_motion_alert and not rhythm_alert and not physiology_alert:
        return (
            "Body motion issue likely",
            "The sacral IMU indicates altered body–club coordination, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if rhythm_alert and not body_motion_alert and not physiology_alert:
        return (
            "Rhythm issue likely",
            "The club IMU indicates altered swing rhythm, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if rhythm_alert and body_motion_alert and not physiology_alert:
        return (
            "Technical issue likely",
            "Both swing rhythm and body–club coordination appear altered, while HR and HRV remain in a normal range.",
            "output-warning"
        )

    if not rhythm_alert and not body_motion_alert and physiology_alert:
        return (
            "Physiological influence to monitor",
            "HR or HRV appears altered, but rhythm and body–club coordination remain stable. "
            "Advice: relax, breathe slowly, and allow a short recovery period before the next swing.",
            "output-warning"
        )

    if rhythm_alert and not body_motion_alert and physiology_alert:
        return (
            "Rhythm affected by physiological state",
            "Swing rhythm appears altered together with an altered physiological state. "
            "Advice: slow down the routine, take controlled breaths, and focus on a consistent tempo before restarting.",
            "output-critical"
        )

    if body_motion_alert and not rhythm_alert and physiology_alert:
        return (
            "Body motion affected by physiological state",
            "Body–club coordination appears altered together with an altered physiological state. "
            "Advice: pause briefly, relax the body, breathe, and repeat the pre-shot routine with controlled movement.",
            "output-critical"
        )

    if rhythm_alert and body_motion_alert and physiology_alert:
        return (
            "Combined effect likely",
            "Swing rhythm, body–club coordination and physiological state are all altered. "
            "Advice: stop the sequence, recover breathing, relax the grip and shoulders, and restart only when the athlete feels stable.",
            "output-critical"
        )

    return (
        "Stable execution",
        "Rhythm, body–club coordination and physiological state appear stable in this session.",
        "output-stable"
    )
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
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70)
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
        margin=dict(l=20, r=20, t=60, b=60)
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
        xaxis=dict(title="Swing Number"),
        yaxis=dict(title="Tempo Ratio", side="left"),
        yaxis2=dict(title="Δt A5-A4 [s]", overlaying="y", side="right"),
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
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
            # Full 3-second pre-swing window
            fig.add_vrect(
                x0=t_swing - pre_window_s,
                x1=t_swing,
                fillcolor="rgba(255, 193, 7, 0.10)",
                line_width=0,
                layer="below"
            )

            # Close pre-swing window used for the advice
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
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70)
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

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Session Analysis", "Metrics Explanation"]
)

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

if page == "Home":

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

    st.markdown('<div class="section-title">Validation Metrics by Session</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Session 01 — Baseline Training</div>
            <div class="card-text">
                First reference session used to understand the athlete’s normal rhythm and coordination.
            </div>
            <div class="big-number">1</div>
            <p>Rhythm · Repeatability · HR/HRV</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Session 02 — Repeated Swings</div>
            <div class="card-text">
                Evaluation of rhythm consistency across multiple shots and body–club timing.
            </div>
            <div class="big-number">2</div>
            <p>Tempo Ratio · Δt · CV</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Session 03 — Physiological Context</div>
            <div class="card-text">
                Analysis of whether rhythm and repeatability change under different physiological states.
            </div>
            <div class="big-number">3</div>
            <p>HR · HRV · Interpretation</p>
        </div>
        """, unsafe_allow_html=True)

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

elif page == "Session Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Session analysis</div>
        <div class="hero-title">Integrated Swing Interpretation</div>
        <div class="hero-subtitle">
            Upload club and sacral IMU data to calculate Tempo Ratio, body–club timing delay and repeatability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_role == "Coach" and not st.session_state.selected_athlete_name:
        st.warning("Please select an athlete from the Home page before opening the session analysis.")
        st.stop()

    st.markdown(f"""
    <div class="selected-athlete-box">
        <b>Current athlete:</b> {st.session_state.selected_athlete_name}
    </div>
    """, unsafe_allow_html=True)

    session_name = st.selectbox("Select session", ["Session 1", "Session 2", "Session 3"])

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

    st.markdown("---")

    if club_file is not None and sacral_file is not None:

        club_raw = read_uploaded_file(club_file)
        sacral_raw = read_uploaded_file(sacral_file)

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
                if hr_file.name.lower().endswith(".bin"):
                    hr_df = process_ecg_bin_to_hr(hr_file)
                else:
                    hr_raw = read_uploaded_file(hr_file)
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
                pre_swing_hr_mean=pre_swing_hr_mean
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
                    use_container_width=True
                )

            st.plotly_chart(
                plot_tempo_ratio_vs_delta_t(swing_analysis_df),
                use_container_width=True
            )

            st.markdown(f"""
            <div class="formula-box">
                <b>CV(Tempo Ratio):</b> {cv_tempo_ratio:.2f}%<br>
                <b>CV(Δt body–club):</b> {cv_delta_t:.2f}%<br>
                <b>Interpretation:</b> a lower CV indicates higher repeatability across the five swings.
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # INTEGRATED INTERPRETATION
            # ============================================================

            st.markdown("---")
            st.markdown('<div class="section-title">Integrated Interpretation</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="{output_class}">
                <h2>{output}</h2>
                <p>{explanation}</p>
                <p>
                <b>Interpretation logic:</b> the app separates club rhythm, sacral body–club coordination and HR/HRV status.
                This allows the system to identify whether the alteration is mainly related to swing rhythm, body motion,
                physiological state, or a combined effect.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # ACCELERATION SIGNALS
            # ============================================================

            st.markdown("---")
            st.markdown('<div class="section-title">Acceleration Signals</div>', unsafe_allow_html=True)

            p1, p2 = st.columns(2)

            with p1:
                st.plotly_chart(
                    plot_club_events_five_swings(club_peak_df, swing_analysis_df),
                    use_container_width=True
                )

            with p2:
                st.plotly_chart(
                    plot_acceleration(sacral_peak_df, "Sacral IMU — body motion signal"),
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown('<div class="section-title">Body–Club Coordination</div>', unsafe_allow_html=True)

            st.plotly_chart(
                plot_combined_club_sacral(club_peak_df, sacral_peak_df),
                use_container_width=True
            )

            st.markdown(f"""
            <div class="formula-box">
                <b>Δt = t<sub>peak,sacral</sub> − t<sub>peak,club</sub></b><br>
                Mean t<sub>peak,club</sub> = {t_peak_club:.3f} s<br>
                Mean t<sub>peak,sacral</sub> = {t_peak_sacral:.3f} s<br>
                Mean Δt = {delta_t:.3f} s
            </div>
            """, unsafe_allow_html=True)

            if hr_df is not None:
                st.markdown("---")
                st.markdown('<div class="section-title">Heart Rate Trend During Session</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="formula-box">
                    The ECG .bin file is processed independently from the IMU signals.
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

elif page == "Metrics Explanation":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Validation metrics</div>
        <div class="hero-title">How the App Interprets the Swing</div>
        <div class="hero-subtitle">
            The prototype connects swing rhythm, body–club coordination and physiological condition.
        </div>
    </div>
    """, unsafe_allow_html=True)

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