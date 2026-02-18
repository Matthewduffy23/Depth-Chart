"""
Squad Depth Chart
=================
• Montserrat font · #0a0f1c background
• Toggle: minutes / goals / assists / role scores / 1920×1080 Canva layout
• Role scores: percentile vs same league + same position pool, red→orange→gold→green
• Best role bold, all roles listed for starters; best role only for depth
• LAMF→LW, RAMF→RW, AMF→AM  |  LWB/RWB slots only accept LB/LWB/RB/RWB
• RCB/RCMF fill right slots, LCB/LCMF fill left slots
• Fallback to next listed positions in order
• On Loan column → green
• Move / Remove / Add player
• Sidebar CSV upload — swap any time

pip install streamlit pandas numpy
streamlit run app.py
"""

import re
import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Squad Depth Chart", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;background:#0a0f1c!important;color:#fff!important}
.stApp{background:#0a0f1c!important}
section[data-testid="stSidebar"]{background:#060a14!important;border-right:1px solid #0d1220!important}
section[data-testid="stSidebar"] *{color:#fff!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] select{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important}
.stSelectbox>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important}
div[data-baseweb="select"]*{background:#0d1424!important;color:#fff!important}
div[data-baseweb="popover"]*{background:#0d1424!important;color:#fff!important}
.stTextInput>div>div>input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important}
.stButton>button{background:#fff!important;color:#0a0f1c!important;font-weight:700!important;
  letter-spacing:.06em!important;text-transform:uppercase!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;border-radius:2px!important}
.stButton>button:hover{background:#e0e0e0!important}
label{color:#2d3a52!important;font-size:9px!important;letter-spacing:.14em!important;text-transform:uppercase!important}
h1,h2,h3{color:#fff!important;font-family:'Montserrat',sans-serif!important}
footer{display:none!important}
hr{border-color:#0d1220!important}
.streamlit-expanderHeader{background:#0d1424!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)

# ── Role Buckets ───────────────────────────────────────────────────────────────
# Compared within same league + same position pool (see POS_POOL_MAP)

ROLE_BUCKETS: dict[str, dict] = {
    "GK": {
        "Shot Stopper GK": {"metrics": {
            "Prevented goals per 90": 3,
            "Save rate, %": 1,
        }},
        "Ball Playing GK": {"metrics": {
            "Passes per 90": 1,
            "Accurate passes, %": 3,
            "Accurate long passes, %": 2,
        }},
        "Sweeper GK": {"metrics": {
            "Exits per 90": 1,
        }},
    },
    "CB": {
        "Ball Playing CB": {"metrics": {
            "Passes per 90": 2, "Accurate passes, %": 2,
            "Forward passes per 90": 2, "Accurate forward passes, %": 2,
            "Progressive passes per 90": 2, "Progressive runs per 90": 1.5,
            "Dribbles per 90": 1.5, "Accurate long passes, %": 1,
            "Passes to final third per 90": 1.5,
        }},
        "Wide CB": {"metrics": {
            "Defensive duels per 90": 1.5, "Defensive duels won, %": 2,
            "Dribbles per 90": 2, "Forward passes per 90": 1,
            "Progressive passes per 90": 1, "Progressive runs per 90": 2,
        }},
        "Box Defender": {"metrics": {
            "Aerial duels per 90": 1, "Aerial duels won, %": 3,
            "PAdj Interceptions": 2, "Shots blocked per 90": 1,
            "Defensive duels won, %": 4,
        }},
    },
    "FB": {
        "Build Up FB": {"metrics": {
            "Passes per 90": 2, "Accurate passes, %": 1.5,
            "Forward passes per 90": 2, "Accurate forward passes, %": 2,
            "Progressive passes per 90": 2.5, "Progressive runs per 90": 2,
            "Dribbles per 90": 2, "Passes to final third per 90": 2, "xA per 90": 1,
        }},
        "Attacking FB": {"metrics": {
            "Crosses per 90": 2, "Dribbles per 90": 3.5, "Accelerations per 90": 1,
            "Successful dribbles, %": 1, "Touches in box per 90": 2,
            "Progressive runs per 90": 3, "Passes to penalty area per 90": 2, "xA per 90": 3,
        }},
        "Defensive FB": {"metrics": {
            "Aerial duels per 90": 1, "Aerial duels won, %": 1.5,
            "Defensive duels per 90": 2, "PAdj Interceptions": 3,
            "Shots blocked per 90": 1, "Defensive duels won, %": 3.5,
        }},
    },
    "CM": {
        "Deep Playmaker CM": {"metrics": {
            "Passes per 90": 1, "Accurate passes, %": 1,
            "Forward passes per 90": 2, "Accurate forward passes, %": 1.5,
            "Progressive passes per 90": 3, "Passes to final third per 90": 2.5,
            "Accurate long passes, %": 1,
        }},
        "Advanced Playmaker CM": {"metrics": {
            "Deep completions per 90": 1.5, "Smart passes per 90": 2,
            "xA per 90": 4, "Passes to penalty area per 90": 2,
        }},
        "Defensive CM": {"metrics": {
            "Defensive duels per 90": 4, "Defensive duels won, %": 4,
            "PAdj Interceptions": 3, "Aerial duels per 90": 0.5, "Aerial duels won, %": 1,
        }},
        "Ball Carrying CM": {"metrics": {
            "Dribbles per 90": 4, "Successful dribbles, %": 2,
            "Progressive runs per 90": 3, "Accelerations per 90": 3,
        }},
    },
    "ATT": {
        "Playmaker ATT": {"metrics": {
            "Passes per 90": 2, "xA per 90": 3, "Key passes per 90": 1,
            "Deep completions per 90": 1.5, "Smart passes per 90": 1.5,
            "Passes to penalty area per 90": 2,
        }},
        "Goal Threat ATT": {"metrics": {
            "xG per 90": 3, "Non-penalty goals per 90": 3,
            "Shots per 90": 2, "Touches in box per 90": 2,
        }},
        "Ball Carrier ATT": {"metrics": {
            "Dribbles per 90": 4, "Successful dribbles, %": 2,
            "Progressive runs per 90": 3, "Accelerations per 90": 3,
        }},
    },
    "CF": {
        "Target Man CF": {"metrics": {
            "Aerial duels per 90": 3, "Aerial duels won, %": 5,
        }},
        "Goal Threat CF": {"metrics": {
            "Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3,
            "Touches in box per 90": 1, "Shots on target, %": 0.5,
        }},
        "Link Up CF": {"metrics": {
            "Passes per 90": 2, "Passes to penalty area per 90": 1.5,
            "Deep completions per 90": 1, "Smart passes per 90": 1.5,
            "Accurate passes, %": 1.5, "Key passes per 90": 1,
            "Dribbles per 90": 2, "Successful dribbles, %": 1,
            "Progressive runs per 90": 2, "xA per 90": 3,
        }},
    },
}

# Which raw first-position tokens map to which role bucket
ROLE_KEY_MAP: dict[str, str] = {
    "GK":   "GK",
    "CB":   "CB",  "LCB":  "CB",  "RCB":  "CB",
    "LB":   "FB",  "RB":   "FB",  "LWB":  "FB",  "RWB":  "FB",
    "DMF":  "CM",  "LDMF": "CM",  "RDMF": "CM",
    "LCMF":"CM",  "RCMF":"CM",
    "AMF":  "ATT",
    "LAMF": "ATT", "LW":   "ATT", "LWF":  "ATT",
    "RAMF": "ATT", "RW":   "ATT", "RWF":  "ATT",
    "CF":   "CF",
}

# Which raw position tokens are compared together for percentiles
POS_POOL_MAP: dict[str, list[str]] = {
    "GK":  ["GK"],
    "CB":  ["CB", "LCB", "RCB"],
    "FB":  ["LB", "RB", "LWB", "RWB"],
    "CM":  ["DMF", "LDMF", "RDMF", "LCMF", "RCMF"],
    "ATT": ["AMF", "LAMF", "RAMF", "LW", "LWF", "RW", "RWF"],
    "CF":  ["CF"],
}

# ── Position → canonical slot group ───────────────────────────────────────────
CANONICAL: dict[str, str] = {
    "GK":   "GK",
    "CB":   "CB",   "LCB":  "LCB",  "RCB":  "RCB",
    "LB":   "LB",   "LWB":  "LWB",
    "RB":   "RB",   "RWB":  "RWB",
    "DMF":  "DM",   "LDMF": "DM",   "RDMF": "DM",
    "LCMF":"CM",   "RCMF":"CM",
    "AMF":  "AM",
    "LAMF": "LW",  "LW":   "LW",   "LWF":  "LW",
    "RAMF": "RW",  "RW":   "RW",   "RWF":  "RW",
    "CF":   "ST",
}

# Side preference for correct slot placement
SIDE_PREF: dict[str, str] = {
    "RCB":"R","RCMF":"R","RDMF":"R","RB":"R","RWB":"R","RW":"R","RWF":"R","RAMF":"R",
    "LCB":"L","LCMF":"L","LDMF":"L","LB":"L","LWB":"L","LW":"L","LWF":"L","LAMF":"L",
}

def _tok(pos: str) -> str:
    return str(pos).split(",")[0].strip().upper()

def _canon(pos: str) -> str:
    return CANONICAL.get(_tok(pos), "CM")

def _side(pos: str) -> str:
    return SIDE_PREF.get(_tok(pos), "N")

def _role_key(pos: str) -> str:
    return ROLE_KEY_MAP.get(_tok(pos), "ATT")

def _all_canons(pos: str) -> list[str]:
    return [CANONICAL.get(t.strip().upper(), "CM") for t in str(pos).split(",")]

# ── Formations ─────────────────────────────────────────────────────────────────
# wb_only=True → ONLY LB/LWB or RB/RWB raw tokens allowed (no LW/RW/etc)

FORMATIONS: dict[str, list[dict]] = {
    "4-2-3-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],              "side":"N"},
        {"id":"LW",  "label":"LW",  "x":13,"y":25, "accepts":["LW","AM"],         "side":"L"},
        {"id":"AM",  "label":"AM",  "x":50,"y":24, "accepts":["AM"],              "side":"N"},
        {"id":"RW",  "label":"RW",  "x":87,"y":25, "accepts":["RW","AM"],         "side":"R"},
        {"id":"DM1", "label":"DM",  "x":35,"y":43, "accepts":["DM","CM"],         "side":"L"},
        {"id":"DM2", "label":"DM",  "x":65,"y":43, "accepts":["DM","CM"],         "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],        "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"],  "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"],  "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],        "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
    "4-3-3": [
        # 4-3-3 middle: DM (left) · CM (centre) · AM (right)
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],              "side":"N"},
        {"id":"LW",  "label":"LW",  "x":14,"y":16, "accepts":["LW"],              "side":"L"},
        {"id":"RW",  "label":"RW",  "x":86,"y":16, "accepts":["RW"],              "side":"R"},
        {"id":"DM",  "label":"DM",  "x":22,"y":36, "accepts":["DM","CM"],         "side":"L"},
        {"id":"CM",  "label":"CM",  "x":50,"y":32, "accepts":["CM","DM","AM"],    "side":"N"},
        {"id":"AM",  "label":"AM",  "x":78,"y":36, "accepts":["AM","CM"],         "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],        "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"],  "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"],  "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],        "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
    "4-4-2": [
        {"id":"ST1", "label":"ST",  "x":35,"y":9,  "accepts":["ST"],              "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":9,  "accepts":["ST"],              "side":"R"},
        {"id":"LW",  "label":"LW",  "x":9, "y":34, "accepts":["LW","AM"],         "side":"L"},
        {"id":"CM1", "label":"CM",  "x":34,"y":38, "accepts":["CM","DM","AM"],    "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":38, "accepts":["CM","DM","AM"],    "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":34, "accepts":["RW","AM"],         "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],        "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"],  "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"],  "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],        "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
    "3-5-2": [
        {"id":"ST1", "label":"ST",  "x":35,"y":9,  "accepts":["ST"],              "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":9,  "accepts":["ST"],              "side":"R"},
        {"id":"LWB", "label":"LWB", "x":9, "y":34, "accepts":["LWB","LB"],        "side":"L","wb_only":True},
        {"id":"CM1", "label":"CM",  "x":28,"y":40, "accepts":["CM","AM"],         "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":36, "accepts":["DM","CM"],         "side":"N"},
        {"id":"CM2", "label":"CM",  "x":72,"y":40, "accepts":["CM","AM"],         "side":"R"},
        {"id":"RWB", "label":"RWB", "x":91,"y":34, "accepts":["RWB","RB"],        "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":64, "accepts":["LCB","CB"],        "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":67, "accepts":["CB","LCB","RCB"],  "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":64, "accepts":["RCB","CB"],        "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
    "3-4-1-2": [
        # LW/RW → AM slot; if still none → depth (not sent to wingback slots)
        {"id":"ST1", "label":"ST",  "x":35,"y":8,  "accepts":["ST"],              "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":8,  "accepts":["ST"],              "side":"R"},
        {"id":"AM",  "label":"AM",  "x":50,"y":20, "accepts":["AM","LW","RW"],    "side":"N"},
        {"id":"LWB", "label":"LWB", "x":9, "y":36, "accepts":["LWB","LB"],        "side":"L","wb_only":True},
        {"id":"CM1", "label":"CM",  "x":34,"y":40, "accepts":["CM","DM"],         "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":40, "accepts":["CM","DM"],         "side":"R"},
        {"id":"RWB", "label":"RWB", "x":91,"y":36, "accepts":["RWB","RB"],        "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":62, "accepts":["LCB","CB"],        "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":65, "accepts":["CB","LCB","RCB"],  "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":62, "accepts":["RCB","CB"],        "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":82, "accepts":["GK"],              "side":"N"},
    ],
    "4-5-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],              "side":"N"},
        {"id":"LW",  "label":"LW",  "x":9, "y":25, "accepts":["LW"],              "side":"L"},
        {"id":"CM1", "label":"CM",  "x":30,"y":33, "accepts":["CM","DM"],         "side":"L"},
        {"id":"AM",  "label":"AM",  "x":50,"y":25, "accepts":["AM"],              "side":"N"},
        {"id":"CM2", "label":"CM",  "x":70,"y":33, "accepts":["CM","DM"],         "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":25, "accepts":["RW"],              "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],        "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"],  "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"],  "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],        "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
    "4-1-4-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],              "side":"N"},
        {"id":"LW",  "label":"LW",  "x":9, "y":26, "accepts":["LW"],              "side":"L"},
        {"id":"CM1", "label":"CM",  "x":31,"y":29, "accepts":["CM","AM"],         "side":"L"},
        {"id":"CM2", "label":"CM",  "x":69,"y":29, "accepts":["CM","AM"],         "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":26, "accepts":["RW"],              "side":"R"},
        {"id":"DM",  "label":"DM",  "x":50,"y":44, "accepts":["DM","CM"],         "side":"N"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],        "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"],  "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"],  "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],        "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],              "side":"N"},
    ],
}

# Fill order: defence first, midfield, attack (so depth goes to more advanced slots last)
PITCH_ORDER = ["GK","LCB","CB","RCB","LB","RB","LWB","RWB","DM","CM","AM","LW","RW","ST"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def contract_years(s) -> int:
    s = str(s or "").strip()
    if s in ("","nan","NaT"): return -1
    m = re.search(r"(20\d{2})", s)
    return max(0, int(m.group(1)) - date.today().year) if m else -1

def is_loan(p: dict) -> bool:
    for k in ("On loan","On Loan","on_loan","Loan","loan","On loan?"):
        if k in p and str(p[k]).strip().lower() in ("yes","y","true","1","on loan"):
            return True
    return False

def player_css_color(yrs: int, loan: bool) -> str:
    if loan:   return "#22c55e"
    if yrs == 0: return "#ef4444"
    if yrs == 1: return "#f59e0b"
    return "#ffffff"

def score_to_color(v: float) -> str:
    """0-100 → red → orange → gold → green"""
    if np.isnan(v): return "#4b5563"
    v = max(0.0, min(100.0, float(v)))
    if v <= 50:
        t = v / 50.0
        r = int(239 + (234 - 239) * t)
        g = int(68  + (179 - 68)  * t)
        b = int(68  + (8   - 68)  * t)
    else:
        t = (v - 50) / 50.0
        r = int(234 + (34  - 234) * t)
        g = int(179 + (197 - 179) * t)
        b = int(8   + (94  - 8)   * t)
    return f"rgb({r},{g},{b})"

# ── Role score computation ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_role_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute weighted-average percentile scores for every role
    in their bucket. Percentile is against same league + same position pool,
    min 200 minutes played to be in reference pool.
    Adds columns:  _rs_<Role Name>
    """
    df = df.copy()

    # make all stat columns numeric
    skip = {"Player","League","Team","Position","Age","Market value",
            "Contract expires","Matches played","Minutes played","Goals",
            "Assists","xG","xA","Birth country","Foot","Height","_ftok","_key"}
    for c in df.columns:
        if c not in skip and not c.startswith("On ") and "loan" not in c.lower():
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for rk, pool_positions in POS_POOL_MAP.items():
        buckets = ROLE_BUCKETS.get(rk, {})
        for role_name, spec in buckets.items():
            col_out = f"_rs_{role_name}"
            df[col_out] = np.nan
            metrics = spec.get("metrics", {})

            for league in df["League"].unique():
                mask = (
                    (df["League"] == league) &
                    (df["_ftok"].isin(pool_positions)) &
                    (df["Minutes played"] >= 200)
                )
                pool = df[mask]
                if pool.empty:
                    continue

                # percentile rank per metric within pool
                pcts: dict[str, pd.Series] = {}
                for met in metrics:
                    if met in pool.columns:
                        s = pd.to_numeric(pool[met], errors="coerce")
                        pcts[met] = s.rank(pct=True, method="average") * 100.0

                for idx in pool.index:
                    vals, wts = [], []
                    for met, w in metrics.items():
                        if met in pcts and idx in pcts[met].index:
                            pv = pcts[met].loc[idx]
                            if not np.isnan(pv):
                                vals.append(float(pv))
                                wts.append(float(w))
                    if vals:
                        df.at[idx, col_out] = float(np.average(vals, weights=wts))

    return df

# ── Player assignment ──────────────────────────────────────────────────────────

def assign_players(players: list[dict], formation_key: str) -> tuple[dict, list]:
    """
    Returns (slot_map, depth).
    slot_map[slot_id] = [starter, depth1, depth2, ...]
    depth = players with no slot match at all
    """
    slots = FORMATIONS.get(formation_key, FORMATIONS["4-2-3-1"])

    # group slots by label for multi-slot positions (CB1+CB2, DM1+DM2 etc)
    by_label: dict[str, list] = {}
    for s in slots:
        by_label.setdefault(s["label"], []).append(s)

    assigned: set = set()
    slot_map: dict[str, list] = {s["id"]: [] for s in slots}

    def wb_ok(p: dict, slot: dict) -> bool:
        """wb_only slots strictly check raw first token."""
        tok = _tok(p.get("Position", ""))
        return tok in {"LB","LWB","RB","RWB"} and _canon(p.get("Position","")) in slot["accepts"]

    def fits_primary(p: dict, slot: dict) -> bool:
        if slot.get("wb_only"):
            return wb_ok(p, slot)
        return _canon(p.get("Position","")) in slot["accepts"]

    def fits_secondary(p: dict, slot: dict) -> bool:
        if slot.get("wb_only"):
            return False   # never use secondary positions for WB slots
        for t in str(p.get("Position","")).split(",")[1:]:
            if CANONICAL.get(t.strip().upper(),"CM") in slot["accepts"]:
                return True
        return False

    def side_score(p: dict, slot_side: str) -> int:
        """0 = perfect, 1 = neutral, 2 = wrong side"""
        ps = _side(p.get("Position",""))
        if slot_side == "N" or ps == "N": return 1
        return 0 if ps == slot_side else 2

    for label in PITCH_ORDER:
        if label not in by_label:
            continue
        slot_list = by_label[label]

        # primary match
        matched = [p for p in players
                   if p["_key"] not in assigned
                   and any(fits_primary(p, s) for s in slot_list)]

        # secondary fallback (no wb_only)
        if not matched:
            matched = [p for p in players
                       if p["_key"] not in assigned
                       and any(fits_secondary(p, s) for s in slot_list)]

        matched.sort(key=lambda p: -float(p.get("Minutes played") or 0))
        for p in matched:
            assigned.add(p["_key"])

        n = len(slot_list)
        if n == 1:
            slot_map[slot_list[0]["id"]] = matched
        else:
            # assign one starter per slot respecting side preference,
            # then all depth goes to the "best match" slot
            ordered = sorted(slot_list, key=lambda s: {"L":0,"N":1,"R":2}[s["side"]])
            starters: list = []
            used_in_starters: set = set()

            for sl in ordered:
                best = None
                best_score = 99
                for p in matched:
                    if id(p) in used_in_starters:
                        continue
                    sc = side_score(p, sl["side"])
                    if sc < best_score:
                        best_score = sc
                        best = p
                if best:
                    starters.append((sl["id"], best))
                    used_in_starters.add(id(best))

            depth_for = [p for p in matched if id(p) not in used_in_starters]

            for sl in slot_list:
                slot_map[sl["id"]] = []

            for sid, p in starters:
                slot_map[sid].append(p)

            # depth goes to first slot
            slot_map[slot_list[0]["id"]].extend(depth_for)

    depth = [p for p in players if p["_key"] not in assigned]
    depth.sort(key=lambda p: -float(p.get("Minutes played") or 0))
    return slot_map, depth

# ── Score HTML helpers ─────────────────────────────────────────────────────────

def all_roles_html(player: dict, df_sc: pd.DataFrame,
                   fs: str = "8px") -> str:
    """All role scores for a position — best bold + coloured, others grey."""
    if df_sc is None or df_sc.empty: return ""
    name = player.get("Player","")
    rows = df_sc[df_sc["Player"] == name]
    if rows.empty: return ""
    row = rows.iloc[0]
    rk = _role_key(player.get("Position",""))
    buckets = ROLE_BUCKETS.get(rk, {})
    scores: dict[str, float] = {}
    for rn in buckets:
        col = f"_rs_{rn}"
        if col in row.index:
            v = row[col]
            if isinstance(v, (int, float)) and not np.isnan(float(v)):
                scores[rn] = float(v)
    if not scores: return ""
    best = max(scores, key=scores.get)
    lines = []
    for rn, sc in sorted(scores.items(), key=lambda x: -x[1]):
        sc_col = score_to_color(sc)
        is_b = rn == best
        fw  = "700" if is_b else "400"
        tc  = sc_col if is_b else "#6b7280"
        lines.append(
            f'<div style="display:flex;justify-content:space-between;gap:5px;'
            f'font-size:{fs};line-height:1.5;">'
            f'<span style="color:{tc};font-weight:{fw};">{rn}</span>'
            f'<span style="color:{sc_col};font-weight:{fw};">{int(sc)}</span></div>'
        )
    return (
        f'<div style="margin-top:3px;padding-top:3px;border-top:1px solid #111827;">'
        + "".join(lines) + "</div>"
    )

def best_role_html(player: dict, df_sc: pd.DataFrame, fs: str = "8px") -> str:
    """Just the best role name + score for depth players."""
    if df_sc is None or df_sc.empty: return ""
    name = player.get("Player","")
    rows = df_sc[df_sc["Player"] == name]
    if rows.empty: return ""
    row = rows.iloc[0]
    rk = _role_key(player.get("Position",""))
    buckets = ROLE_BUCKETS.get(rk, {})
    scores: dict[str, float] = {}
    for rn in buckets:
        col = f"_rs_{rn}"
        if col in row.index:
            v = row[col]
            if isinstance(v, (int, float)) and not np.isnan(float(v)):
                scores[rn] = float(v)
    if not scores: return ""
    best = max(scores, key=scores.get)
    sc = scores[best]
    sc_col = score_to_color(sc)
    return (
        f'<div style="font-size:{fs};line-height:1.5;margin-top:2px;">'
        f'<span style="color:#6b7280;font-weight:400;">{best} </span>'
        f'<span style="color:{sc_col};font-weight:700;">{int(sc)}</span></div>'
    )

# ── Pitch renderer ─────────────────────────────────────────────────────────────

def render_pitch(
    team: str, league: str, formation: str,
    slots: list, slot_map: dict, depth: list,
    df_sc: pd.DataFrame,
    show_mins: bool, show_goals: bool, show_assists: bool,
    show_roles: bool, canva: bool,
) -> str:

    aspect  = "56.25%" if canva else "140%"
    nm_sz   = "14px"   if canva else "12px"
    badge_sz= "10px"   if canva else "8.5px"
    stat_sz = "9px"    if canva else "7px"
    role_sz = "8.5px"  if canva else "7.5px"
    dep_nm  = "11px"   if canva else "10px"
    dep_rs  = "9px"    if canva else "8px"

    nodes = ""
    for slot in slots:
        ps = slot_map.get(slot["id"], [])
        badge = (
            f'<div style="display:inline-block;padding:2px 9px;'
            f'border:2px solid #ef4444;color:#ef4444;font-size:{badge_sz};'
            f'font-weight:900;letter-spacing:.12em;margin-bottom:4px;'
            f'background:rgba(10,15,28,.92);">'
            f'{slot["label"]}</div>'
        )
        rows = ""
        for i, p in enumerate(ps):
            yrs    = contract_years(p.get("Contract expires",""))
            yr_str = f"+{yrs}" if yrs >= 0 else "+?"
            loan   = is_loan(p)
            col    = player_css_color(yrs, loan)
            fw     = "800" if i == 0 else "500"
            tag    = " (L)" if loan else ""

            stat_parts = []
            if show_mins:
                stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}′")
            if show_goals:
                g = float(p.get("Goals") or 0)
                if g > 0: stat_parts.append(f"{int(g)}⚽")
            if show_assists:
                a = float(p.get("Assists") or 0)
                if a > 0: stat_parts.append(f"{int(a)}🅰")
            stat_html = (
                f'<div style="color:#374151;font-size:{stat_sz};line-height:1.2;">'
                f'{" ".join(stat_parts)}</div>'
            ) if stat_parts else ""

            rs_html = (
                all_roles_html(p, df_sc, role_sz) if (i == 0 and show_roles) else
                best_role_html(p, df_sc, role_sz)  if (i > 0  and show_roles) else ""
            )

            rows += (
                f'<div style="color:{col};font-size:{nm_sz};line-height:1.5;'
                f'font-weight:{fw};white-space:nowrap;'
                f'text-shadow:0 1px 6px rgba(0,0,0,.9);">'
                f'{p["Player"]} {yr_str}{tag}</div>'
                f'{stat_html}{rs_html}'
            )

        if not ps:
            rows = f'<div style="color:#1f2937;font-size:{stat_sz};">—</div>'

        nodes += (
            f'<div style="position:absolute;left:{slot["x"]}%;top:{slot["y"]}%;'
            f'transform:translate(-50%,-50%);text-align:center;min-width:92px;z-index:10;">'
            f'{badge}<div>{rows}</div></div>'
        )

    # ── Depth bar ──────────────────────────────────────────────────────────────
    depth_html = ""
    if depth:
        cards = ""
        for p in depth:
            yrs    = contract_years(p.get("Contract expires",""))
            yr_str = f"+{yrs}" if yrs >= 0 else "+?"
            loan   = is_loan(p)
            col    = player_css_color(yrs, loan)
            pos_tag = str(p.get("Position","")).split(",")[0].strip()
            br_html = best_role_html(p, df_sc, dep_rs) if show_roles else ""
            cards += (
                f'<div style="background:#0d1220;border:1px solid #111827;'
                f'padding:5px 9px;min-width:100px;text-align:center;'
                f'flex-shrink:0;">'
                f'<div style="color:{col};font-size:{dep_nm};font-weight:700;">'
                f'{p["Player"]} {yr_str}</div>'
                f'<div style="color:#374151;font-size:7px;">{pos_tag}</div>'
                f'{br_html}</div>'
            )
        depth_html = (
            f'<div style="margin-top:12px;border-top:1px solid #111827;padding-top:8px;">'
            f'<div style="font-size:9px;font-weight:800;letter-spacing:.18em;color:#374151;'
            f'margin-bottom:6px;text-align:center;">DEPTH</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">'
            f'{cards}</div></div>'
        )

    # legend
    legend_stats = ""
    if show_mins:    legend_stats += " · ′=mins"
    if show_goals:   legend_stats += " · ⚽=goals"
    if show_assists: legend_stats += " · 🅰=assists"

    return f"""
<div style="font-family:Montserrat,sans-serif;color:#fff;background:#0a0f1c;padding:0 4px 12px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
    <div style="font-size:10px;color:#1f2937;letter-spacing:.12em;text-transform:uppercase;">{league}</div>
    <div style="font-weight:900;font-size:20px;letter-spacing:.05em;text-transform:uppercase;">{team}</div>
    <div style="font-size:10px;color:#1f2937;">{formation}</div>
  </div>

  <div style="position:relative;background:#0a0f1c;padding-bottom:{aspect};overflow:hidden;border:1px solid #0d1220;">
    <svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"
         viewBox="0 0 100 140" preserveAspectRatio="none">
      <rect  x="2"   y="2"     width="96" height="136" fill="none" stroke="#111827" stroke-width=".8"/>
      <line  x1="2"  y1="70"   x2="98"   y2="70"      stroke="#111827" stroke-width=".5"/>
      <circle cx="50" cy="70"  r="10"                  fill="none" stroke="#111827" stroke-width=".5"/>
      <circle cx="50" cy="70"  r=".9"                  fill="#111827"/>
      <rect  x="22"  y="2"     width="56" height="18"  fill="none" stroke="#111827" stroke-width=".45"/>
      <rect  x="36"  y="2"     width="28" height="7"   fill="none" stroke="#111827" stroke-width=".35"/>
      <rect  x="42.5" y=".3"   width="15" height="2.5" fill="none" stroke="#111827" stroke-width=".4"/>
      <circle cx="50" cy="14"  r=".65"                 fill="#111827"/>
      <rect  x="22"  y="120"   width="56" height="18"  fill="none" stroke="#111827" stroke-width=".45"/>
      <rect  x="36"  y="131"   width="28" height="7"   fill="none" stroke="#111827" stroke-width=".35"/>
      <rect  x="42.5" y="135.5" width="15" height="2.5" fill="none" stroke="#111827" stroke-width=".4"/>
      <circle cx="50" cy="124" r=".65"                 fill="#111827"/>
    </svg>
    {nodes}
  </div>

  {depth_html}

  <div style="text-align:center;font-size:8px;color:#1f2937;margin-top:8px;letter-spacing:.07em;">
    Player + years left on contract{legend_stats}
  </div>
  <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;
              font-size:10px;font-weight:700;margin-top:6px;">
    <span style="color:#fff;">Contracted</span>
    <span style="color:#f59e0b;">Final Year</span>
    <span style="color:#ef4444;">Out of Contract</span>
    <span style="color:#22c55e;">On Loan</span>
  </div>
</div>"""

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {"slot_map":{}, "depth":[], "move_player":None,
             "df":None, "df_sc":None, "last_team":None, "last_formation":None}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Squad Chart")
    st.markdown("---")
    st.markdown("**DATA**")
    uploaded = st.file_uploader("Upload CSV  (swap any time)", type=["csv"])

    if uploaded:
        @st.cache_data
        def _load(f) -> pd.DataFrame:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            for c in ["Player","Team","Position","League"]:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.strip()
            for c in ["Minutes played","Goals","Assists","Age","xG","xA"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["_ftok"] = df["Position"].apply(_tok)
            df["_key"]  = df["Player"]
            return df

        raw = _load(uploaded)
        if st.session_state.df is None or len(raw) != len(st.session_state.df):
            st.session_state.df   = raw
            st.session_state.df_sc = None   # force recompute

        if st.session_state.df_sc is None:
            with st.spinner("Computing role scores…"):
                st.session_state.df_sc = compute_role_scores(st.session_state.df)
        st.success(f"✓ {len(st.session_state.df):,} players · role scores ready")

    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df
        lgs = ["All"] + sorted(df["League"].unique())
        lg  = st.selectbox("League", lgs)
        fdf = df if lg == "All" else df[df["League"] == lg]
        sel_team = st.selectbox("Team", sorted(fdf["Team"].unique()))
        formation = st.selectbox("Formation", list(FORMATIONS.keys()))

        st.markdown("---")
        st.markdown("**DISPLAY**")
        show_mins    = st.toggle("Minutes played",  True)
        show_goals   = st.toggle("Goals",            True)
        show_assists = st.toggle("Assists",           True)
        show_roles   = st.toggle("Role scores",      True)
        canva_mode   = st.toggle("1920×1080 Canva",  False)

        st.markdown("---")
        changed = (sel_team   != st.session_state.last_team or
                   formation  != st.session_state.last_formation)
        if st.button("🔄 Build / Rebuild") or changed:
            tdf = df[df["Team"] == sel_team].copy()
            tdf["_key"] = tdf["Player"]
            sm, dep = assign_players(tdf.to_dict("records"), formation)
            st.session_state.slot_map      = sm
            st.session_state.depth         = dep
            st.session_state.last_team     = sel_team
            st.session_state.last_formation = formation
            st.session_state.move_player   = None

        # ── Move panel ────────────────────────────────────────────────────────
        if st.session_state.move_player:
            mp = st.session_state.move_player
            st.markdown(f"**MOVING:** {mp['player']['Player']}")
            opts = {f"{s['label']} ({s['id']})": s["id"]
                    for s in FORMATIONS[formation]}
            dest_lbl = st.selectbox("Move to", list(opts.keys()))
            if st.button("✅ Confirm"):
                p = mp["player"]; fid = mp["from_slot"]
                dest_id = opts[dest_lbl]
                if fid == "_depth":
                    st.session_state.depth = [
                        x for x in st.session_state.depth if x["_key"] != p["_key"]]
                elif fid in st.session_state.slot_map:
                    st.session_state.slot_map[fid] = [
                        x for x in st.session_state.slot_map[fid] if x["_key"] != p["_key"]]
                st.session_state.slot_map.setdefault(dest_id, []).append(p)
                st.session_state.move_player = None
                st.rerun()
            if st.button("❌ Cancel"):
                st.session_state.move_player = None
                st.rerun()

        st.markdown("---")
        # ── Add player ────────────────────────────────────────────────────────
        st.markdown("**ADD PLAYER**")
        nn  = st.text_input("Name", key="nn")
        np_ = st.selectbox("Position", list(CANONICAL.keys()), key="np_")
        nm_ = st.number_input("Minutes", 0, 5000, 0, 10, key="nm_")
        ng_ = st.number_input("Goals",   0, 100,  0,  key="ng_")
        na_ = st.number_input("Assists", 0, 100,  0,  key="na_")
        ne_ = st.text_input("Contract expires", "2026-06-30", key="ne_")
        nl_ = st.checkbox("On Loan?", key="nl_")
        sl_opts = {f"{s['label']} ({s['id']})": s["id"]
                   for s in FORMATIONS.get(formation, [])}
        ns_ = st.selectbox("Add to slot", list(sl_opts.keys()), key="ns_")
        if st.button("➕ Add") and nn.strip():
            new_p = {
                "Player": nn.strip(), "Position": np_, "_key": f"custom_{nn}",
                "Minutes played": nm_, "Goals": ng_, "Assists": na_,
                "Contract expires": ne_, "On Loan": "yes" if nl_ else "no",
                "League": lg, "Team": sel_team,
            }
            st.session_state.slot_map.setdefault(sl_opts[ns_], []).append(new_p)
            st.rerun()
    else:
        st.info("Upload a CSV to get started.")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;font-size:20px;letter-spacing:.18em;margin:0 0 4px;'>"
    "SQUAD DEPTH CHART</h1>"
    "<p style='text-align:center;font-size:9px;color:#111827;letter-spacing:.13em;margin:0 0 14px;'>"
    "ORDERED BY MINUTES PLAYED · ROLE SCORES VS SAME LEAGUE &amp; POSITION GROUP</p>",
    unsafe_allow_html=True,
)

if not st.session_state.slot_map:
    st.markdown(
        "<div style='text-align:center;color:#0d1220;font-size:11px;"
        "padding:100px 20px;border:1px dashed #0d1220;letter-spacing:.12em;'>"
        "UPLOAD A CSV AND SELECT A TEAM TO GET STARTED</div>",
        unsafe_allow_html=True,
    )
    st.stop()

formation = st.session_state.last_formation or "4-2-3-1"
team_name = st.session_state.last_team or ""
league_nm = ""
if st.session_state.df is not None and team_name:
    tdf2 = st.session_state.df[st.session_state.df["Team"] == team_name]
    if not tdf2.empty and "League" in tdf2.columns:
        league_nm = tdf2["League"].iloc[0]

slots    = FORMATIONS[formation]
slot_map = st.session_state.slot_map
depth    = st.session_state.depth
df_sc    = st.session_state.df_sc    # may be None

# retrieve toggle values from session state (set by sidebar widgets)
def _tog(k, d): return st.session_state.get(k, d)

pitch = render_pitch(
    team_name, league_nm, formation,
    slots, slot_map, depth, df_sc,
    _tog("show_mins", True), _tog("show_goals", True), _tog("show_assists", True),
    _tog("show_roles", True), _tog("canva_mode", False),
)

if _tog("canva_mode", False):
    st.markdown(
        f'<div style="width:100%;aspect-ratio:16/9;overflow:hidden;">{pitch}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(pitch, unsafe_allow_html=True)

# ── Move / Remove ──────────────────────────────────────────────────────────────
st.markdown("---")
all_on = []
for sl in slots:
    for p in slot_map.get(sl["id"], []):
        all_on.append({"sid": sl["id"], "lbl": sl["label"], "player": p})
for p in depth:
    all_on.append({"sid": "_depth", "lbl": "DEPTH", "player": p})

if all_on:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div style='font-size:9px;color:#374151;letter-spacing:.1em;margin-bottom:3px;'>MOVE PLAYER</div>",
                    unsafe_allow_html=True)
        mv_opts = {f"{e['player']['Player']} ({e['lbl']})": e for e in all_on}
        mv_sel  = st.selectbox("", list(mv_opts.keys()), key="mv_sel",
                                label_visibility="collapsed")
        if st.button("Select for Move"):
            e = mv_opts[mv_sel]
            st.session_state.move_player = {"player": e["player"], "from_slot": e["sid"]}
            st.rerun()
    with c2:
        st.markdown("<div style='font-size:9px;color:#374151;letter-spacing:.1em;margin-bottom:3px;'>REMOVE PLAYER</div>",
                    unsafe_allow_html=True)
        rm_opts = {f"{e['player']['Player']} ({e['lbl']})": e for e in all_on}
        rm_sel  = st.selectbox("", list(rm_opts.keys()), key="rm_sel",
                                label_visibility="collapsed")
        if st.button("🗑 Remove"):
            e = rm_opts[rm_sel]; sid = e["sid"]; pk = e["player"]["_key"]
            if sid == "_depth":
                st.session_state.depth = [x for x in st.session_state.depth
                                          if x["_key"] != pk]
            else:
                st.session_state.slot_map[sid] = [
                    x for x in st.session_state.slot_map.get(sid, [])
                    if x["_key"] != pk]
            st.rerun()

# ── Full squad table ───────────────────────────────────────────────────────────
if st.session_state.df is not None and team_name:
    with st.expander("📋 Full Squad"):
        tdf3 = st.session_state.df[st.session_state.df["Team"] == team_name]
        show_c = [c for c in ["Player","Position","Minutes played","Goals","Assists",
                               "Market value","Contract expires","Age"]
                  if c in tdf3.columns]
        st.dataframe(
            tdf3[show_c].sort_values("Minutes played", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )
