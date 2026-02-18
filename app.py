"""
Squad Depth Chart
=================
• Upload any CSV via sidebar — swap any time, no code changes
• Select team + formation → auto-builds depth chart
• Click any player to move them to a different slot
• Add custom players manually
• Remove players from the chart

Run:
    pip install streamlit pandas
    streamlit run app.py
"""

import re
import math
from datetime import date
import streamlit as st
import pandas as pd

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Squad Depth Chart", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background: #000 !important;
    color: #fff !important;
}
.stApp { background: #000 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid #1c1c1c !important;
}
section[data-testid="stSidebar"] * { color: #fff !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    color: #fff !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Dropdowns */
.stSelectbox > div > div { background: #0c0c0c !important; border: 1px solid #2a2a2a !important; }
div[data-baseweb="select"] * { background: #0c0c0c !important; color: #fff !important; }
div[data-baseweb="popover"] * { background: #111 !important; color: #fff !important; }

/* Inputs */
.stTextInput > div > div > input {
    background: #0c0c0c !important; border: 1px solid #2a2a2a !important;
    color: #fff !important; font-family: 'Share Tech Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: #fff !important; color: #000 !important;
    font-weight: 900 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; border: none !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
}
.stButton > button:hover { background: #e0e0e0 !important; }

/* Labels */
label, .stSelectbox label, .stTextInput label {
    color: #555 !important; font-size: 9px !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0c0c0c !important; border: 1px solid #1c1c1c !important;
    color: #fff !important; font-size: 11px !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid #1c1c1c !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2a2a !important;
    background: #080808 !important;
}

/* Dividers */
hr { border-color: #1c1c1c !important; }

h1, h2, h3 { color: #fff !important; letter-spacing: 0.15em !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Position → canonical group ───────────────────────────────────────────────
# First listed position in CSV determines slot. Groups defined per user spec.

CANONICAL: dict[str, str] = {
    # GK
    "GK":   "GK",
    # Centre backs
    "CB":   "CB",
    "LCB":  "LCB",
    "RCB":  "RCB",
    # Left back / wingback — both map to LB group (formation picks LB vs LWB slot)
    "LB":   "LB",
    "LWB":  "LWB",
    # Right back / wingback
    "RB":   "RB",
    "RWB":  "RWB",
    # Defensive mid — LDMF, DMF, RDMF → DM
    "DMF":  "DM",
    "LDMF": "DM",
    "RDMF": "DM",
    # Central mid
    "LCMF": "CM",
    "RCMF": "CM",
    # Attacking mid — AMF only (central)
    "AMF":  "AM",
    # Left wing — LW, LWF, LAMF all → LW
    "LAMF": "LW",
    "LW":   "LW",
    "LWF":  "LW",
    # Right wing — RW, RWF, RAMF all → RW
    "RAMF": "RW",
    "RW":   "RW",
    "RWF":  "RW",
    # Striker
    "CF":   "ST",
}

def get_canon(position_str: str) -> str:
    first = str(position_str).split(",")[0].strip().upper()
    return CANONICAL.get(first, "CM")

# ─── Formations ───────────────────────────────────────────────────────────────
# accepts = list of canonical groups that can go in this slot.
# LB slots accept ["LB","LWB"], RB slots accept ["RB","RWB"]
# 3-back formations have LCB/CB/RCB slots; LWB/RWB slots for wingbacks.

FORMATIONS: dict[str, list[dict]] = {

    "4-2-3-1": [
        # attack
        {"id":"ST",   "label":"ST",  "x":50, "y":9,  "accepts":["ST"]},
        {"id":"LW",   "label":"LW",  "x":13, "y":25, "accepts":["LW","AM"]},
        {"id":"AM",   "label":"AM",  "x":50, "y":24, "accepts":["AM"]},
        {"id":"RW",   "label":"RW",  "x":87, "y":25, "accepts":["RW","AM"]},
        # midfield
        {"id":"DM1",  "label":"DM",  "x":35, "y":43, "accepts":["DM","CM"]},
        {"id":"DM2",  "label":"DM",  "x":65, "y":43, "accepts":["DM","CM"]},
        # defence
        {"id":"LB",   "label":"LB",  "x":9,  "y":63, "accepts":["LB","LWB"]},
        {"id":"CB1",  "label":"CB",  "x":32, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"CB2",  "label":"CB",  "x":68, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RB",   "label":"RB",  "x":91, "y":63, "accepts":["RB","RWB"]},
        {"id":"GK",   "label":"GK",  "x":50, "y":84, "accepts":["GK"]},
    ],

    "4-3-3": [
        {"id":"ST",  "label":"ST", "x":50, "y":9,  "accepts":["ST"]},
        {"id":"LW",  "label":"LW", "x":15, "y":16, "accepts":["LW"]},
        {"id":"RW",  "label":"RW", "x":85, "y":16, "accepts":["RW"]},
        {"id":"CM1", "label":"CM", "x":22, "y":36, "accepts":["CM","DM","AM"]},
        {"id":"CM2", "label":"CM", "x":50, "y":32, "accepts":["CM","DM","AM"]},
        {"id":"CM3", "label":"CM", "x":78, "y":36, "accepts":["CM","DM","AM"]},
        {"id":"LB",  "label":"LB", "x":9,  "y":63, "accepts":["LB","LWB"]},
        {"id":"CB1", "label":"CB", "x":32, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"CB2", "label":"CB", "x":68, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RB",  "label":"RB", "x":91, "y":63, "accepts":["RB","RWB"]},
        {"id":"GK",  "label":"GK", "x":50, "y":84, "accepts":["GK"]},
    ],

    "4-4-2": [
        {"id":"ST1", "label":"ST", "x":35, "y":9,  "accepts":["ST"]},
        {"id":"ST2", "label":"ST", "x":65, "y":9,  "accepts":["ST"]},
        {"id":"LW",  "label":"LW", "x":9,  "y":34, "accepts":["LW","AM"]},
        {"id":"CM1", "label":"CM", "x":34, "y":38, "accepts":["CM","DM","AM"]},
        {"id":"CM2", "label":"CM", "x":66, "y":38, "accepts":["CM","DM","AM"]},
        {"id":"RW",  "label":"RW", "x":91, "y":34, "accepts":["RW","AM"]},
        {"id":"LB",  "label":"LB", "x":9,  "y":63, "accepts":["LB","LWB"]},
        {"id":"CB1", "label":"CB", "x":32, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"CB2", "label":"CB", "x":68, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RB",  "label":"RB", "x":91, "y":63, "accepts":["RB","RWB"]},
        {"id":"GK",  "label":"GK", "x":50, "y":84, "accepts":["GK"]},
    ],

    # 3-back: LCB, CB, RCB + LWB, RWB as wing backs
    "3-5-2": [
        {"id":"ST1", "label":"ST",  "x":35, "y":9,  "accepts":["ST"]},
        {"id":"ST2", "label":"ST",  "x":65, "y":9,  "accepts":["ST"]},
        {"id":"LWB", "label":"LWB", "x":9,  "y":34, "accepts":["LWB","LB","LW"]},
        {"id":"CM1", "label":"CM",  "x":28, "y":40, "accepts":["CM","AM"]},
        {"id":"DM",  "label":"DM",  "x":50, "y":36, "accepts":["DM","CM"]},
        {"id":"CM2", "label":"CM",  "x":72, "y":40, "accepts":["CM","AM"]},
        {"id":"RWB", "label":"RWB", "x":91, "y":34, "accepts":["RWB","RB","RW"]},
        {"id":"LCB", "label":"LCB", "x":25, "y":64, "accepts":["LCB","CB"]},
        {"id":"CB",  "label":"CB",  "x":50, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RCB", "label":"RCB", "x":75, "y":64, "accepts":["RCB","CB"]},
        {"id":"GK",  "label":"GK",  "x":50, "y":84, "accepts":["GK"]},
    ],

    # 3-4-1-2: like 3-5-2 but AM sits between mids and strikers
    "3-4-1-2": [
        {"id":"ST1", "label":"ST",  "x":35, "y":8,  "accepts":["ST"]},
        {"id":"ST2", "label":"ST",  "x":65, "y":8,  "accepts":["ST"]},
        {"id":"AM",  "label":"AM",  "x":50, "y":20, "accepts":["AM","LW","RW"]},
        {"id":"LWB", "label":"LWB", "x":9,  "y":36, "accepts":["LWB","LB","LW"]},
        {"id":"CM1", "label":"CM",  "x":34, "y":40, "accepts":["CM","DM"]},
        {"id":"CM2", "label":"CM",  "x":66, "y":40, "accepts":["CM","DM"]},
        {"id":"RWB", "label":"RWB", "x":91, "y":36, "accepts":["RWB","RB","RW"]},
        {"id":"LCB", "label":"LCB", "x":25, "y":62, "accepts":["LCB","CB"]},
        {"id":"CB",  "label":"CB",  "x":50, "y":65, "accepts":["CB","LCB","RCB"]},
        {"id":"RCB", "label":"RCB", "x":75, "y":62, "accepts":["RCB","CB"]},
        {"id":"GK",  "label":"GK",  "x":50, "y":82, "accepts":["GK"]},
    ],

    "4-5-1": [
        {"id":"ST",  "label":"ST",  "x":50, "y":9,  "accepts":["ST"]},
        {"id":"LW",  "label":"LW",  "x":9,  "y":25, "accepts":["LW"]},
        {"id":"CM1", "label":"CM",  "x":30, "y":33, "accepts":["CM","DM"]},
        {"id":"AM",  "label":"AM",  "x":50, "y":25, "accepts":["AM"]},
        {"id":"CM2", "label":"CM",  "x":70, "y":33, "accepts":["CM","DM"]},
        {"id":"RW",  "label":"RW",  "x":91, "y":25, "accepts":["RW"]},
        {"id":"LB",  "label":"LB",  "x":9,  "y":63, "accepts":["LB","LWB"]},
        {"id":"CB1", "label":"CB",  "x":32, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"CB2", "label":"CB",  "x":68, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RB",  "label":"RB",  "x":91, "y":63, "accepts":["RB","RWB"]},
        {"id":"GK",  "label":"GK",  "x":50, "y":84, "accepts":["GK"]},
    ],

    "4-1-4-1": [
        {"id":"ST",  "label":"ST",  "x":50, "y":9,  "accepts":["ST"]},
        {"id":"LW",  "label":"LW",  "x":9,  "y":26, "accepts":["LW"]},
        {"id":"CM1", "label":"CM",  "x":31, "y":29, "accepts":["CM","AM"]},
        {"id":"CM2", "label":"CM",  "x":69, "y":29, "accepts":["CM","AM"]},
        {"id":"RW",  "label":"RW",  "x":91, "y":26, "accepts":["RW"]},
        {"id":"DM",  "label":"DM",  "x":50, "y":44, "accepts":["DM","CM"]},
        {"id":"LB",  "label":"LB",  "x":9,  "y":63, "accepts":["LB","LWB"]},
        {"id":"CB1", "label":"CB",  "x":32, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"CB2", "label":"CB",  "x":68, "y":67, "accepts":["CB","LCB","RCB"]},
        {"id":"RB",  "label":"RB",  "x":91, "y":63, "accepts":["RB","RWB"]},
        {"id":"GK",  "label":"GK",  "x":50, "y":84, "accepts":["GK"]},
    ],
}

# Pitch order for priority filling (attack first)
PITCH_ORDER = ["GK","LCB","CB","RCB","LB","RB","LWB","RWB","DM","CM","AM","LW","RW","ST"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def contract_years(expires_str: str) -> int:
    if not expires_str or str(expires_str).strip() in ("", "nan", "NaT"):
        return -1
    m = re.search(r"(20\d{2})", str(expires_str))
    if not m:
        return -1
    return max(0, int(m.group(1)) - date.today().year)

def player_color(yrs: int) -> str:
    if yrs == 0:  return "#ef4444"
    if yrs == 1:  return "#f59e0b"
    return "#ffffff"

def assign_auto(players: list[dict], formation_key: str) -> dict[str, list]:
    """Auto-assign players to slots by position match, ordered by minutes played."""
    slots = FORMATIONS.get(formation_key, FORMATIONS["4-2-3-1"])

    # Group slots by label to split players across same-label slots
    label_groups: dict[str, list] = {}
    for s in slots:
        label_groups.setdefault(s["label"], []).append(s)

    assigned: set = set()
    slot_map: dict[str, list] = {}

    for label in PITCH_ORDER:
        if label not in label_groups:
            continue
        slot_list = label_groups[label]
        accepts   = {a for s in slot_list for a in s["accepts"]}

        matched = [
            p for p in players
            if p["_key"] not in assigned
            and get_canon(p.get("Position", "")) in accepts
        ]
        matched.sort(key=lambda p: -float(p.get("Minutes played") or 0))
        for p in matched:
            assigned.add(p["_key"])

        n     = len(slot_list)
        chunk = math.ceil(len(matched) / n) if n and matched else max(len(matched), 1)
        for i, slot in enumerate(slot_list):
            slot_map[slot["id"]] = matched[i * chunk:(i + 1) * chunk]

    for slot in slots:
        slot_map.setdefault(slot["id"], [])

    return slot_map

# ─── Session state ─────────────────────────────────────────────────────────────

def init_state():
    if "slot_map" not in st.session_state:
        st.session_state.slot_map = {}
    if "move_player" not in st.session_state:
        st.session_state.move_player = None   # {"player": {...}, "from_slot": str}
    if "df" not in st.session_state:
        st.session_state.df = None
    if "last_team" not in st.session_state:
        st.session_state.last_team = None
    if "last_formation" not in st.session_state:
        st.session_state.last_formation = None

init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # CSV upload — swap any time
    st.markdown("**DATA SOURCE**")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
        help="Wyscout export. Needs: Player, Team, Position, Minutes played.")

    if uploaded:
        @st.cache_data
        def load_csv(f) -> pd.DataFrame:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            for col in ["Player","Team","Position","League"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            for col in ["Minutes played","Goals","Assists","Age"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df
        st.session_state.df = load_csv(uploaded)
        st.success(f"✓ {len(st.session_state.df)} players loaded")

    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df

        # League filter
        leagues = ["All leagues"] + sorted(df["League"].unique()) if "League" in df.columns else ["All leagues"]
        league_filter = st.selectbox("League", leagues)

        filtered = df if league_filter == "All leagues" else df[df["League"] == league_filter]
        teams = sorted(filtered["Team"].unique())
        selected_team = st.selectbox("Team", teams)

        formation = st.selectbox("Formation", list(FORMATIONS.keys()))

        # Auto-build when team/formation changes
        team_changed = (selected_team != st.session_state.last_team or
                        formation    != st.session_state.last_formation)

        if st.button("🔄 Auto-Build Chart") or team_changed:
            team_df = df[df["Team"] == selected_team].copy()
            team_df["_key"] = team_df["Player"]
            players = team_df.to_dict("records")
            st.session_state.slot_map = assign_auto(players, formation)
            st.session_state.last_team = selected_team
            st.session_state.last_formation = formation
            st.session_state.move_player = None

        st.markdown("---")

        # ── Move player panel ──────────────────────────────────────────────
        if st.session_state.move_player:
            mp = st.session_state.move_player
            st.markdown(f"**MOVING:** {mp['player']['Player']}")
            slot_labels = {s["id"]: s["label"] for s in FORMATIONS[formation]}
            dest = st.selectbox("Move to slot", list(slot_labels.keys()),
                                format_func=lambda x: f"{slot_labels[x]} ({x})")
            if st.button("✅ Confirm Move"):
                p    = mp["player"]
                from_id = mp["from_slot"]
                # Remove from current slot
                if from_id in st.session_state.slot_map:
                    st.session_state.slot_map[from_id] = [
                        x for x in st.session_state.slot_map[from_id]
                        if x["_key"] != p["_key"]
                    ]
                # Add to destination
                st.session_state.slot_map.setdefault(dest, []).append(p)
                st.session_state.move_player = None
                st.rerun()
            if st.button("❌ Cancel"):
                st.session_state.move_player = None
                st.rerun()

        st.markdown("---")

        # ── Add custom player ──────────────────────────────────────────────
        st.markdown("**ADD PLAYER**")
        new_name    = st.text_input("Name")
        new_pos_raw = st.selectbox("Position", list(CANONICAL.keys()))
        new_mins    = st.number_input("Minutes played", 0, 5000, 0, step=10)
        new_goals   = st.number_input("Goals", 0, 100, 0)
        new_assists = st.number_input("Assists", 0, 100, 0)
        new_expires = st.text_input("Contract expires (YYYY-MM-DD)", "2026-06-30")
        slot_labels = {s["id"]: s["label"] for s in FORMATIONS.get(formation, [])}
        add_to_slot = st.selectbox("Add to slot", list(slot_labels.keys()),
                                   format_func=lambda x: f"{slot_labels[x]} ({x})")

        if st.button("➕ Add Player") and new_name.strip():
            new_player = {
                "Player": new_name.strip(),
                "Position": new_pos_raw,
                "_key": f"custom_{new_name.strip()}",
                "Minutes played": new_mins,
                "Goals": new_goals,
                "Assists": new_assists,
                "Contract expires": new_expires,
                "League": "", "Team": selected_team,
            }
            st.session_state.slot_map.setdefault(add_to_slot, []).append(new_player)
            st.rerun()

    else:
        st.info("Upload a CSV to get started.")

# ─── Main pitch area ──────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;font-size:18px;letter-spacing:.2em;"
    "margin-bottom:2px;margin-top:0;'>SQUAD DEPTH CHART</h1>"
    "<p style='text-align:center;font-size:9px;color:#333;letter-spacing:.14em;"
    "margin-top:0;margin-bottom:16px;'>ORDERED BY MINUTES PLAYED</p>",
    unsafe_allow_html=True,
)

if not st.session_state.slot_map:
    st.markdown(
        "<div style='text-align:center;color:#1e1e1e;font-size:11px;"
        "padding:100px 20px;border:1px dashed #111;letter-spacing:.12em;'>"
        "UPLOAD A CSV AND SELECT A TEAM TO GET STARTED</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Determine formation from session (last used)
formation = st.session_state.last_formation or "4-2-3-1"
team_name = st.session_state.last_team or ""
league_name = ""
if st.session_state.df is not None and team_name:
    tdf = st.session_state.df[st.session_state.df["Team"] == team_name]
    if not tdf.empty and "League" in tdf.columns:
        league_name = tdf["League"].iloc[0]

slots   = FORMATIONS[formation]
slot_map = st.session_state.slot_map

# ── Render pitch as HTML ───────────────────────────────────────────────────────
nodes = ""
for slot in slots:
    ps = slot_map.get(slot["id"], [])
    badge = (
        f'<div style="display:inline-block;padding:1px 6px;'
        f'border:1.5px solid #ef4444;color:#ef4444;font-size:9px;'
        f'font-weight:900;letter-spacing:.15em;margin-bottom:3px;'
        f'background:rgba(0,0,0,.75);">{slot["label"]}</div>'
    )
    rows = ""
    for i, p in enumerate(ps):
        mins    = int(float(p.get("Minutes played") or 0))
        yrs     = contract_years(str(p.get("Contract expires", "")))
        yr_str  = f"+{yrs}" if yrs >= 0 else "+?"
        goals   = float(p.get("Goals") or 0)
        assists = float(p.get("Assists") or 0)
        col     = player_color(yrs)
        fw      = "bold" if i == 0 else "normal"
        stat_parts = [f"{mins}′"]
        if goals   > 0: stat_parts.append(f"{int(goals)}⚽")
        if assists > 0: stat_parts.append(f"{int(assists)}🅰")
        stat_html = (
            f'<div style="color:#555;font-size:7px;line-height:1.3;">'
            f'{" ".join(stat_parts)}</div>'
        )
        rows += (
            f'<div style="color:{col};font-size:8.5px;line-height:1.55;'
            f'font-weight:{fw};white-space:nowrap;text-shadow:0 0 6px #000;">'
            f'{p["Player"]} {yr_str}</div>{stat_html}'
        )
    if not ps:
        rows = '<div style="color:#1e1e1e;font-size:8px;">—</div>'

    nodes += (
        f'<div style="position:absolute;left:{slot["x"]}%;top:{slot["y"]}%;'
        f'transform:translate(-50%,-50%);text-align:center;min-width:76px;z-index:10;">'
        f'{badge}<div>{rows}</div></div>'
    )

pitch_html = f"""
<div style="font-family:'Share Tech Mono',monospace;color:#fff;background:#000;padding:0 4px 12px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
    <div style="font-size:10px;color:#444;letter-spacing:.1em;">{league_name}</div>
    <div style="font-weight:900;font-size:16px;letter-spacing:.1em;">{team_name}</div>
    <div style="font-size:10px;color:#444;letter-spacing:.08em;">{formation}</div>
  </div>

  <div style="position:relative;background:#0d1a0d;padding-bottom:134%;overflow:hidden;">
    <svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"
         viewBox="0 0 100 134" preserveAspectRatio="none">
      <rect  x="2"    y="2"      width="96"  height="130" fill="none" stroke="#fff" stroke-width=".55" opacity=".7"/>
      <line  x1="2"   y1="67"    x2="98"    y2="67"      stroke="#fff" stroke-width=".4"  opacity=".6"/>
      <circle cx="50" cy="67"    r="10"                   fill="none" stroke="#fff" stroke-width=".4"  opacity=".6"/>
      <circle cx="50" cy="67"    r=".65"                  fill="#fff" opacity=".5"/>
      <rect  x="22"   y="2"      width="56"  height="17"  fill="none" stroke="#fff" stroke-width=".4"  opacity=".55"/>
      <rect  x="36"   y="2"      width="28"  height="6"   fill="none" stroke="#fff" stroke-width=".3"  opacity=".5"/>
      <rect  x="42.5" y=".3"     width="15"  height="2.5" fill="none" stroke="#fff" stroke-width=".4"  opacity=".45"/>
      <circle cx="50" cy="13.5"  r=".5"                   fill="#fff" opacity=".45"/>
      <rect  x="22"   y="115"    width="56"  height="17"  fill="none" stroke="#fff" stroke-width=".4"  opacity=".55"/>
      <rect  x="36"   y="126"    width="28"  height="6"   fill="none" stroke="#fff" stroke-width=".3"  opacity=".5"/>
      <rect  x="42.5" y="131.5"  width="15"  height="2.5" fill="none" stroke="#fff" stroke-width=".4"  opacity=".45"/>
      <circle cx="50" cy="119"   r=".5"                   fill="#fff" opacity=".45"/>
    </svg>
    {nodes}
  </div>

  <div style="text-align:center;font-size:9px;color:#444;margin-top:10px;letter-spacing:.08em;">
    <span style="color:#666;">How to read:</span>
    Player + contract years &nbsp;·&nbsp; ′=mins &nbsp;⚽=goals &nbsp;🅰=assists
  </div>
  <div style="display:flex;gap:18px;justify-content:center;flex-wrap:wrap;
              font-size:10px;font-weight:700;letter-spacing:.07em;margin-top:8px;">
    <span style="color:#fff;">Under Contract</span>
    <span style="color:#f59e0b;">Final Year</span>
    <span style="color:#ef4444;">Out of Contract</span>
  </div>
</div>"""

st.markdown(pitch_html, unsafe_allow_html=True)

# ── Move/Remove controls ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:9px;color:#444;letter-spacing:.14em;margin-bottom:8px;'>"
    "MOVE OR REMOVE PLAYERS</div>",
    unsafe_allow_html=True,
)

# Build a flat list of all players currently on the chart
all_on_chart = []
for slot in slots:
    for p in slot_map.get(slot["id"], []):
        all_on_chart.append({"slot_id": slot["id"], "slot_label": slot["label"], "player": p})

if all_on_chart:
    col_move, col_remove = st.columns(2)

    with col_move:
        st.markdown("<div style='font-size:9px;color:#555;letter-spacing:.1em;'>MOVE PLAYER</div>", unsafe_allow_html=True)
        move_options = {f"{e['player']['Player']} ({e['slot_label']})": e for e in all_on_chart}
        move_sel = st.selectbox("Select player to move", list(move_options.keys()), key="move_sel")
        if st.button("Select for Move"):
            entry = move_options[move_sel]
            st.session_state.move_player = {
                "player": entry["player"],
                "from_slot": entry["slot_id"],
            }
            st.rerun()

    with col_remove:
        st.markdown("<div style='font-size:9px;color:#555;letter-spacing:.1em;'>REMOVE PLAYER</div>", unsafe_allow_html=True)
        remove_options = {f"{e['player']['Player']} ({e['slot_label']})": e for e in all_on_chart}
        remove_sel = st.selectbox("Select player to remove", list(remove_options.keys()), key="remove_sel")
        if st.button("🗑 Remove from Chart"):
            entry = remove_options[remove_sel]
            sid   = entry["slot_id"]
            pkey  = entry["player"]["_key"]
            st.session_state.slot_map[sid] = [
                x for x in st.session_state.slot_map[sid] if x["_key"] != pkey
            ]
            st.rerun()

# ── Full squad table ───────────────────────────────────────────────────────────
if st.session_state.df is not None and team_name:
    with st.expander("📋 Full Squad — sorted by minutes played"):
        tdf = st.session_state.df[st.session_state.df["Team"] == team_name]
        show = [c for c in ["Player","Position","Minutes played","Goals","Assists",
                             "Market value","Contract expires","Age"] if c in tdf.columns]
        display = tdf[show].sort_values("Minutes played", ascending=False).reset_index(drop=True)
        display.index += 1
        st.dataframe(display, use_container_width=True)
