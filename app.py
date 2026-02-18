"""
Squad Depth Chart — v5
pip install streamlit pandas numpy
streamlit run app.py
"""

import re, base64
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
.stTextInput>div>div>input,.stNumberInput input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important}
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
ROLE_BUCKETS: dict[str, dict] = {
    "GK": {
        "Shot Stopper GK": {"metrics": {"Prevented goals per 90": 3, "Save rate, %": 1}},
        "Ball Playing GK": {"metrics": {"Passes per 90": 1, "Accurate passes, %": 3, "Accurate long passes, %": 2}},
        "Sweeper GK":      {"metrics": {"Exits per 90": 1}},
    },
    "CB": {
        "Ball Playing CB": {"metrics": {"Passes per 90":2,"Accurate passes, %":2,"Forward passes per 90":2,
            "Accurate forward passes, %":2,"Progressive passes per 90":2,"Progressive runs per 90":1.5,
            "Dribbles per 90":1.5,"Accurate long passes, %":1,"Passes to final third per 90":1.5}},
        "Wide CB":         {"metrics": {"Defensive duels per 90":1.5,"Defensive duels won, %":2,
            "Dribbles per 90":2,"Forward passes per 90":1,"Progressive passes per 90":1,"Progressive runs per 90":2}},
        "Box Defender":    {"metrics": {"Aerial duels per 90":1,"Aerial duels won, %":3,
            "PAdj Interceptions":2,"Shots blocked per 90":1,"Defensive duels won, %":4}},
    },
    "FB": {
        "Build Up FB":  {"metrics": {"Passes per 90":2,"Accurate passes, %":1.5,"Forward passes per 90":2,
            "Accurate forward passes, %":2,"Progressive passes per 90":2.5,"Progressive runs per 90":2,
            "Dribbles per 90":2,"Passes to final third per 90":2,"xA per 90":1}},
        "Attacking FB": {"metrics": {"Crosses per 90":2,"Dribbles per 90":3.5,"Accelerations per 90":1,
            "Successful dribbles, %":1,"Touches in box per 90":2,"Progressive runs per 90":3,
            "Passes to penalty area per 90":2,"xA per 90":3}},
        "Defensive FB": {"metrics": {"Aerial duels per 90":1,"Aerial duels won, %":1.5,
            "Defensive duels per 90":2,"PAdj Interceptions":3,"Shots blocked per 90":1,"Defensive duels won, %":3.5}},
    },
    "CM": {
        "Deep Playmaker CM":    {"metrics": {"Passes per 90":1,"Accurate passes, %":1,"Forward passes per 90":2,
            "Accurate forward passes, %":1.5,"Progressive passes per 90":3,"Passes to final third per 90":2.5,
            "Accurate long passes, %":1}},
        "Advanced Playmaker CM":{"metrics": {"Deep completions per 90":1.5,"Smart passes per 90":2,
            "xA per 90":4,"Passes to penalty area per 90":2}},
        "Defensive CM":         {"metrics": {"Defensive duels per 90":4,"Defensive duels won, %":4,
            "PAdj Interceptions":3,"Aerial duels per 90":0.5,"Aerial duels won, %":1}},
        "Ball Carrying CM":     {"metrics": {"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "ATT": {
        "Playmaker ATT":   {"metrics": {"Passes per 90":2,"xA per 90":3,"Key passes per 90":1,
            "Deep completions per 90":1.5,"Smart passes per 90":1.5,"Passes to penalty area per 90":2}},
        "Goal Threat ATT": {"metrics": {"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,"Touches in box per 90":2}},
        "Ball Carrier ATT":{"metrics": {"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "CF": {
        "Target Man CF":  {"metrics": {"Aerial duels per 90":3,"Aerial duels won, %":5}},
        "Goal Threat CF": {"metrics": {"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,
            "Touches in box per 90":1,"Shots on target, %":0.5}},
        "Link Up CF":     {"metrics": {"Passes per 90":2,"Passes to penalty area per 90":1.5,
            "Deep completions per 90":1,"Smart passes per 90":1.5,"Accurate passes, %":1.5,
            "Key passes per 90":1,"Dribbles per 90":2,"Successful dribbles, %":1,
            "Progressive runs per 90":2,"xA per 90":3}},
    },
}

ROLE_KEY_MAP: dict[str,str] = {
    "GK":"GK","CB":"CB","LCB":"CB","RCB":"CB",
    "LB":"FB","RB":"FB","LWB":"FB","RWB":"FB",
    "DMF":"CM","LDMF":"CM","RDMF":"CM","LCMF":"CM","RCMF":"CM",
    "AMF":"ATT","LAMF":"ATT","LW":"ATT","LWF":"ATT","RAMF":"ATT","RW":"ATT","RWF":"ATT",
    "CF":"CF",
}
POS_POOL_MAP: dict[str,list] = {
    "GK":["GK"],"CB":["CB","LCB","RCB"],"FB":["LB","RB","LWB","RWB"],
    "CM":["DMF","LDMF","RDMF","LCMF","RCMF"],
    "ATT":["AMF","LAMF","RAMF","LW","LWF","RW","RWF"],"CF":["CF"],
}
CANONICAL: dict[str,str] = {
    "GK":"GK","CB":"CB","LCB":"LCB","RCB":"RCB",
    "LB":"LB","LWB":"LWB","RB":"RB","RWB":"RWB",
    "DMF":"DM","LDMF":"DM","RDMF":"DM","LCMF":"CM","RCMF":"CM",
    "AMF":"AM","LAMF":"LW","LW":"LW","LWF":"LW",
    "RAMF":"RW","RW":"RW","RWF":"RW","CF":"ST",
}
SIDE_PREF: dict[str,str] = {
    "RCB":"R","RCMF":"R","RDMF":"R","RB":"R","RWB":"R","RW":"R","RWF":"R","RAMF":"R",
    "LCB":"L","LCMF":"L","LDMF":"L","LB":"L","LWB":"L","LW":"L","LWF":"L","LAMF":"L",
}

def _tok(pos:str)->str:  return str(pos).split(",")[0].strip().upper()
def _canon(pos:str)->str: return CANONICAL.get(_tok(pos),"CM")
def _side(pos:str)->str:  return SIDE_PREF.get(_tok(pos),"N")
def _role_key(pos:str)->str: return ROLE_KEY_MAP.get(_tok(pos),"ATT")
def _all_toks(pos:str)->list[str]: return [t.strip().upper() for t in str(pos).split(",") if t.strip()]
def _all_canons(pos:str)->list[str]: return [CANONICAL.get(t,"CM") for t in _all_toks(pos)]
def _multi_role(pos:str)->bool: return len(_all_toks(pos)) >= 3

# ── Formations ─────────────────────────────────────────────────────────────────
# 3-5-2: AM left · DM lower-centre · AM right  (fix #4)
FORMATIONS: dict[str,list[dict]] = {
    "4-2-3-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":13,"y":25, "accepts":["LW","AM"],        "side":"L"},
        {"id":"AM",  "label":"AM",  "x":50,"y":24, "accepts":["AM"],             "side":"N"},
        {"id":"RW",  "label":"RW",  "x":87,"y":25, "accepts":["RW","AM"],        "side":"R"},
        {"id":"DM1", "label":"DM",  "x":35,"y":43, "accepts":["DM","CM"],        "side":"L"},
        {"id":"DM2", "label":"DM",  "x":65,"y":43, "accepts":["DM","CM"],        "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
    "4-3-3": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":14,"y":16, "accepts":["LW"],             "side":"L"},
        {"id":"RW",  "label":"RW",  "x":86,"y":16, "accepts":["RW"],             "side":"R"},
        {"id":"DM",  "label":"DM",  "x":22,"y":36, "accepts":["DM","CM"],        "side":"L"},
        {"id":"CM",  "label":"CM",  "x":50,"y":32, "accepts":["CM","DM","AM"],   "side":"N"},
        {"id":"AM",  "label":"AM",  "x":78,"y":36, "accepts":["AM","CM"],        "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
    "4-4-2": [
        {"id":"ST1", "label":"ST",  "x":35,"y":9,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":9,  "accepts":["ST"],             "side":"R"},
        {"id":"LW",  "label":"LW",  "x":9, "y":34, "accepts":["LW","AM"],        "side":"L"},
        {"id":"CM1", "label":"CM",  "x":34,"y":38, "accepts":["CM","DM","AM"],   "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":38, "accepts":["CM","DM","AM"],   "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":34, "accepts":["RW","AM"],        "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
    # ── 3-5-2: AM left, DM lower-centre, AM right (fix #4) ──
    "3-5-2": [
        {"id":"ST1", "label":"ST",  "x":35,"y":9,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":9,  "accepts":["ST"],             "side":"R"},
        {"id":"LWB", "label":"LWB", "x":9, "y":32, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"AM1", "label":"AM",  "x":30,"y":36, "accepts":["AM","CM"],        "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":44, "accepts":["DM","CM"],        "side":"N"},
        {"id":"AM2", "label":"AM",  "x":70,"y":36, "accepts":["AM","CM"],        "side":"R"},
        {"id":"RWB", "label":"RWB", "x":91,"y":32, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":62, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":66, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":62, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":83, "accepts":["GK"],             "side":"N"},
    ],
    "3-4-1-2": [
        {"id":"ST1", "label":"ST",  "x":35,"y":8,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":8,  "accepts":["ST"],             "side":"R"},
        {"id":"AM",  "label":"AM",  "x":50,"y":20, "accepts":["AM","LW","RW"],   "side":"N"},
        {"id":"LWB", "label":"LWB", "x":9, "y":35, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"CM1", "label":"CM",  "x":34,"y":39, "accepts":["CM","DM"],        "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":39, "accepts":["CM","DM"],        "side":"R"},
        {"id":"RWB", "label":"RWB", "x":91,"y":35, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":61, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":65, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":61, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":82, "accepts":["GK"],             "side":"N"},
    ],
    "4-5-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":9, "y":25, "accepts":["LW"],             "side":"L"},
        {"id":"CM1", "label":"CM",  "x":30,"y":33, "accepts":["CM","DM"],        "side":"L"},
        {"id":"AM",  "label":"AM",  "x":50,"y":25, "accepts":["AM"],             "side":"N"},
        {"id":"CM2", "label":"CM",  "x":70,"y":33, "accepts":["CM","DM"],        "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":25, "accepts":["RW"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
    "4-1-4-1": [
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":9, "y":26, "accepts":["LW"],             "side":"L"},
        {"id":"CM1", "label":"CM",  "x":31,"y":29, "accepts":["CM","AM"],        "side":"L"},
        {"id":"CM2", "label":"CM",  "x":69,"y":29, "accepts":["CM","AM"],        "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":26, "accepts":["RW"],             "side":"R"},
        {"id":"DM",  "label":"DM",  "x":50,"y":44, "accepts":["DM","CM"],        "side":"N"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
}

PITCH_ORDER = ["GK","LCB","CB","RCB","LB","RB","LWB","RWB","DM","CM","AM","LW","RW","ST"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def contract_years(s) -> int:
    s = str(s or "").strip()
    if s in ("","nan","NaT"): return -1
    m = re.search(r"(20\d{2})", s)
    return max(0, int(m.group(1)) - date.today().year) if m else -1

def is_loan(p:dict) -> bool:
    for k in ("On loan","On Loan","on_loan","Loan","loan","On loan?"):
        if k in p and str(p[k]).strip().lower() in ("yes","y","true","1","on loan"):
            return True
    return False

def player_css_color(yrs:int, loan:bool) -> str:
    if loan:     return "#22c55e"
    if yrs == 0: return "#ef4444"
    if yrs == 1: return "#f59e0b"
    return "#ffffff"

def score_to_color(v:float) -> str:
    if np.isnan(v): return "#4b5563"
    v = max(0.0,min(100.0,float(v)))
    if v <= 50:
        t=v/50; r=int(239+(234-239)*t); g=int(68+(179-68)*t); b=int(68+(8-68)*t)
    else:
        t=(v-50)/50; r=int(234+(34-234)*t); g=int(179+(197-179)*t); b=int(8+(94-8)*t)
    return f"rgb({r},{g},{b})"

# ── Role scores ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_role_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    skip = {"Player","League","Team","Position","Age","Market value",
            "Contract expires","Matches played","Minutes played","Goals",
            "Assists","xG","xA","Birth country","Foot","Height","_ftok","_key"}
    for c in df.columns:
        if c not in skip and not c.startswith("On ") and "loan" not in c.lower():
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for rk, pool_pos in POS_POOL_MAP.items():
        for role_name, spec in ROLE_BUCKETS.get(rk,{}).items():
            col_out = f"_rs_{role_name}"
            df[col_out] = np.nan
            metrics = spec.get("metrics",{})
            for league in df["League"].unique():
                mask = ((df["League"]==league)&(df["_ftok"].isin(pool_pos))&(df["Minutes played"]>=200))
                pool = df[mask]
                if pool.empty: continue
                pcts: dict[str,pd.Series] = {}
                for met in metrics:
                    if met in pool.columns:
                        pcts[met] = pd.to_numeric(pool[met],errors="coerce").rank(pct=True,method="average")*100.0
                for idx in pool.index:
                    vals,wts=[],[]
                    for met,w in metrics.items():
                        if met in pcts and idx in pcts[met].index:
                            pv=pcts[met].loc[idx]
                            if not np.isnan(pv): vals.append(float(pv)); wts.append(float(w))
                    if vals: df.at[idx,col_out]=float(np.average(vals,weights=wts))
    return df

# ── Player assignment ──────────────────────────────────────────────────────────
def assign_players(players:list, formation_key:str) -> tuple[dict,list]:
    slots = FORMATIONS.get(formation_key, FORMATIONS["4-2-3-1"])
    by_label: dict[str,list] = {}
    for s in slots: by_label.setdefault(s["label"],[]).append(s)

    assigned: set = set()
    slot_map: dict[str,list] = {s["id"]:[] for s in slots}

    def primary_fits(p:dict, slot:dict)->bool:
        tok = _tok(p.get("Position",""))
        if slot.get("wb_only"):
            return tok in {"LB","LWB","RB","RWB"} and _canon(p.get("Position","")) in slot["accepts"]
        return _canon(p.get("Position","")) in slot["accepts"]

    def secondary_fits(p:dict, slot:dict)->bool:
        if slot.get("wb_only"): return False
        for t in _all_toks(p.get("Position",""))[1:]:
            if CANONICAL.get(t,"CM") in slot["accepts"]: return True
        return False

    def side_score(p:dict, ss:str)->int:
        ps=_side(p.get("Position",""))
        if ss=="N" or ps=="N": return 1
        return 0 if ps==ss else 2

    for label in PITCH_ORDER:
        if label not in by_label: continue
        slot_list = by_label[label]
        is_primary = True

        matched = [p for p in players if p["_key"] not in assigned
                   and any(primary_fits(p,s) for s in slot_list)]
        if not matched:
            is_primary = False
            matched = [p for p in players if p["_key"] not in assigned
                       and any(secondary_fits(p,s) for s in slot_list)]

        matched.sort(key=lambda p:-float(p.get("Minutes played") or 0))
        for p in matched: assigned.add(p["_key"])

        n = len(slot_list)
        if n == 1:
            slot_map[slot_list[0]["id"]] = matched
        else:
            ordered = sorted(slot_list, key=lambda s:{"L":0,"N":1,"R":2}[s["side"]])
            starters=[]; used=set()
            for sl in ordered:
                best=None; best_sc=99
                for p in matched:
                    if id(p) in used: continue
                    sc=side_score(p,sl["side"])
                    if sc<best_sc: best_sc=sc; best=p
                if best: starters.append((sl["id"],best)); used.add(id(best))
            depth_rem=[p for p in matched if id(p) not in used]
            for sl in slot_list: slot_map[sl["id"]]=[]
            for sid,p in starters: slot_map[sid].append(p)
            slot_map[slot_list[0]["id"]].extend(depth_rem)

    # annotate out-of-position
    for sid, ps in slot_map.items():
        slot_def = next((s for s in slots if s["id"]==sid), None)
        for p in ps:
            fits = primary_fits(p, slot_def) if slot_def else True
            p["_oop"]         = not fits
            p["_primary_pos"] = _tok(p.get("Position",""))

    depth = [p for p in players if p["_key"] not in assigned]
    depth.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    return slot_map, depth

# ── Score HTML ─────────────────────────────────────────────────────────────────
def all_roles_html(player:dict, df_sc, fs:str="8px")->str:
    if df_sc is None or df_sc.empty: return ""
    rows = df_sc[df_sc["Player"]==player.get("Player","")]
    if rows.empty: return ""
    row = rows.iloc[0]
    rk = _role_key(player.get("Position",""))
    scores = {}
    for rn in ROLE_BUCKETS.get(rk,{}):
        v=row.get(f"_rs_{rn}",np.nan)
        if isinstance(v,(int,float)) and not np.isnan(float(v)): scores[rn]=float(v)
    if not scores: return ""
    best=max(scores,key=scores.get)
    lines=[]
    for rn,sc in sorted(scores.items(),key=lambda x:-x[1]):
        sc_col=score_to_color(sc); is_b=rn==best
        lines.append(
            f'<div style="display:flex;justify-content:space-between;gap:5px;font-size:{fs};line-height:1.5;">'
            f'<span style="color:{sc_col if is_b else "#6b7280"};font-weight:{"700" if is_b else "400"};">{rn}</span>'
            f'<span style="color:{sc_col};font-weight:{"700" if is_b else "400"};">{int(sc)}</span></div>')
    return f'<div style="margin-top:3px;padding-top:3px;border-top:1px solid #111827;">{"".join(lines)}</div>'

def best_role_html(player:dict, df_sc, fs:str="8px")->str:
    if df_sc is None or df_sc.empty: return ""
    rows = df_sc[df_sc["Player"]==player.get("Player","")]
    if rows.empty: return ""
    row=rows.iloc[0]; rk=_role_key(player.get("Position",""))
    scores={}
    for rn in ROLE_BUCKETS.get(rk,{}):
        v=row.get(f"_rs_{rn}",np.nan)
        if isinstance(v,(int,float)) and not np.isnan(float(v)): scores[rn]=float(v)
    if not scores: return ""
    best=max(scores,key=scores.get); sc=scores[best]; sc_col=score_to_color(sc)
    return (f'<div style="font-size:{fs};line-height:1.5;margin-top:2px;">'
            f'<span style="color:#6b7280;">{best} </span>'
            f'<span style="color:{sc_col};font-weight:700;">{int(sc)}</span></div>')

# ── Pitch SVG markings ─────────────────────────────────────────────────────────
PORTRAIT_SVG = """
  <rect  x="2"   y="2"     width="96" height="138" fill="none" stroke="#111827" stroke-width=".8"/>
  <line  x1="2"  y1="71"   x2="98"   y2="71"      stroke="#111827" stroke-width=".5"/>
  <circle cx="50" cy="71" r="10"                   fill="none" stroke="#111827" stroke-width=".5"/>
  <circle cx="50" cy="71" r=".9"                   fill="#111827"/>
  <rect  x="22"  y="2"     width="56" height="18"  fill="none" stroke="#111827" stroke-width=".45"/>
  <rect  x="36"  y="2"     width="28" height="7"   fill="none" stroke="#111827" stroke-width=".35"/>
  <circle cx="50" cy="14" r=".65"                  fill="#111827"/>
  <rect  x="22"  y="122"   width="56" height="18"  fill="none" stroke="#111827" stroke-width=".45"/>
  <rect  x="36"  y="133"   width="28" height="7"   fill="none" stroke="#111827" stroke-width=".35"/>
  <circle cx="50" cy="126" r=".65"                 fill="#111827"/>"""

# ── Render pitch ───────────────────────────────────────────────────────────────
def render_pitch(
    team:str, league:str, formation:str,
    slots:list, slot_map:dict, depth:list, df_sc,
    show_mins:bool, show_goals:bool, show_assists:bool,
    show_roles:bool, show_depth:bool, canva:bool,
) -> str:

    # ── #1: Canva = portrait pitch centred in 1920×1080 slide, NO rotation ──
    # ── #10: normal view has bigger fonts; canva keeps compact ──
    if canva:
        # Canva: portrait pitch centred in a landscape wrapper
        # Outer div will be 16:9; pitch sits in middle column
        badge_sz = "11px"; nm_sz = "12px"; stat_sz = "8px"; role_sz = "7.5px"
        dep_nm   = "10px"; dep_rs = "7.5px"
        aspect   = "142%"   # portrait pitch
        wrapper_style = ("display:flex;justify-content:center;align-items:flex-start;"
                         "background:#0a0f1c;aspect-ratio:16/9;overflow:hidden;padding:20px 0;")
        inner_style   = "width:38%;min-width:340px;"   # narrow column centred
    else:
        # Normal app view: +3 badge, +2 name vs canva
        badge_sz = "15px"; nm_sz = "14px"; stat_sz = "9px"; role_sz = "8px"
        dep_nm   = "12px"; dep_rs = "8px"
        aspect   = "142%"
        wrapper_style = ""
        inner_style   = ""

    nodes = ""
    for slot in slots:
        ps = slot_map.get(slot["id"],[])
        badge = (f'<div style="display:inline-block;padding:3px 10px;'
                 f'border:2px solid #ef4444;color:#ef4444;font-size:{badge_sz};'
                 f'font-weight:900;letter-spacing:.1em;margin-bottom:4px;'
                 f'background:rgba(10,15,28,.92);">{slot["label"]}</div>')
        rows=""
        for i,p in enumerate(ps):
            yrs     = contract_years(p.get("Contract expires",""))
            yr_str  = f"+{yrs}" if yrs>=0 else "+?"
            loan    = is_loan(p)
            col     = player_css_color(yrs,loan)
            fw      = "800" if i==0 else "500"
            multi   = " 🔁" if _multi_role(p.get("Position","")) else ""   # #6
            loan_t  = " (L)" if loan else ""
            oop_tag = f" ({p.get('_primary_pos','')})" if p.get("_oop") else ""  # #5
            suffix  = f"{yr_str}{loan_t}{oop_tag}{multi}"

            stat_parts=[]
            if show_mins:   stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}′")
            if show_goals:
                g=float(p.get("Goals") or 0)
                if g>0: stat_parts.append(f"{int(g)}⚽")
            if show_assists:
                a=float(p.get("Assists") or 0)
                if a>0: stat_parts.append(f"{int(a)}🅰")
            # #3 stats in white
            stat_html = (f'<div style="color:#ffffff;font-size:{stat_sz};line-height:1.2;opacity:.8;">'
                         f'{" ".join(stat_parts)}</div>') if stat_parts else ""

            rs_html = (all_roles_html(p,df_sc,role_sz) if (i==0 and show_roles)
                       else best_role_html(p,df_sc,role_sz) if (i>0 and show_roles) else "")

            rows += (f'<div style="color:{col};font-size:{nm_sz};line-height:1.5;'
                     f'font-weight:{fw};white-space:nowrap;text-shadow:0 1px 6px rgba(0,0,0,.9);">'
                     f'{p["Player"]} {suffix}</div>{stat_html}{rs_html}')

        if not ps:
            rows = f'<div style="color:#1f2937;font-size:{stat_sz};">—</div>'

        nodes += (f'<div style="position:absolute;left:{slot["x"]}%;top:{slot["y"]}%;'
                  f'transform:translate(-50%,-50%);text-align:center;min-width:88px;z-index:10;">'
                  f'{badge}<div>{rows}</div></div>')

    # Depth section
    depth_html=""
    if show_depth and depth:
        cards=""
        for p in depth:
            yrs=contract_years(p.get("Contract expires","")); yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); col=player_css_color(yrs,loan)
            multi="🔁" if _multi_role(p.get("Position","")) else ""
            pos_t=_tok(p.get("Position",""))
            br=best_role_html(p,df_sc,dep_rs) if show_roles else ""
            cards += (f'<div style="background:#0d1220;border:1px solid #111827;'
                      f'padding:5px 9px;min-width:100px;text-align:center;flex-shrink:0;">'
                      f'<div style="color:{col};font-size:{dep_nm};font-weight:700;">'
                      f'{p["Player"]} {yr_str} {multi}</div>'
                      f'<div style="color:#374151;font-size:7px;">{pos_t}</div>{br}</div>')
        depth_html=(f'<div style="margin-top:12px;border-top:1px solid #111827;padding-top:8px;">'
                    f'<div style="font-size:9px;font-weight:800;letter-spacing:.18em;color:#374151;'
                    f'margin-bottom:6px;text-align:center;">DEPTH</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">'
                    f'{cards}</div></div>')

    legend_stats=""
    if show_mins:    legend_stats+=" · ′=mins"
    if show_goals:   legend_stats+=" · ⚽=goals"
    if show_assists: legend_stats+=" · 🅰=assists"

    # ── #10: Title = "Team Squad Depth" for normal; none for Canva ──
    title_html = ("" if canva else
                  f'<div style="font-weight:900;font-size:22px;letter-spacing:.05em;'
                  f'text-transform:uppercase;text-align:center;margin-bottom:6px;">'
                  f'{team} Squad Depth</div>')

    header_html=(f'<div style="display:flex;justify-content:space-between;'
                 f'align-items:baseline;margin-bottom:6px;font-size:10px;color:#1f2937;">'
                 f'<span>{league}</span><span>{formation}</span></div>')

    legend_bar=(f'<div style="text-align:center;font-size:8px;color:#1f2937;margin-top:8px;'
                f'letter-spacing:.07em;">Name + contract years{legend_stats} · 🔁=3+ positions</div>'
                f'<div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;'
                f'font-size:10px;font-weight:700;margin-top:6px;">'
                f'<span style="color:#fff;">Contracted</span>'
                f'<span style="color:#f59e0b;">Final Year</span>'
                f'<span style="color:#ef4444;">Out of Contract</span>'
                f'<span style="color:#22c55e;">On Loan</span></div>')

    pitch_block = f"""
<div style="font-family:Montserrat,sans-serif;color:#fff;background:#0a0f1c;padding:0 4px 12px;{inner_style}">
  {title_html}
  {header_html}
  <div style="position:relative;background:#0a0f1c;padding-bottom:{aspect};overflow:hidden;border:1px solid #0d1220;">
    <svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"
         viewBox="0 0 100 142" preserveAspectRatio="none">{PORTRAIT_SVG}
    </svg>
    {nodes}
  </div>
  {depth_html}
  {legend_bar}
</div>"""

    # ── #1: Canva wraps the portrait pitch in a 16:9 landscape div ──
    if canva:
        return f'<div style="{wrapper_style}">{pitch_block}</div>'
    return pitch_block

# ── Image download via html2canvas (client-side) ──────────────────────────────
CAPTURE_JS = """
<script>
async function downloadPNG() {
  const btn = document.getElementById('png-btn');
  btn.textContent = 'Capturing…';
  // load html2canvas dynamically
  await new Promise(r => {
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
    s.onload = r; document.head.appendChild(s);
  });
  // find the pitch div (first child of stMarkdown that has Montserrat font)
  const el = document.querySelector('[style*="Montserrat"]');
  if (!el) { btn.textContent = '⚠ Not found'; return; }
  const canvas = await html2canvas(el, {backgroundColor:'#0a0f1c', scale:2, useCORS:true});
  const link = document.createElement('a');
  link.download = 'squad_depth_chart.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
  btn.textContent = '⬇ Download PNG';
}
</script>
<button id="png-btn"
  onclick="downloadPNG()"
  style="background:#fff;color:#0a0f1c;font-weight:700;font-family:Montserrat,sans-serif;
         font-size:11px;letter-spacing:.06em;text-transform:uppercase;border:none;
         padding:6px 16px;cursor:pointer;border-radius:2px;">
  ⬇ Download PNG
</button>
"""

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in {"slot_map":{},"depth":[],"move_player":None,"df":None,"df_sc":None,
             "last_team":None,"last_formation":None,"edit_contract_player":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def _tog(k,d=False): return st.session_state.get(k,d)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Squad Chart")
    st.markdown("---")
    st.markdown("**DATA**")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        @st.cache_data
        def _load(f)->pd.DataFrame:
            df=pd.read_csv(f)
            df.columns=df.columns.str.strip()
            for c in ["Player","Team","Position","League"]:
                if c in df.columns: df[c]=df[c].astype(str).str.strip()
            for c in ["Minutes played","Goals","Assists","Age","xG","xA"]:
                if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
            df["_ftok"]=df["Position"].apply(_tok)
            df["_key"]=df["Player"]
            return df
        raw=_load(uploaded)
        if st.session_state.df is None or len(raw)!=len(st.session_state.df):
            st.session_state.df=raw; st.session_state.df_sc=None
        if st.session_state.df_sc is None:
            with st.spinner("Computing role scores…"):
                st.session_state.df_sc=compute_role_scores(st.session_state.df)
        st.success(f"✓ {len(st.session_state.df):,} players · role scores ready")

    st.markdown("---")

    if st.session_state.df is not None:
        df=st.session_state.df
        lgs=["All"]+sorted(df["League"].unique())
        lg=st.selectbox("League",lgs)
        fdf=df if lg=="All" else df[df["League"]==lg]

        # ── #7: Minutes filter ──
        max_mins=int(df["Minutes played"].max()) if "Minutes played" in df.columns else 5000
        min_mins=st.slider("Min minutes played",0,max_mins,0,50)
        fdf=fdf[fdf["Minutes played"]>=min_mins]

        sel_team  = st.selectbox("Team", sorted(fdf["Team"].unique()))
        formation = st.selectbox("Formation", list(FORMATIONS.keys()))

        st.markdown("---")
        st.markdown("**DISPLAY**")
        st.toggle("Minutes played", True,  key="show_mins")
        st.toggle("Goals",          True,  key="show_goals")
        st.toggle("Assists",        True,  key="show_assists")
        st.toggle("Role scores",    True,  key="show_roles")
        st.toggle("Show depth",     True,  key="show_depth")   # #8
        st.toggle("Canva 1920×1080",False, key="canva_mode")   # #1

        st.markdown("---")
        changed=(sel_team!=st.session_state.last_team or
                 formation!=st.session_state.last_formation)
        if st.button("🔄 Build / Rebuild") or changed:
            tdf=fdf[fdf["Team"]==sel_team].copy()
            tdf["_key"]=tdf["Player"]
            sm,dep=assign_players(tdf.to_dict("records"),formation)
            st.session_state.slot_map=sm; st.session_state.depth=dep
            st.session_state.last_team=sel_team; st.session_state.last_formation=formation
            st.session_state.move_player=None

        # Move panel
        if st.session_state.move_player:
            mp=st.session_state.move_player
            st.markdown(f"**MOVING:** {mp['player']['Player']}")
            opts={f"{s['label']} ({s['id']})":s["id"] for s in FORMATIONS[formation]}
            dest_lbl=st.selectbox("Move to",list(opts.keys()))
            if st.button("✅ Confirm"):
                p=mp["player"]; fid=mp["from_slot"]; did=opts[dest_lbl]
                if fid=="_depth": st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=p["_key"]]
                elif fid in st.session_state.slot_map:
                    st.session_state.slot_map[fid]=[x for x in st.session_state.slot_map[fid] if x["_key"]!=p["_key"]]
                st.session_state.slot_map.setdefault(did,[]).append(p)
                st.session_state.move_player=None; st.rerun()
            if st.button("❌ Cancel"):
                st.session_state.move_player=None; st.rerun()

        # ── #9: Edit contract ──
        if st.session_state.edit_contract_player:
            ec=st.session_state.edit_contract_player
            st.markdown(f"**EDIT CONTRACT:** {ec['player']['Player']}")
            new_exp=st.text_input("Expires (YYYY-MM-DD)",ec["player"].get("Contract expires",""),key="new_exp")
            if st.button("💾 Save Contract"):
                pk=ec["player"]["_key"]
                for sid,ps in st.session_state.slot_map.items():
                    for p in ps:
                        if p["_key"]==pk: p["Contract expires"]=new_exp
                for p in st.session_state.depth:
                    if p["_key"]==pk: p["Contract expires"]=new_exp
                st.session_state.edit_contract_player=None; st.rerun()
            if st.button("✖ Cancel edit"):
                st.session_state.edit_contract_player=None; st.rerun()

        st.markdown("---")
        st.markdown("**ADD PLAYER**")
        nn=st.text_input("Name",key="nn")
        np_=st.selectbox("Position",list(CANONICAL.keys()),key="np_")
        nm_=st.number_input("Minutes",0,5000,0,10,key="nm_")
        ng_=st.number_input("Goals",0,100,0,key="ng_")
        na_=st.number_input("Assists",0,100,0,key="na_")
        ne_=st.text_input("Contract expires","2026-06-30",key="ne_")
        nl_=st.checkbox("On Loan?",key="nl_")
        sl_opts={f"{s['label']} ({s['id']})":s["id"] for s in FORMATIONS.get(formation,[])}
        ns_=st.selectbox("Add to slot",list(sl_opts.keys()),key="ns_")
        if st.button("➕ Add") and nn.strip():
            new_p={"Player":nn.strip(),"Position":np_,"_key":f"custom_{nn}",
                   "Minutes played":nm_,"Goals":ng_,"Assists":na_,
                   "Contract expires":ne_,"On Loan":"yes" if nl_ else "no",
                   "League":lg,"Team":sel_team}
            st.session_state.slot_map.setdefault(sl_opts[ns_],[]).append(new_p)
            st.rerun()
    else:
        st.info("Upload a CSV to get started.")

# ── Main ───────────────────────────────────────────────────────────────────────
if not st.session_state.slot_map:
    st.markdown("<div style='text-align:center;color:#0d1220;font-size:11px;"
                "padding:120px 20px;border:1px dashed #0d1220;letter-spacing:.12em;'>"
                "UPLOAD A CSV AND SELECT A TEAM TO GET STARTED</div>",
                unsafe_allow_html=True)
    st.stop()

formation=st.session_state.last_formation or "4-2-3-1"
team_name=st.session_state.last_team or ""
league_nm=""
if st.session_state.df is not None and team_name:
    tdf2=st.session_state.df[st.session_state.df["Team"]==team_name]
    if not tdf2.empty and "League" in tdf2.columns:
        league_nm=tdf2["League"].iloc[0]

slots=FORMATIONS[formation]; slot_map=st.session_state.slot_map
depth=st.session_state.depth; df_sc=st.session_state.df_sc

pitch=render_pitch(
    team_name,league_nm,formation,slots,slot_map,depth,df_sc,
    _tog("show_mins",True),_tog("show_goals",True),_tog("show_assists",True),
    _tog("show_roles",True),_tog("show_depth",True),_tog("canva_mode"),
)

st.markdown(pitch, unsafe_allow_html=True)

# ── Downloads ──────────────────────────────────────────────────────────────────
canva_w=_tog("canva_mode")
page_w="1920px" if canva_w else "960px"
html_dl=f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{team_name} Squad Depth</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
body{{margin:0;background:#0a0f1c;font-family:Montserrat,sans-serif}}
.wrap{{width:{page_w};margin:0 auto;padding:20px}}
</style></head><body><div class="wrap">{pitch}</div></body></html>"""

dl1,dl2,_ = st.columns([1,1,4])
with dl1:
    st.download_button("⬇ HTML", html_dl.encode("utf-8"),
        f"{team_name.replace(' ','_')}_squad_depth.html","text/html")
with dl2:
    # ── #2: PNG download via html2canvas ──
    st.markdown(CAPTURE_JS, unsafe_allow_html=True)

# ── Move / Remove / Edit Contract ─────────────────────────────────────────────
st.markdown("---")
all_on=[]
for sl in slots:
    for p in slot_map.get(sl["id"],[]):
        all_on.append({"sid":sl["id"],"lbl":sl["label"],"player":p})
for p in depth:
    all_on.append({"sid":"_depth","lbl":"DEPTH","player":p})

if all_on:
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div style='font-size:9px;color:#374151;letter-spacing:.1em;margin-bottom:3px;'>MOVE</div>",unsafe_allow_html=True)
        mv_opts={f"{e['player']['Player']} ({e['lbl']})":e for e in all_on}
        mv_sel=st.selectbox("",list(mv_opts.keys()),key="mv_sel",label_visibility="collapsed")
        if st.button("Select for Move"):
            e=mv_opts[mv_sel]
            st.session_state.move_player={"player":e["player"],"from_slot":e["sid"]}; st.rerun()
    with c2:
        st.markdown("<div style='font-size:9px;color:#374151;letter-spacing:.1em;margin-bottom:3px;'>REMOVE</div>",unsafe_allow_html=True)
        rm_opts={f"{e['player']['Player']} ({e['lbl']})":e for e in all_on}
        rm_sel=st.selectbox("",list(rm_opts.keys()),key="rm_sel",label_visibility="collapsed")
        if st.button("🗑 Remove"):
            e=rm_opts[rm_sel]; sid=e["sid"]; pk=e["player"]["_key"]
            if sid=="_depth": st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=pk]
            else: st.session_state.slot_map[sid]=[x for x in st.session_state.slot_map.get(sid,[]) if x["_key"]!=pk]
            st.rerun()
    with c3:
        st.markdown("<div style='font-size:9px;color:#374151;letter-spacing:.1em;margin-bottom:3px;'>EDIT CONTRACT</div>",unsafe_allow_html=True)
        ec_opts={f"{e['player']['Player']} ({e['lbl']})":e for e in all_on}
        ec_sel=st.selectbox("",list(ec_opts.keys()),key="ec_sel",label_visibility="collapsed")
        if st.button("✏️ Edit Contract"):
            e=ec_opts[ec_sel]
            st.session_state.edit_contract_player={"player":e["player"],"sid":e["sid"]}; st.rerun()

# ── Full squad ─────────────────────────────────────────────────────────────────
if st.session_state.df is not None and team_name:
    with st.expander("📋 Full Squad"):
        tdf3=st.session_state.df[st.session_state.df["Team"]==team_name]
        show_c=[c for c in ["Player","Position","Minutes played","Goals","Assists",
                             "Market value","Contract expires","Age"] if c in tdf3.columns]
        st.dataframe(tdf3[show_c].sort_values("Minutes played",ascending=False).reset_index(drop=True),
                     use_container_width=True)
