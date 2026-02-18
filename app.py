"""
Squad Depth Chart — v7
pip install streamlit pandas numpy pillow
streamlit run app.py
"""
import re, io, base64
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Squad Depth Chart", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;background:#0a0f1c!important;color:#fff!important}
.stApp{background:#0a0f1c!important}
section[data-testid="stSidebar"]{background:#060a14!important;border-right:1px solid #0d1220!important}
section[data-testid="stSidebar"] *{color:#fff!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important}
.stSelectbox>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important}
div[data-baseweb="select"]*{background:#0d1424!important;color:#fff!important}
div[data-baseweb="popover"]*{background:#0d1424!important;color:#fff!important}
.stTextInput>div>div>input,.stNumberInput input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important}
/* #7: white bg buttons → BLACK text */
.stButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  letter-spacing:.06em!important;text-transform:uppercase!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;border-radius:2px!important}
.stButton>button:hover{background:#e0e0e0!important;color:#000000!important}
.stDownloadButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  letter-spacing:.06em!important;text-transform:uppercase!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;border-radius:2px!important}
.stDownloadButton>button:hover{background:#e0e0e0!important;color:#000000!important}
label{color:#4b5563!important;font-size:9px!important;letter-spacing:.14em!important;text-transform:uppercase!important}
h1,h2,h3{color:#fff!important;font-family:'Montserrat',sans-serif!important}
footer{display:none!important}
hr{border-color:#0d1220!important}
.streamlit-expanderHeader{background:#0d1424!important;color:#fff!important}
</style>
""", unsafe_allow_html=True)

# ── Role Buckets ───────────────────────────────────────────────────────────────
ROLE_BUCKETS: dict[str,dict] = {
    "GK":{
        "Shot Stopper GK":{"metrics":{"Prevented goals per 90":3,"Save rate, %":1}},
        "Ball Playing GK":{"metrics":{"Passes per 90":1,"Accurate passes, %":3,"Accurate long passes, %":2}},
        "Sweeper GK":     {"metrics":{"Exits per 90":1}},
    },
    "CB":{
        "Ball Playing CB":{"metrics":{"Passes per 90":2,"Accurate passes, %":2,"Forward passes per 90":2,
            "Accurate forward passes, %":2,"Progressive passes per 90":2,"Progressive runs per 90":1.5,
            "Dribbles per 90":1.5,"Accurate long passes, %":1,"Passes to final third per 90":1.5}},
        "Wide CB":        {"metrics":{"Defensive duels per 90":1.5,"Defensive duels won, %":2,
            "Dribbles per 90":2,"Forward passes per 90":1,"Progressive passes per 90":1,"Progressive runs per 90":2}},
        "Box Defender":   {"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":3,
            "PAdj Interceptions":2,"Shots blocked per 90":1,"Defensive duels won, %":4}},
    },
    "FB":{
        "Build Up FB":  {"metrics":{"Passes per 90":2,"Accurate passes, %":1.5,"Forward passes per 90":2,
            "Accurate forward passes, %":2,"Progressive passes per 90":2.5,"Progressive runs per 90":2,
            "Dribbles per 90":2,"Passes to final third per 90":2,"xA per 90":1}},
        "Attacking FB": {"metrics":{"Crosses per 90":2,"Dribbles per 90":3.5,"Accelerations per 90":1,
            "Successful dribbles, %":1,"Touches in box per 90":2,"Progressive runs per 90":3,
            "Passes to penalty area per 90":2,"xA per 90":3}},
        "Defensive FB": {"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":1.5,
            "Defensive duels per 90":2,"PAdj Interceptions":3,"Shots blocked per 90":1,"Defensive duels won, %":3.5}},
    },
    "CM":{
        "Deep Playmaker CM":    {"metrics":{"Passes per 90":1,"Accurate passes, %":1,"Forward passes per 90":2,
            "Accurate forward passes, %":1.5,"Progressive passes per 90":3,"Passes to final third per 90":2.5,
            "Accurate long passes, %":1}},
        "Advanced Playmaker CM":{"metrics":{"Deep completions per 90":1.5,"Smart passes per 90":2,
            "xA per 90":4,"Passes to penalty area per 90":2}},
        "Defensive CM":         {"metrics":{"Defensive duels per 90":4,"Defensive duels won, %":4,
            "PAdj Interceptions":3,"Aerial duels per 90":0.5,"Aerial duels won, %":1}},
        "Ball Carrying CM":     {"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "ATT":{
        "Playmaker ATT":   {"metrics":{"Passes per 90":2,"xA per 90":3,"Key passes per 90":1,
            "Deep completions per 90":1.5,"Smart passes per 90":1.5,"Passes to penalty area per 90":2}},
        "Goal Threat ATT": {"metrics":{"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,"Touches in box per 90":2}},
        "Ball Carrier ATT":{"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,
            "Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "CF":{
        "Target Man CF":  {"metrics":{"Aerial duels per 90":3,"Aerial duels won, %":5}},
        "Goal Threat CF": {"metrics":{"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,
            "Touches in box per 90":1,"Shots on target, %":0.5}},
        "Link Up CF":     {"metrics":{"Passes per 90":2,"Passes to penalty area per 90":1.5,
            "Deep completions per 90":1,"Smart passes per 90":1.5,"Accurate passes, %":1.5,
            "Key passes per 90":1,"Dribbles per 90":2,"Successful dribbles, %":1,
            "Progressive runs per 90":2,"xA per 90":3}},
    },
}
ROLE_KEY_MAP:dict[str,str]={
    "GK":"GK","CB":"CB","LCB":"CB","RCB":"CB",
    "LB":"FB","RB":"FB","LWB":"FB","RWB":"FB",
    "DMF":"CM","LDMF":"CM","RDMF":"CM","LCMF":"CM","RCMF":"CM",
    "AMF":"ATT","LAMF":"ATT","LW":"ATT","LWF":"ATT","RAMF":"ATT","RW":"ATT","RWF":"ATT",
    "CF":"CF",
}
POS_POOL_MAP:dict[str,list]={
    "GK":["GK"],"CB":["CB","LCB","RCB"],"FB":["LB","RB","LWB","RWB"],
    "CM":["DMF","LDMF","RDMF","LCMF","RCMF"],
    "ATT":["AMF","LAMF","RAMF","LW","LWF","RW","RWF"],"CF":["CF"],
}
CANONICAL:dict[str,str]={
    "GK":"GK","CB":"CB","LCB":"LCB","RCB":"RCB",
    "LB":"LB","LWB":"LWB","RB":"RB","RWB":"RWB",
    "DMF":"DM","LDMF":"DM","RDMF":"DM","LCMF":"CM","RCMF":"CM",
    "AMF":"AM","LAMF":"LW","LW":"LW","LWF":"LW",
    "RAMF":"RW","RW":"RW","RWF":"RW","CF":"ST",
}
SIDE_PREF:dict[str,str]={
    "RCB":"R","RCMF":"R","RDMF":"R","RB":"R","RWB":"R","RW":"R","RWF":"R","RAMF":"R",
    "LCB":"L","LCMF":"L","LDMF":"L","LB":"L","LWB":"L","LW":"L","LWF":"L","LAMF":"L",
}

def _tok(pos:str)->str:   return str(pos).split(",")[0].strip().upper()
def _canon(pos:str)->str: return CANONICAL.get(_tok(pos),"CM")
def _side(pos:str)->str:  return SIDE_PREF.get(_tok(pos),"N")
def _role_key(pos:str)->str: return ROLE_KEY_MAP.get(_tok(pos),"ATT")
def _all_toks(pos:str)->list: return [t.strip().upper() for t in str(pos).split(",") if t.strip()]
# #1: emoji at 4+ positions
def _multi_role(pos:str)->bool: return len(_all_toks(pos))>=4

# ── Formations ─────────────────────────────────────────────────────────────────
# Portrait coords: x=left-right%, y=top-bottom% (y=9=attack top, y=84=GK bottom)
# #8: 4-3-3: CM(L,y=36) + AM(R,y=36) ABOVE DM(centre,y=46)
# #8: 3-5-2: AM(L,y=36) + AM(R,y=36) ABOVE DM(centre,y=46)
FORMATIONS:dict[str,list[dict]]={
    "4-2-3-1":[
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
    "4-3-3":[
        {"id":"ST",  "label":"ST",  "x":50,"y":9,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":14,"y":16, "accepts":["LW"],             "side":"L"},
        {"id":"RW",  "label":"RW",  "x":86,"y":16, "accepts":["RW"],             "side":"R"},
        {"id":"CM",  "label":"CM",  "x":30,"y":36, "accepts":["CM","AM"],        "side":"L"},
        {"id":"AM",  "label":"AM",  "x":70,"y":36, "accepts":["AM","CM"],        "side":"R"},
        {"id":"DM",  "label":"DM",  "x":50,"y":46, "accepts":["DM","CM"],        "side":"N"},
        {"id":"LB",  "label":"LB",  "x":9, "y":63, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":67, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":67, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":91,"y":63, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":84, "accepts":["GK"],             "side":"N"},
    ],
    "4-4-2":[
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
    "3-5-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":9,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":9,  "accepts":["ST"],             "side":"R"},
        {"id":"LWB", "label":"LWB", "x":9, "y":32, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"AM1", "label":"AM",  "x":30,"y":36, "accepts":["AM","CM"],        "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":46, "accepts":["DM","CM"],        "side":"N"},
        {"id":"AM2", "label":"AM",  "x":70,"y":36, "accepts":["AM","CM"],        "side":"R"},
        {"id":"RWB", "label":"RWB", "x":91,"y":32, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":62, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":66, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":62, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":83, "accepts":["GK"],             "side":"N"},
    ],
    "3-4-1-2":[
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
    "4-5-1":[
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
    "4-1-4-1":[
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

PITCH_ORDER=["GK","LCB","CB","RCB","LB","RB","LWB","RWB","DM","CM","AM","LW","RW","ST"]

# ── Landscape: convert portrait % coords to fixed pixel coords (1920×1080)
# Pitch area: 1800×880px, offset (60, 100) within 1920×1080
# GK(portrait y=84) → landscape left (small lx); ST(y=9) → landscape right
CANVA_W,CANVA_H = 1920,1080
PITCH_PX_W,PITCH_PX_H = 1800,840
PITCH_PX_X,PITCH_PX_Y = 60,120   # pitch offset within canva div
CANVA_Y_MIN,CANVA_Y_MAX = 7.0,87.0

def landscape_px(ox:float,oy:float)->tuple[int,int]:
    """Portrait % → landscape absolute pixel coords within 1920×1080"""
    lx_pct=95-(oy-CANVA_Y_MIN)/(CANVA_Y_MAX-CANVA_Y_MIN)*90
    ly_pct=ox
    px=PITCH_PX_X+lx_pct/100*PITCH_PX_W
    py=PITCH_PX_Y+ly_pct/100*PITCH_PX_H
    return round(px),round(py)

# ── Helpers ────────────────────────────────────────────────────────────────────
def contract_years(s)->int:
    s=str(s or "").strip()
    if s in ("","nan","NaT"): return -1
    m=re.search(r"(20\d{2})",s)
    return max(0,int(m.group(1))-date.today().year) if m else -1

def is_loan(p:dict)->bool:
    for k in ("On loan","On Loan","on_loan","Loan","loan","On loan?"):
        if k in p and str(p[k]).strip().lower() in ("yes","y","true","1","on loan"):
            return True
    return False

def player_css_color(yrs:int,loan:bool)->str:
    if loan:   return "#22c55e"
    if yrs==0: return "#ef4444"
    if yrs==1: return "#f59e0b"
    return "#ffffff"

def score_to_color(v:float)->str:
    if np.isnan(v): return "#4b5563"
    v=max(0.0,min(100.0,float(v)))
    if v<=50:
        t=v/50; r=int(239+(234-239)*t); g=int(68+(179-68)*t); b=int(68+(8-68)*t)
    else:
        t=(v-50)/50; r=int(234+(34-234)*t); g=int(179+(197-179)*t); b=int(8+(94-8)*t)
    return f"rgb({r},{g},{b})"

# ── Role scores ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_role_scores(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    skip={"Player","League","Team","Position","Age","Market value","Contract expires",
          "Matches played","Minutes played","Goals","Assists","xG","xA",
          "Birth country","Foot","Height","_ftok","_key"}
    for c in df.columns:
        if c not in skip and not c.startswith("On ") and "loan" not in c.lower():
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    for rk,pool_pos in POS_POOL_MAP.items():
        for role_name,spec in ROLE_BUCKETS.get(rk,{}).items():
            col_out=f"_rs_{role_name}"; df[col_out]=np.nan
            metrics=spec.get("metrics",{})
            for league in df["League"].unique():
                mask=(df["League"]==league)&(df["_ftok"].isin(pool_pos))&(df["Minutes played"]>=200)
                pool=df[mask]
                if pool.empty: continue
                pcts={}
                for met in metrics:
                    if met in pool.columns:
                        pcts[met]=pd.to_numeric(pool[met],errors="coerce").rank(pct=True,method="average")*100.0
                for idx in pool.index:
                    vals,wts=[],[]
                    for met,w in metrics.items():
                        if met in pcts and idx in pcts[met].index:
                            pv=pcts[met].loc[idx]
                            if not np.isnan(pv): vals.append(float(pv)); wts.append(float(w))
                    if vals: df.at[idx,col_out]=float(np.average(vals,weights=wts))
    return df

# ── Player assignment ──────────────────────────────────────────────────────────
def assign_players(players:list,formation_key:str)->tuple[dict,list]:
    slots=FORMATIONS.get(formation_key,FORMATIONS["4-2-3-1"])
    by_label:dict[str,list]={}
    for s in slots: by_label.setdefault(s["label"],[]).append(s)
    assigned:set=set()
    slot_map:dict[str,list]={s["id"]:[] for s in slots}

    def primary_fits(p,slot):
        tok=_tok(p.get("Position",""))
        if slot.get("wb_only"):
            return tok in {"LB","LWB","RB","RWB"} and _canon(p.get("Position","")) in slot["accepts"]
        return _canon(p.get("Position","")) in slot["accepts"]

    def secondary_fits(p,slot):
        if slot.get("wb_only"): return False
        for t in _all_toks(p.get("Position",""))[1:]:
            if CANONICAL.get(t,"CM") in slot["accepts"]: return True
        return False

    def side_score(p,ss):
        ps=_side(p.get("Position",""))
        if ss=="N" or ps=="N": return 1
        return 0 if ps==ss else 2

    for label in PITCH_ORDER:
        if label not in by_label: continue
        slot_list=by_label[label]
        matched=[p for p in players if p["_key"] not in assigned
                 and any(primary_fits(p,s) for s in slot_list)]
        if not matched:
            matched=[p for p in players if p["_key"] not in assigned
                     and any(secondary_fits(p,s) for s in slot_list)]
        matched.sort(key=lambda p:-float(p.get("Minutes played") or 0))
        for p in matched: assigned.add(p["_key"])
        n=len(slot_list)
        if n==1:
            slot_map[slot_list[0]["id"]]=matched
        else:
            ordered=sorted(slot_list,key=lambda s:{"L":0,"N":1,"R":2}[s["side"]])
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

    for sid,ps in slot_map.items():
        slot_def=next((s for s in slots if s["id"]==sid),None)
        for p in ps:
            p["_oop"]=not primary_fits(p,slot_def) if slot_def else False
            p["_primary_pos"]=_tok(p.get("Position",""))

    depth=[p for p in players if p["_key"] not in assigned]
    depth.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    return slot_map,depth

# ── Score HTML ─────────────────────────────────────────────────────────────────
def all_roles_html(player,df_sc,fs="8px"):
    if df_sc is None or df_sc.empty: return ""
    rows=df_sc[df_sc["Player"]==player.get("Player","")]
    if rows.empty: return ""
    row=rows.iloc[0]; rk=_role_key(player.get("Position",""))
    scores={}
    for rn in ROLE_BUCKETS.get(rk,{}):
        v=row.get(f"_rs_{rn}",np.nan)
        if isinstance(v,(int,float)) and not np.isnan(float(v)): scores[rn]=float(v)
    if not scores: return ""
    best=max(scores,key=scores.get); lines=[]
    for rn,sc in sorted(scores.items(),key=lambda x:-x[1]):
        sc_col=score_to_color(sc); is_b=rn==best
        lines.append(
            f'<div style="display:flex;justify-content:space-between;gap:5px;font-size:{fs};line-height:1.5;">'
            f'<span style="color:{sc_col if is_b else "#6b7280"};font-weight:{"700" if is_b else "400"};">{rn}</span>'
            f'<span style="color:{sc_col};font-weight:{"700" if is_b else "400"};">{int(sc)}</span></div>')
    return f'<div style="margin-top:2px;">{"".join(lines)}</div>'

def best_role_html(player,df_sc,fs="8px"):
    if df_sc is None or df_sc.empty: return ""
    rows=df_sc[df_sc["Player"]==player.get("Player","")]
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

# ── SVG lines: #2 more visible (near-white stroke) ────────────────────────────
PORTRAIT_SVG="""
  <rect  x="2"   y="2"     width="96" height="138" fill="none" stroke="#9ca3af" stroke-width="1.2"/>
  <line  x1="2"  y1="71"   x2="98"   y2="71"      stroke="#9ca3af" stroke-width=".8"/>
  <circle cx="50" cy="71" r="10"                   fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <circle cx="50" cy="71" r="1.2"                  fill="#9ca3af"/>
  <rect  x="22"  y="2"     width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <rect  x="36"  y="2"     width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"/>
  <circle cx="50" cy="14" r=".9"                   fill="#9ca3af"/>
  <rect  x="22"  y="122"   width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <rect  x="36"  y="133"   width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"/>
  <circle cx="50" cy="126" r=".9"                   fill="#9ca3af"/>"""

# Landscape SVG: pitch runs left (GK) to right (ST), standard football landscape
# viewBox="0 0 142 100" matches the div's aspect ratio approach
LANDSCAPE_SVG="""
  <rect  x="2"   y="2"     width="138" height="96" fill="none" stroke="#9ca3af" stroke-width="1.2"/>
  <line  x1="71" y1="2"    x2="71"    y2="98"     stroke="#9ca3af" stroke-width=".8"/>
  <circle cx="71" cy="50" r="10"                   fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <circle cx="71" cy="50" r="1.2"                  fill="#9ca3af"/>
  <rect  x="2"   y="22"    width="18"  height="56" fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <rect  x="2"   y="36"    width="7"   height="28" fill="none" stroke="#9ca3af" stroke-width=".6"/>
  <circle cx="13" cy="50" r=".9"                   fill="#9ca3af"/>
  <rect  x="122" y="22"    width="18"  height="56" fill="none" stroke="#9ca3af" stroke-width=".8"/>
  <rect  x="133" y="36"    width="7"   height="28" fill="none" stroke="#9ca3af" stroke-width=".6"/>
  <circle cx="129" cy="50" r=".9"                  fill="#9ca3af"/>"""

# ── Render pitch ───────────────────────────────────────────────────────────────
def render_pitch(
    team:str, league:str, formation:str,
    slots:list, slot_map:dict, depth:list, df_sc,
    show_mins:bool, show_goals:bool, show_assists:bool,
    show_positions:bool, show_roles:bool, xi_only:bool, canva:bool,
)->str:

    if canva:
        badge_sz="9px"; nm_sz="9px"; stat_sz="6.5px"; role_sz="6px"
    else:
        badge_sz="15px"; nm_sz="14px"; stat_sz="9px"; role_sz="8px"

    # ── Build player node HTML ──────────────────────────────────────────────────
    def make_node(slot, pos_style:str)->str:
        ps_all=slot_map.get(slot["id"],[])
        ps=ps_all[:1] if xi_only else ps_all
        badge=(f'<div style="display:inline-block;padding:2px 8px;'
               f'border:2px solid #ef4444;color:#ef4444;font-size:{badge_sz};'
               f'font-weight:900;letter-spacing:.1em;margin-bottom:3px;'
               f'background:rgba(10,15,28,.95);">{slot["label"]}</div>')
        rows=""
        for i,p in enumerate(ps):
            yrs=contract_years(p.get("Contract expires",""))
            yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); col=player_css_color(yrs,loan)
            fw="800" if i==0 else "500"
            multi=" \U0001f501" if _multi_role(p.get("Position","")) else ""
            loan_t=" (L)" if loan else ""
            oop_tag=f" ({p.get('_primary_pos','')})" if p.get("_oop") else ""
            suffix=f"{yr_str}{loan_t}{oop_tag}{multi}"
            stat_parts=[]
            if show_mins:   stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}\u2032")
            if show_goals:
                g=float(p.get("Goals") or 0)
                if g>0: stat_parts.append(f"{int(g)}\u26bd")
            if show_assists:
                a=float(p.get("Assists") or 0)
                if a>0: stat_parts.append(f"{int(a)}\U0001f170")
            stat_html=(f'<div style="color:#ffffff;font-size:{stat_sz};line-height:1.2;opacity:.9;">'
                       f'{" ".join(stat_parts)}</div>') if stat_parts else ""
            all_pos=", ".join(_all_toks(p.get("Position","")))
            pos_html=(f'<div style="color:#9ca3af;font-size:{stat_sz};line-height:1.2;">'
                      f'{all_pos}</div>') if (show_positions and all_pos) else ""
            rs_html=(all_roles_html(p,df_sc,role_sz) if (i==0 and show_roles)
                     else best_role_html(p,df_sc,role_sz) if (i>0 and show_roles) else "")
            rows+=(f'<div style="color:{col};font-size:{nm_sz};line-height:1.4;'
                   f'font-weight:{fw};white-space:nowrap;text-shadow:0 1px 4px rgba(0,0,0,.95);">'
                   f'{p["Player"]} {suffix}</div>{pos_html}{stat_html}{rs_html}')
        if not ps:
            rows=f'<div style="color:#1f2937;font-size:{stat_sz};">&#8212;</div>'
        return (f'<div style="position:absolute;{pos_style}'
                f'transform:translate(-50%,-50%);text-align:center;min-width:80px;z-index:10;">'
                f'{badge}<div>{rows}</div></div>')

    # ── Portrait mode ───────────────────────────────────────────────────────────
    if not canva:
        nodes="".join(make_node(s,f'left:{s["x"]}%;top:{s["y"]}%;') for s in slots)

        depth_html=""
        if not xi_only and depth:
            dep_nm="11px"; dep_rs="8px"; cards=""
            for p in depth:
                yrs=contract_years(p.get("Contract expires","")); yr_str=f"+{yrs}" if yrs>=0 else "+?"
                loan=is_loan(p); col=player_css_color(yrs,loan)
                multi="\U0001f501" if _multi_role(p.get("Position","")) else ""
                pos_t=_tok(p.get("Position",""))
                br=best_role_html(p,df_sc,dep_rs) if show_roles else ""
                cards+=(f'<div style="background:#0d1220;border:1px solid #1f2937;'
                        f'padding:5px 9px;min-width:100px;text-align:center;flex-shrink:0;">'
                        f'<div style="color:{col};font-size:{dep_nm};font-weight:700;">'
                        f'{p["Player"]} {yr_str} {multi}</div>'
                        f'<div style="color:#6b7280;font-size:7px;">{pos_t}</div>{br}</div>')
            depth_html=(f'<div style="margin-top:10px;border-top:1px solid #1f2937;padding-top:8px;">'
                        f'<div style="font-size:9px;font-weight:800;letter-spacing:.18em;color:#6b7280;'
                        f'margin-bottom:6px;text-align:center;">DEPTH</div>'
                        f'<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">'
                        f'{cards}</div></div>')

        legend_stats=""
        if show_mins:    legend_stats+=" \u00b7 \u2032=mins"
        if show_goals:   legend_stats+=" \u00b7 \u26bd=goals"
        if show_assists: legend_stats+=" \u00b7 \U0001f170=assists"

        title_html=(f'<div style="font-weight:900;font-size:20px;letter-spacing:.05em;'
                    f'text-transform:uppercase;text-align:center;margin-bottom:4px;">'
                    f'{team} Squad Depth</div>')
        header_html=(f'<div style="display:flex;justify-content:space-between;'
                     f'align-items:baseline;margin-bottom:4px;font-size:9px;color:#6b7280;">'
                     f'<span>{league}</span><span>{formation}</span></div>')
        legend_bar=(f'<div style="text-align:center;font-size:8px;color:#6b7280;margin-top:6px;">'
                    f'Name + contract years{legend_stats} \u00b7 \U0001f501=4+ positions</div>'
                    f'<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'
                    f'font-size:9px;font-weight:700;margin-top:4px;">'
                    f'<span style="color:#fff;">Contracted</span>'
                    f'<span style="color:#f59e0b;">Final Year</span>'
                    f'<span style="color:#ef4444;">Out of Contract</span>'
                    f'<span style="color:#22c55e;">On Loan</span></div>')
        return (f'<div id="pitch-root" style="font-family:Montserrat,sans-serif;color:#fff;'
                f'background:#0a0f1c;padding:0 4px 10px;">'
                f'{title_html}{header_html}'
                f'<div style="position:relative;background:#0a0f1c;padding-bottom:142%;'
                f'overflow:hidden;border:1px solid #0d1220;">'
                f'<svg style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"'
                f' viewBox="0 0 100 142" preserveAspectRatio="none">{PORTRAIT_SVG}</svg>'
                f'{nodes}</div>{depth_html}{legend_bar}</div>')

    # ── Canva mode: fixed 1920×1080 px absolute layout ──────────────────────────
    # Convert all slots to absolute pixel positions
    nodes=""
    for slot in slots:
        px,py=landscape_px(float(slot["x"]),float(slot["y"]))
        nodes+=make_node(slot,f'left:{px}px;top:{py}px;')

    # Landscape SVG pitch: fits within PITCH_PX_W × PITCH_PX_H at offset PITCH_PX_X,PITCH_PX_Y
    pw,ph=PITCH_PX_W,PITCH_PX_H; ox,oy=PITCH_PX_X,PITCH_PX_Y
    hw=ph//2+oy   # half-way y
    # penalty areas proportional to full pitch
    pa_h=round(ph*0.36); pa_w=round(pw*0.115)
    ga_h=round(ph*0.2);  ga_w=round(pw*0.05)
    pa_y=oy+round((ph-pa_h)/2); ga_y=oy+round((ph-ga_h)/2)
    cr=round(min(pw,ph)*0.07)
    svg_w=CANVA_W; svg_h=CANVA_H
    landscape_svg=(
        f'<svg style="position:absolute;left:0;top:0;width:{svg_w}px;height:{svg_h}px;pointer-events:none;"'
        f' viewBox="0 0 {svg_w} {svg_h}">'
        # outer pitch rect
        f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#9ca3af" stroke-width="2"/>'
        # half-way line
        f'<line x1="{ox+pw//2}" y1="{oy}" x2="{ox+pw//2}" y2="{oy+ph}" stroke="#9ca3af" stroke-width="1.5"/>'
        # centre circle
        f'<circle cx="{ox+pw//2}" cy="{oy+ph//2}" r="{cr}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>'
        f'<circle cx="{ox+pw//2}" cy="{oy+ph//2}" r="5" fill="#9ca3af"/>'
        # left penalty area
        f'<rect x="{ox}" y="{pa_y}" width="{pa_w}" height="{pa_h}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>'
        f'<rect x="{ox}" y="{ga_y}" width="{ga_w}" height="{ga_h}" fill="none" stroke="#9ca3af" stroke-width="1"/>'
        f'<circle cx="{ox+round(pw*0.09)}" cy="{oy+ph//2}" r="4" fill="#9ca3af"/>'
        # right penalty area
        f'<rect x="{ox+pw-pa_w}" y="{pa_y}" width="{pa_w}" height="{pa_h}" fill="none" stroke="#9ca3af" stroke-width="1.5"/>'
        f'<rect x="{ox+pw-ga_w}" y="{ga_y}" width="{ga_w}" height="{ga_h}" fill="none" stroke="#9ca3af" stroke-width="1"/>'
        f'<circle cx="{ox+pw-round(pw*0.09)}" cy="{oy+ph//2}" r="4" fill="#9ca3af"/>'
        f'</svg>')

    legend_stats=""
    if show_mins:    legend_stats+=" \u00b7 \u2032=mins"
    if show_goals:   legend_stats+=" \u00b7 \u26bd=goals"
    if show_assists: legend_stats+=" \u00b7 \U0001f170=assists"
    legend_bar=(f'<div style="position:absolute;bottom:8px;left:0;right:0;text-align:center;'
                f'font-size:9px;color:#6b7280;">'
                f'Name + contract years{legend_stats} \u00b7 \U0001f501=4+ positions&nbsp;&nbsp;'
                f'<span style="color:#fff;font-weight:700;">Contracted</span>&nbsp;'
                f'<span style="color:#f59e0b;font-weight:700;">Final Year</span>&nbsp;'
                f'<span style="color:#ef4444;font-weight:700;">Out of Contract</span>&nbsp;'
                f'<span style="color:#22c55e;font-weight:700;">On Loan</span></div>')

    return (f'<div id="pitch-root" style="font-family:Montserrat,sans-serif;color:#fff;'
            f'background:#0a0f1c;width:{CANVA_W}px;height:{CANVA_H}px;'
            f'position:relative;overflow:hidden;">'
            f'{landscape_svg}{nodes}{legend_bar}</div>')

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
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    if uploaded:
        @st.cache_data
        def _load(f)->pd.DataFrame:
            df=pd.read_csv(f); df.columns=df.columns.str.strip()
            for c in ["Player","Team","Position","League"]:
                if c in df.columns: df[c]=df[c].astype(str).str.strip()
            for c in ["Minutes played","Goals","Assists","Age","xG","xA"]:
                if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
            df["_ftok"]=df["Position"].apply(_tok); df["_key"]=df["Player"]
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
        max_mins=int(df["Minutes played"].max()) if "Minutes played" in df.columns else 5000
        min_mins=st.slider("Min minutes played",0,max_mins,0,50)
        fdf=fdf[fdf["Minutes played"]>=min_mins]
        sel_team=st.selectbox("Team",sorted(fdf["Team"].unique()))
        formation=st.selectbox("Formation",list(FORMATIONS.keys()))

        st.markdown("---")
        st.markdown("**DISPLAY**")
        st.toggle("Minutes played",  True,  key="show_mins")
        st.toggle("Goals",           True,  key="show_goals")
        st.toggle("Assists",         True,  key="show_assists")
        st.toggle("Show positions",  False, key="show_positions")  # #4
        st.toggle("Role scores",     True,  key="show_roles")
        st.toggle("XI only",         False, key="xi_only")         # #3
        st.toggle("Canva 1920×1080", False, key="canva_mode")      # #5

        st.markdown("---")
        changed=(sel_team!=st.session_state.last_team or
                 formation!=st.session_state.last_formation)
        if st.button("🔄 Build / Rebuild") or changed:
            tdf=fdf[fdf["Team"]==sel_team].copy(); tdf["_key"]=tdf["Player"]
            sm,dep=assign_players(tdf.to_dict("records"),formation)
            st.session_state.slot_map=sm; st.session_state.depth=dep
            st.session_state.last_team=sel_team; st.session_state.last_formation=formation
            st.session_state.move_player=None

        if st.session_state.move_player:
            mp=st.session_state.move_player
            st.markdown(f"**MOVING:** {mp['player']['Player']}")
            opts={f"{s['label']} ({s['id']})":s["id"] for s in FORMATIONS[formation]}
            dest_lbl=st.selectbox("Move to",list(opts.keys()))
            if st.button("✅ Confirm Move"):
                p=mp["player"]; fid=mp["from_slot"]; did=opts[dest_lbl]
                if fid=="_depth":
                    st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=p["_key"]]
                elif fid in st.session_state.slot_map:
                    st.session_state.slot_map[fid]=[x for x in st.session_state.slot_map[fid] if x["_key"]!=p["_key"]]
                st.session_state.slot_map.setdefault(did,[]).append(p)
                st.session_state.move_player=None; st.rerun()
            if st.button("❌ Cancel Move"):
                st.session_state.move_player=None; st.rerun()

        if st.session_state.edit_contract_player:
            ec=st.session_state.edit_contract_player
            st.markdown(f"**EDIT CONTRACT:** {ec['player']['Player']}")
            new_exp=st.text_input("Expires (YYYY-MM-DD)",
                                  value=ec["player"].get("Contract expires",""),key="new_exp")
            if st.button("💾 Save Contract"):
                pk=ec["player"]["_key"]
                for sid,ps in st.session_state.slot_map.items():
                    for p in ps:
                        if p["_key"]==pk: p["Contract expires"]=new_exp
                for p in st.session_state.depth:
                    if p["_key"]==pk: p["Contract expires"]=new_exp
                st.session_state.edit_contract_player=None; st.rerun()
            if st.button("✖ Cancel Edit"):
                st.session_state.edit_contract_player=None; st.rerun()

        st.markdown("---")
        st.markdown("**ADD PLAYER**")
        nn=st.text_input("Name",key="nn")
        np_=st.selectbox("Position",list(CANONICAL.keys()),key="np_")
        # #1: manual extra positions input
        extra_pos=st.text_input("Extra positions (e.g. LCMF,AMF)",key="extra_pos",
                                 help="Add extra positions separated by commas. 4+ positions = 🔁 emoji")
        nm_=st.number_input("Minutes",0,5000,0,10,key="nm_")
        ng_=st.number_input("Goals",0,100,0,key="ng_")
        na_=st.number_input("Assists",0,100,0,key="na_")
        ne_=st.text_input("Contract expires","2026-06-30",key="ne_")
        nl_=st.checkbox("On Loan?",key="nl_")
        sl_opts={f"{s['label']} ({s['id']})":s["id"] for s in FORMATIONS.get(formation,[])}
        ns_=st.selectbox("Add to slot",list(sl_opts.keys()),key="ns_")
        if st.button("➕ Add Player") and nn.strip():
            pos_str=np_
            if extra_pos.strip(): pos_str+=","+extra_pos.strip()
            new_p={"Player":nn.strip(),"Position":pos_str,"_key":f"custom_{nn}",
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
canva=_tog("canva_mode")

pitch=render_pitch(
    team_name,league_nm,formation,slots,slot_map,depth,df_sc,
    _tog("show_mins",True),_tog("show_goals",True),_tog("show_assists",True),
    _tog("show_positions"),_tog("show_roles",True),_tog("xi_only"),canva,
)

if canva:
    # Scale 1920×1080 fixed layout to fit browser window
    st.markdown(
        f'<div style="width:100%;overflow-x:auto;background:#0a0f1c;">'
        f'<div style="transform-origin:top left;width:1920px;">'
        f'{pitch}</div></div>',
        unsafe_allow_html=True)
else:
    st.markdown(pitch, unsafe_allow_html=True)

# ── Downloads ──────────────────────────────────────────────────────────────────
# Build standalone HTML for download (used for both HTML and PNG routes)
page_bg="#0a0f1c"
wrap_style = "display:inline-block;" if canva else "width:800px;margin:0 auto;"
html_body=f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{team_name} Squad Depth</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{page_bg};font-family:Montserrat,sans-serif;}}
</style></head>
<body><div style="{wrap_style}">{pitch}</div>
</body></html>"""

# #6: PNG download — use server-side pillow+cairosvg if available,
# otherwise provide the HTML-that-auto-saves-PNG approach
# Best reliable approach for Streamlit Cloud: download HTML that on open triggers png save
cap_w = "1920" if canva else str(round(800))
cap_h = "1080" if canva else "auto"
png_trigger_html=f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Saving PNG…</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{page_bg};font-family:Montserrat,sans-serif;}}
#msg{{color:#fff;font-size:13px;text-align:center;padding:12px;letter-spacing:.1em;font-family:Montserrat,sans-serif;}}
</style>
</head>
<body>
<div id="msg">GENERATING PNG — PLEASE WAIT…</div>
<div id="capture" style="display:inline-block;">{pitch}</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.addEventListener('load',function(){{
  document.fonts.ready.then(function(){{
    setTimeout(function(){{
      var el=document.getElementById('capture');
      var cfg={{
        backgroundColor:'{page_bg}',
        scale:2,
        useCORS:true,
        allowTaint:false,
        logging:false,
        width:el.offsetWidth,
        height:el.offsetHeight,
        windowWidth:el.offsetWidth,
        windowHeight:el.offsetHeight
      }};
      html2canvas(el,cfg).then(function(canvas){{
        var link=document.createElement('a');
        link.download='{team_name.replace(" ","_")}_squad_depth.png';
        link.href=canvas.toDataURL('image/png');
        link.click();
        document.getElementById('msg').textContent='✓ PNG SAVED — YOU CAN CLOSE THIS TAB';
      }}).catch(function(e){{
        document.getElementById('msg').textContent='ERROR: '+e;
      }});
    }},1200);
  }});
}});
</script>
</body></html>"""

dl1,dl2,_=st.columns([1,1,4])
with dl1:
    st.download_button(
        "⬇ HTML",
        html_body.encode("utf-8"),
        f"{team_name.replace(' ','_')}_squad_depth.html",
        "text/html",
    )
with dl2:
    # #6: Download this HTML file → open in browser → auto-saves PNG
    st.download_button(
        "⬇ PNG",
        png_trigger_html.encode("utf-8"),
        f"{team_name.replace(' ','_')}_OPEN_TO_SAVE_PNG.html",
        "text/html",
        help="Download → open in Chrome/Edge → PNG auto-saves to your Downloads folder",
    )

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
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>MOVE</div>",
                    unsafe_allow_html=True)
        mv_opts={f"{e['player']['Player']} ({e['lbl']})":e for e in all_on}
        mv_sel=st.selectbox("",list(mv_opts.keys()),key="mv_sel",label_visibility="collapsed")
        if st.button("Select for Move"):
            e=mv_opts[mv_sel]
            st.session_state.move_player={"player":e["player"],"from_slot":e["sid"]}; st.rerun()
    with c2:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>REMOVE</div>",
                    unsafe_allow_html=True)
        rm_opts={f"{e['player']['Player']} ({e['lbl']})":e for e in all_on}
        rm_sel=st.selectbox("",list(rm_opts.keys()),key="rm_sel",label_visibility="collapsed")
        if st.button("🗑 Remove"):
            e=rm_opts[rm_sel]; sid=e["sid"]; pk=e["player"]["_key"]
            if sid=="_depth":
                st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=pk]
            else:
                st.session_state.slot_map[sid]=[x for x in st.session_state.slot_map.get(sid,[]) if x["_key"]!=pk]
            st.rerun()
    with c3:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>EDIT CONTRACT</div>",
                    unsafe_allow_html=True)
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
