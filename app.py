"""
Squad Depth Chart — v8
pip install streamlit pandas numpy
streamlit run app.py
"""
import re
import hashlib
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
def _multi_role(pos:str)->bool: return len(_all_toks(pos))>=4

FORMATIONS:dict[str,list[dict]]={
    "4-2-3-1":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":13,"y":30, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"AM",  "label":"AM",  "x":50,"y":32, "accepts":["AM"],             "side":"N","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"RW",  "label":"RW",  "x":87,"y":30, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"DM",  "label":"DM",  "x":35,"y":51, "accepts":["DM"],             "side":"L"},
        {"id":"CM",  "label":"CM",  "x":65,"y":51, "accepts":["CM"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "4-3-3":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":14,"y":21, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"RW",  "label":"RW",  "x":86,"y":21, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"CM",  "label":"CM",  "x":22,"y":41, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":49, "accepts":["DM"],             "side":"N"},
        {"id":"AM",  "label":"AM",  "x":78,"y":41, "accepts":["AM"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "4-4-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":14,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":14,  "accepts":["ST"],             "side":"R"},
        {"id":"LW",  "label":"LW",  "x":5, "y":33, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"CM1", "label":"CM",  "x":34,"y":43, "accepts":["CM"],             "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":43, "accepts":["CM"],             "side":"R"},
        {"id":"RW",  "label":"RW",  "x":95,"y":33, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "3-5-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":14,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":14,  "accepts":["ST"],             "side":"R"},
        {"id":"LWB", "label":"LWB", "x":13,"y":37, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"AM",  "label":"AM",  "x":30,"y":41, "accepts":["AM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":48, "accepts":["DM"],             "side":"N"},
        {"id":"CM",  "label":"CM",  "x":70,"y":41, "accepts":["CM"],             "side":"R"},
        {"id":"RWB", "label":"RWB", "x":87,"y":37, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":67, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":71, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":67, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":88, "accepts":["GK"],             "side":"N"},
    ],
    "3-4-1-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":13,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":13,  "accepts":["ST"],             "side":"R"},
        {"id":"AM",  "label":"AM",  "x":50,"y":25, "accepts":["AM","LW","RW"],   "side":"N","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"LWB", "label":"LWB", "x":13,"y":40, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"CM1", "label":"CM",  "x":34,"y":44, "accepts":["CM"],             "side":"L"},
        {"id":"CM2", "label":"CM",  "x":66,"y":44, "accepts":["CM"],             "side":"R"},
        {"id":"RWB", "label":"RWB", "x":87,"y":40, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":66, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":70, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":66, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":87, "accepts":["GK"],             "side":"N"},
    ],
    "3-4-3":[
        {"id":"LW",  "label":"LW",  "x":14,"y":21, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"RW",  "label":"RW",  "x":86,"y":21, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LWB", "label":"LWB", "x":13,"y":45, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"CM",  "label":"CM",  "x":38,"y":43, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":62,"y":43, "accepts":["DM"],             "side":"R"},
        {"id":"RWB", "label":"RWB", "x":87,"y":45, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":67, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":71, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":67, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":88, "accepts":["GK"],             "side":"N"},
    ],
    "4-1-4-1":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":9, "y":31, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"AM",  "label":"AM",  "x":30,"y":38, "accepts":["AM"],             "side":"L","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"DM",  "label":"DM",  "x":50,"y":41, "accepts":["DM"],             "side":"N"},
        {"id":"CM",  "label":"CM",  "x":70,"y":38, "accepts":["CM"],             "side":"R"},
        {"id":"RW",  "label":"RW",  "x":91,"y":31, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "4-2-3-1 (CM)":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":13,"y":30, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"AM",  "label":"AM",  "x":50,"y":32, "accepts":["AM"],             "side":"N","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"RW",  "label":"RW",  "x":87,"y":30, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LCM", "label":"CM",  "x":35,"y":51, "accepts":["CM"],             "side":"L"},
        {"id":"RCM", "label":"CM",  "x":65,"y":51, "accepts":["CM"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "4-3-3 (CM)":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LW",  "label":"LW",  "x":14,"y":21, "accepts":["LW"],             "side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"RW",  "label":"RW",  "x":86,"y":21, "accepts":["RW"],             "side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"CM1", "label":"CM",  "x":22,"y":41, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":49, "accepts":["DM"],             "side":"N"},
        {"id":"CM2", "label":"CM",  "x":78,"y":41, "accepts":["CM"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "4-3-1-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":14,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":14,  "accepts":["ST"],             "side":"R"},
        {"id":"AM",  "label":"AM",  "x":50,"y":28, "accepts":["AM"],             "side":"N","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"CM1", "label":"CM",  "x":22,"y":42, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":48, "accepts":["DM"],             "side":"N"},
        {"id":"CM2", "label":"CM",  "x":78,"y":42, "accepts":["CM"],             "side":"R"},
        {"id":"LB",  "label":"LB",  "x":12,"y":66, "accepts":["LB","LWB"],       "side":"L","wb_only":True},
        {"id":"CB1", "label":"CB",  "x":32,"y":72, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":68,"y":72, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RB",  "label":"RB",  "x":88,"y":66, "accepts":["RB","RWB"],       "side":"R","wb_only":True},
        {"id":"GK",  "label":"GK",  "x":50,"y":89, "accepts":["GK"],             "side":"N"},
    ],
    "3-4-2-1":[
        {"id":"ST",  "label":"ST",  "x":50,"y":14,  "accepts":["ST"],             "side":"N"},
        {"id":"LAM", "label":"AM",  "x":22,"y":24, "accepts":["LW","AM"],        "side":"L","native_toks":["LW","LWF","LAMF","AMF"]},
        {"id":"RAM", "label":"AM",  "x":78,"y":24, "accepts":["RW","AM"],        "side":"R","native_toks":["RW","RWF","RAMF","AMF"]},
        {"id":"LWB", "label":"LWB", "x":13,"y":45, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"CM",  "label":"CM",  "x":38,"y":43, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":62,"y":43, "accepts":["DM"],             "side":"R"},
        {"id":"RWB", "label":"RWB", "x":87,"y":45, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":25,"y":67, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB",  "label":"CB",  "x":50,"y":71, "accepts":["CB","LCB","RCB"], "side":"N"},
        {"id":"RCB", "label":"RCB", "x":75,"y":67, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":88, "accepts":["GK"],             "side":"N"},
    ],
    "5-3-2":[
        {"id":"ST1", "label":"ST",  "x":35,"y":14,  "accepts":["ST"],             "side":"L"},
        {"id":"ST2", "label":"ST",  "x":65,"y":14,  "accepts":["ST"],             "side":"R"},
        {"id":"LWB", "label":"LWB", "x":13,"y":53, "accepts":["LWB","LB"],       "side":"L","wb_only":True},
        {"id":"CM1", "label":"CM",  "x":30,"y":41, "accepts":["CM"],             "side":"L"},
        {"id":"DM",  "label":"DM",  "x":50,"y":48, "accepts":["DM"],             "side":"N"},
        {"id":"CM2", "label":"CM",  "x":70,"y":41, "accepts":["CM"],             "side":"R"},
        {"id":"RWB", "label":"RWB", "x":87,"y":53, "accepts":["RWB","RB"],       "side":"R","wb_only":True},
        {"id":"LCB", "label":"LCB", "x":17,"y":69, "accepts":["LCB","CB"],       "side":"L"},
        {"id":"CB1", "label":"CB",  "x":37,"y":73, "accepts":["CB","LCB","RCB"], "side":"L"},
        {"id":"CB2", "label":"CB",  "x":63,"y":73, "accepts":["CB","LCB","RCB"], "side":"R"},
        {"id":"RCB", "label":"RCB", "x":83,"y":69, "accepts":["RCB","CB"],       "side":"R"},
        {"id":"GK",  "label":"GK",  "x":50,"y":88, "accepts":["GK"],             "side":"N"},
    ],
}

PITCH_ORDER=["GK","LCB","CB","RCB","LB","RB","LWB","RWB","CM","DM","AM","LW","RW","ST"]

# ── Wide-slot edge guard ──────────────────────────────────────────────────────
# Portrait nodes are centred on their slot's x%, and an edge node's text block is
# capped at EDGE_NODE_MAX_W (see the max-width in render_pitch's make_node and the
# name_maxw passed in render_pitch_png). A slot centred less than half that width
# inside the touchline therefore has its name, minutes and role scores crossing the
# pitch border — 4-4-2's wingers sat at x=5/95, well outside it.
#
# Clamping the slot list is the one place both renderers inherit: the HTML render
# and the PNG render are handed the same list, so neither needs its own guard.
EDGE_NODE_MAX_W = 115.0   # px — portrait edge-node max-width
EDGE_NODE_MARGIN = 20.0   # px — breathing room between the text block and the line.
                          # Sized so the guard lands at x≈11.2%, which matches how far
                          # inside the already-fine slots sit (LB/RB at 12/88 keep ~25px)
                          # and leaves them untouched. Raising it past ~25 would start
                          # moving those too — keep min_slot_x() ≤ 12.
PITCH_SIDE_PAD = 4.0      # px — #pitch-root's own left/right padding

def min_slot_x(pitch_width_px: float = 700.0) -> float:
    """Smallest x% whose node still clears the touchline at this pitch width."""
    field_w = max(float(pitch_width_px) - 2 * PITCH_SIDE_PAD, 1.0)
    return 100.0 * (EDGE_NODE_MAX_W / 2.0 + EDGE_NODE_MARGIN) / field_w

def layout_slots(formation: str, pitch_width_px: float = 700.0) -> list:
    """FORMATIONS[formation] with any over-wide slot pulled back inside the pitch.

    Returns the original dicts untouched where nothing needs moving, so formations
    that already sit inside the safe band (4-2-3-1, 4-3-3, the 3-5-2 wingbacks …)
    render exactly as before.
    """
    lo = min_slot_x(pitch_width_px)
    hi = 100.0 - lo
    out = []
    for sl in FORMATIONS.get(formation, []):
        x = float(sl.get("x", 50))
        nx = min(max(x, lo), hi)
        out.append(sl if nx == x else dict(sl, x=round(nx, 2)))
    return out

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

def is_loaned_out(p:dict)->bool:
    return str(p.get("Loaned Out","")).strip().lower() in ("yes","y","true","1")

def is_youth(p:dict)->bool:
    return str(p.get("Youth Player","")).strip().lower() in ("yes","y","true","1")

def is_esc(p:dict)->bool:
    pk = p.get("_key","")
    return pk in st.session_state.get("esc_players", set())

def player_css_color(yrs:int,loan:bool,loaned_out:bool=False,youth:bool=False,esc:bool=False,esc_blue:bool=False)->str:
    if esc and esc_blue: return "#60a5fa"  # light blue for ESC when toggle on
    if loaned_out: return "#c084fc"   # light purple — loaned out
    if youth:      return "#9ca3af"   # light grey — youth player
    if loan:       return "#22c55e"   # green — on loan (incoming)
    if yrs==0:     return "#ef4444"   # red — out of contract
    if yrs==1:     return "#f59e0b"   # amber — final year
    return "#ffffff"

def score_to_color(v:float)->str:
    if np.isnan(v): return "#4b5563"
    v=max(0.0,min(100.0,float(v)))
    if v<=50:
        t=v/50; r=int(239+(234-239)*t); g=int(68+(179-68)*t); b=int(68+(8-68)*t)
    else:
        t=(v-50)/50; r=int(234+(34-234)*t); g=int(179+(197-179)*t); b=int(8+(94-8)*t)
    return f"rgb({r},{g},{b})"

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


# Fallback canonical: maps raw token → ordered list of slot labels to try when player unassigned
FALLBACK_CANON:dict[str,list]={
    "DMF":["DM","CM"],"LDMF":["DM","CM"],"RDMF":["DM","CM"],
    "LCMF":["CM","DM"],"RCMF":["CM","DM"],
    "AMF":["AM","CM","LW","RW"],"LAMF":["LW","AM","RW"],"RAMF":["RW","AM","LW"],
    "LW":["LW","AM"],"RW":["RW","AM"],"LWF":["LW","AM"],"RWF":["RW","AM"],
    "CF":["ST"],"GK":["GK"],
    "CB":["CB","LCB","RCB"],"LCB":["LCB","CB"],"RCB":["RCB","CB"],
    "LB":["LB","LWB"],"RB":["RB","RWB"],"LWB":["LWB","LB"],"RWB":["RWB","RB"],
}
def assign_players(players:list,formation_key:str)->tuple[dict,list]:
    slots=FORMATIONS.get(formation_key,FORMATIONS["4-2-3-1"])
    by_label:dict[str,list]={}
    for s in slots: by_label.setdefault(s["label"],[]).append(s)
    assigned:set=set()
    slot_map:dict[str,list]={s["id"]:[] for s in slots}

    # All canonical slot labels present in this formation
    formation_labels:set=set(by_label.keys())

    def first_tok_fits(p,slot):
        """True only if the player's FIRST position token canonically matches this slot."""
        tok=_tok(p.get("Position",""))
        if slot.get("wb_only"):
            return tok in {"LB","LWB","RB","RWB"} and CANONICAL.get(tok,"CM") in slot["accepts"]
        return CANONICAL.get(tok,"CM") in slot["accepts"]

    def primary_fits(p,slot):
        """Used for OOP flagging only — same as first_tok_fits."""
        return first_tok_fits(p,slot)

    def has_any_primary_slot(p):
        """True if player's first token has a matching slot label in this formation."""
        tok=_tok(p.get("Position",""))
        canon=CANONICAL.get(tok,"CM")
        return canon in formation_labels

    def secondary_fits(p,slot):
        """Only secondary tokens — and only used for players with no primary slot."""
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

        # Pass 1: players whose FIRST token fits this slot
        matched=[p for p in players if p["_key"] not in assigned
                 and any(first_tok_fits(p,s) for s in slot_list)]

        # Pass 2: only if no primary matches — take players who have no primary slot
        # anywhere in the formation AND whose secondary tokens fit here
        if not matched:
            matched=[p for p in players if p["_key"] not in assigned
                     and not has_any_primary_slot(p)
                     and any(secondary_fits(p,s) for s in slot_list)]

        matched.sort(key=lambda p:-float(p.get("Minutes played") or 0))

        # priority_toks: within first-token matches only, boost specific tokens to front
        # (e.g. AMF before LAMF/RAMF for AM slot) — never pulls in outsiders
        pt=set()
        for sl in slot_list: pt.update(sl.get("priority_toks",[]))
        if pt:
            matched.sort(key=lambda p:(0 if _tok(p.get("Position","")) in pt else 1,
                                       -float(p.get("Minutes played") or 0)))

        for p in matched: assigned.add(p["_key"])
        n=len(slot_list)
        if n==1:
            slot_map[slot_list[0]["id"]]=matched
        else:
            ordered=sorted(slot_list,key=lambda s:{"L":0,"N":1,"R":2}[s["side"]])
            for sl in slot_list: slot_map[sl["id"]]=[]
            starters=[]; used=set()
            # Pick best-fit starter for each slot
            for sl in ordered:
                best=None; best_sc=99
                for p in matched:
                    if id(p) in used: continue
                    sc=side_score(p,sl["side"])
                    if sc<best_sc: best_sc=sc; best=p
                if best: starters.append((sl["id"],best)); used.add(id(best))
            for sid,p in starters: slot_map[sid].append(p)
            # Distribute remaining players round-robin across slots by minutes order
            depth_rem=[p for p in matched if id(p) not in used]
            for i,p in enumerate(depth_rem):
                slot_map[ordered[i % n]["id"]].append(p)


    # ── Fix 4: 4-back CB redistribution by position token ────────────────────
    FOUR_BACK_FORMATIONS={"4-2-3-1","4-2-3-1 (CM)","4-3-3","4-3-3 (CM)","4-3-1-2","4-4-2","4-1-4-1","3-4-3"}
    if formation_key in FOUR_BACK_FORMATIONS:
        cb1_id=next((s["id"] for s in slots if s["id"]=="CB1"),None)
        cb2_id=next((s["id"] for s in slots if s["id"]=="CB2"),None)
        if cb1_id and cb2_id:
            all_cbs4=[]
            for sid in (cb1_id,cb2_id):
                all_cbs4.extend(slot_map.get(sid,[]))
            all_cbs4.sort(key=lambda p:-float(p.get("Minutes played") or 0))
            lcb_p=[p for p in all_cbs4 if _tok(p.get("Position",""))=="LCB"]
            rcb_p=[p for p in all_cbs4 if _tok(p.get("Position",""))=="RCB"]
            cb_p =[p for p in all_cbs4 if _tok(p.get("Position",""))=="CB"]
            oth_p=[p for p in all_cbs4 if _tok(p.get("Position","")) not in {"CB","LCB","RCB"}]
            # Left slot = CB1, Right slot = CB2
            left=[]; right=[]
            # Assign specific sided players first
            left.extend(lcb_p); right.extend(rcb_p)
            # Distribute pure CB alternately starting with left (most mins first)
            for i,p in enumerate(cb_p):
                (left if i%2==0 else right).append(p)
            # Any others (OOP) fill by minutes alternately
            for i,p in enumerate(oth_p):
                (left if i%2==0 else right).append(p)
            slot_map[cb1_id]=left
            slot_map[cb2_id]=right
    # ── End Fix 4 ────────────────────────────────────────────────────────────
    # ── Fix 6: 3-back CB redistribution ──────────────────────────────────────
    # For 3-back formations, re-distribute CB/LCB/RCB players correctly:
    # Pure CB → middle; LCB → left; RCB → right.
    # If no pure CB, alternate by minutes: 1st→CB, 2nd→RCB, 3rd→CB, 4th→RCB...
    THREE_BACK_FORMATIONS={"3-5-2","3-4-1-2","3-4-3","3-4-2-1"}
    if formation_key in THREE_BACK_FORMATIONS:
        lcb_id=next((s["id"] for s in slots if s["id"]=="LCB"),None)
        cb_id =next((s["id"] for s in slots if s["id"]=="CB"), None)
        rcb_id=next((s["id"] for s in slots if s["id"]=="RCB"),None)
        if lcb_id and cb_id and rcb_id:
            # Collect all players currently in these three slots
            all_cbs=[]
            for sid in (lcb_id,cb_id,rcb_id):
                all_cbs.extend(slot_map.get(sid,[]))
            # Sort by minutes descending
            all_cbs.sort(key=lambda p:-float(p.get("Minutes played") or 0))
            # Separate by primary position token
            pure_cb =[p for p in all_cbs if _tok(p.get("Position",""))=="CB"]
            pure_lcb=[p for p in all_cbs if _tok(p.get("Position",""))=="LCB"]
            pure_rcb=[p for p in all_cbs if _tok(p.get("Position",""))=="RCB"]
            other   =[p for p in all_cbs if _tok(p.get("Position","")) not in {"CB","LCB","RCB"}]
            # Fill slots:
            # LCB slot: LCB players first, then overflow from other
            # CB slot:  pure CB players first
            # RCB slot: RCB players first
            # If pure_cb empty, distribute non-LCB/RCB players alternately CB→RCB
            cb_starters=[]; rcb_starters=[]; lcb_starters=list(pure_lcb)
            if pure_cb:
                cb_starters=pure_cb
                rcb_starters=pure_rcb
                # Any remaining RCB go to LCB depth if not enough LCB players
                if not lcb_starters: lcb_starters=other
            else:
                # No pure CB — interleave remaining (sorted by mins) between CB and RCB
                remaining=sorted([p for p in all_cbs if p not in pure_lcb],
                                 key=lambda p:-float(p.get("Minutes played") or 0))
                for i,p in enumerate(remaining):
                    if i%2==0: cb_starters.append(p)
                    else:      rcb_starters.append(p)
            # Assign
            slot_map[lcb_id]=lcb_starters if lcb_starters else other
            slot_map[cb_id] =cb_starters
            slot_map[rcb_id]=rcb_starters
    # ── 5-3-2 CB redistribution (4 CB slots) ────────────────────────────────
    if formation_key == "5-3-2":
        lcb_id=next((s["id"] for s in slots if s["id"]=="LCB"),None)
        cb1_id=next((s["id"] for s in slots if s["id"]=="CB1"),None)
        cb2_id=next((s["id"] for s in slots if s["id"]=="CB2"),None)
        rcb_id=next((s["id"] for s in slots if s["id"]=="RCB"),None)
        ids_532=[i for i in [lcb_id,cb1_id,cb2_id,rcb_id] if i]
        if len(ids_532)==4:
            all_cbs5=[]
            for sid in ids_532: all_cbs5.extend(slot_map.get(sid,[]))
            all_cbs5.sort(key=lambda p:-float(p.get("Minutes played") or 0))
            pure_lcb=[p for p in all_cbs5 if _tok(p.get("Position",""))=="LCB"]
            pure_rcb=[p for p in all_cbs5 if _tok(p.get("Position",""))=="RCB"]
            pure_cb =[p for p in all_cbs5 if _tok(p.get("Position",""))=="CB"]
            oth5    =[p for p in all_cbs5 if _tok(p.get("Position","")) not in {"CB","LCB","RCB"}]
            slot_map[lcb_id]=pure_lcb or []
            slot_map[rcb_id]=pure_rcb or []
            # Distribute pure CB evenly between inner slots L/R, alternating by mins
            inner=[]; [inner.append(p) for p in pure_cb+oth5]
            slot_map[cb1_id]=[p for i,p in enumerate(inner) if i%2==0]
            slot_map[cb2_id]=[p for i,p in enumerate(inner) if i%2==1]
    # ── End 5-3-2 CB ─────────────────────────────────────────────────────────
    # ── End Fix 6 ────────────────────────────────────────────────────────────

    for sid,ps in slot_map.items():
        slot_def=next((s for s in slots if s["id"]==sid),None)
        for p in ps:
            p["_oop"]=not primary_fits(p,slot_def) if slot_def else False
            p["_primary_pos"]=_tok(p.get("Position",""))

    # ── Fallback pass: cascade remaining players into best-fit slot ─────────
    # Players who couldn't fit their primary slot get assigned to nearest slot
    # that exists in the formation, marked as OOP. No one goes to depth unless
    # there is genuinely no slot that can accommodate them.
    by_label_id:dict[str,list]={s["label"]:[] for s in slots}
    for s in slots: by_label_id[s["label"]].append(s["id"])

    remaining_after_main=[p for p in players if p["_key"] not in assigned]
    remaining_after_main.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    for p in remaining_after_main:
        tok=_tok(p.get("Position",""))
        placed=False
        for try_label in FALLBACK_CANON.get(tok,[tok]):
            if try_label in by_label_id:
                # pick the slot with label try_label that has fewest players so far
                best_sid=min(by_label_id[try_label],
                             key=lambda sid:len(slot_map.get(sid,[])))
                slot_map.setdefault(best_sid,[]).append(p)
                assigned.add(p["_key"])
                placed=True
                break
        if not placed:
            # Try any slot as absolute last resort (pick least populated)
            best_sid=min((s["id"] for s in slots),
                         key=lambda sid:len(slot_map.get(sid,[])))
            slot_map.setdefault(best_sid,[]).append(p)
            assigned.add(p["_key"])
    # ── End fallback pass ────────────────────────────────────────────────────

    # Re-flag _oop and _primary_pos for ALL players now (including fallback-placed)
    for sid,ps in slot_map.items():
        slot_def=next((s for s in slots if s["id"]==sid),None)
        for p in ps:
            p["_oop"]=not primary_fits(p,slot_def) if slot_def else False
            p["_primary_pos"]=_tok(p.get("Position",""))
            # _show_pos: also show position when tok is not native to this slot
            native=slot_def.get("native_toks") if slot_def else None
            p["_show_pos"]=(p["_oop"] or (native is not None and p["_primary_pos"] not in native))

    depth=[p for p in players if p["_key"] not in assigned]
    depth.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    return slot_map,depth

# ── Score HTML ─────────────────────────────────────────────────────────────────
def all_roles_html(player,df_sc,fs="8px",flip=False):
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
        name_col = sc_col if is_b else "#7a8494"
        if flip:
            # Right-anchored node: role name on LEFT, score on RIGHT so it reads toward pitch
            lines.append(
                f'<div style="display:flex;justify-content:flex-end;gap:6px;font-size:{fs};line-height:1.4;white-space:nowrap;">'
                f'<span style="color:{name_col};font-weight:{"700" if is_b else "400"};min-width:110px;text-align:right;">{rn}</span>'
                f'<span style="color:{sc_col};font-weight:{"700" if is_b else "400"};width:22px;flex-shrink:0;">{int(sc)}</span></div>')
        else:
            lines.append(
                f'<div style="display:flex;justify-content:space-between;gap:4px;font-size:{fs};line-height:1.4;min-width:90px;">'
                f'<span style="color:{name_col};font-weight:{"700" if is_b else "400"};">{rn}</span>'
                f'<span style="color:{sc_col};font-weight:{"700" if is_b else "400"};min-width:22px;text-align:right;">{int(sc)}</span></div>')
    return f'<div style="margin-top:2px;">{"".join(lines)}</div>'

def best_role_html(player,df_sc,fs="8px",flip=False):
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
    if flip:
        # Right-anchored: role name on LEFT, score on RIGHT
        return (f'<div style="display:flex;justify-content:flex-end;gap:6px;font-size:{fs};line-height:1.4;margin-top:2px;min-width:90px;">'
                f'<span style="color:#7a8494;text-align:right;">{best}</span>'
                f'<span style="color:{sc_col};font-weight:700;min-width:22px;">{int(sc)}</span></div>')
    else:
        return (f'<div style="display:flex;justify-content:space-between;gap:4px;font-size:{fs};line-height:1.4;margin-top:2px;min-width:90px;">'
                f'<span style="color:#7a8494;">{best}</span>'
                f'<span style="color:{sc_col};font-weight:700;min-width:22px;text-align:right;">{int(sc)}</span></div>')

# ── SVG pitch lines — dimmed so text always wins ──────────────────────────────
# Opacity 0.18 so pitch outline is visible as a guide but never overpowers text
PORTRAIT_SVG="""
  <rect  x="2"   y="2"     width="96" height="138" fill="none" stroke="#9ca3af" stroke-width="1.2" opacity=".18"/>
  <line  x1="2"  y1="71"   x2="98"   y2="71"      stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <circle cx="50" cy="71" r="10"                   fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <circle cx="50" cy="71" r="1.2"                  fill="#9ca3af" opacity=".18"/>
  <rect  x="22"  y="2"     width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <rect  x="36"  y="2"     width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"  opacity=".18"/>
  <circle cx="50" cy="14" r=".9"                   fill="#9ca3af" opacity=".18"/>
  <rect  x="22"  y="122"   width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <rect  x="36"  y="133"   width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"  opacity=".18"/>
  <circle cx="50" cy="126" r=".9"                  fill="#9ca3af" opacity=".18"/>"""

# ── Canva landscape layout constants ─────────────────────────────────────────
# 1920×1080 slide: portrait pitch centred, players read GK→ST left to right
# Pitch sits in horizontal centre, rotated 90° to landscape
# We use a PORTRAIT pitch in the centre of the slide (narrower, taller),
# with GK at bottom and ST at top, matching Image 3 template style
# Players are arranged with depth info flanking the pitch

# For the Canva slide we render a landscape SVG pitch occupying most of the slide:
# Pitch block: 1520px wide × 870px tall, centred in 1920×1080
CANVA_W, CANVA_H = 1920, 1080
# Landscape pitch: GK left → ST right, fills almost all slide
# Tight margins: 40px sides, 80px top/bottom (for legend bar)
CPX, CPY = 40, 78       # top-left of pitch
CPW, CPH = 1840, 924    # pitch width × height
# penalty area proportions
CP_PAW = round(CPW * 0.11)
CP_PAH = round(CPH * 0.40)
CP_GAW = round(CPW * 0.035)
CP_GAH = round(CPH * 0.22)
CP_CR  = round(min(CPW,CPH) * 0.08)

def canva_landscape_svg()->str:
    """Landscape pitch SVG for 1920×1080 canvas"""
    ox,oy,pw,ph=CPX,CPY,CPW,CPH
    pa_y=oy+round((ph-CP_PAH)/2); ga_y=oy+round((ph-CP_GAH)/2)
    cx=ox+pw//2; cy=oy+ph//2
    return (
        f'<svg style="position:absolute;left:0;top:0;width:{CANVA_W}px;height:{CANVA_H}px;'
        f'pointer-events:none;z-index:1;" viewBox="0 0 {CANVA_W} {CANVA_H}">'
        # pitch fill - subtle green tint
        f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="#0d1820" opacity=".6"/>'
        # outer border
        f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#374151" stroke-width="2"/>'
        # halfway line
        f'<line x1="{cx}" y1="{oy}" x2="{cx}" y2="{oy+ph}" stroke="#374151" stroke-width="1.5"/>'
        # centre circle
        f'<circle cx="{cx}" cy="{cy}" r="{CP_CR}" fill="none" stroke="#374151" stroke-width="1.5"/>'
        f'<circle cx="{cx}" cy="{cy}" r="5" fill="#374151"/>'
        # left pen area (GK side)
        f'<rect x="{ox}" y="{pa_y}" width="{CP_PAW}" height="{CP_PAH}" fill="none" stroke="#374151" stroke-width="1.5"/>'
        f'<rect x="{ox}" y="{ga_y}" width="{CP_GAW}" height="{CP_GAH}" fill="none" stroke="#374151" stroke-width="1"/>'
        f'<circle cx="{ox+round(CPW*0.08)}" cy="{cy}" r="4" fill="#374151"/>'
        # right pen area (ST side)
        f'<rect x="{ox+pw-CP_PAW}" y="{pa_y}" width="{CP_PAW}" height="{CP_PAH}" fill="none" stroke="#374151" stroke-width="1.5"/>'
        f'<rect x="{ox+pw-CP_GAW}" y="{ga_y}" width="{CP_GAW}" height="{CP_GAH}" fill="none" stroke="#374151" stroke-width="1"/>'
        f'<circle cx="{ox+pw-round(CPW*0.08)}" cy="{cy}" r="4" fill="#374151"/>'
        f'</svg>'
    )

def canva_slot_px(slot_x:float, slot_y:float, slot_id:str="")->tuple[int,int,str,str]:
    """Portrait % → landscape px + smart CSS anchor for nodes.
    Portrait y%: small=attack(ST), large=defence(GK)
    Landscape: GK → left side (small lx), ST → right side (large lx)
    Portrait x%: small=left wing (LW), large=right wing (RW)
    Landscape: LW → top (small ly), RW → bottom (large ly)
    Returns: (lx, ly, css_transform, text_align)
    
    Special handling: LWB/RWB in 3-back formations pushed closer to edges
    """
    Y_MIN,Y_MAX=7.0,87.0
    # Very small inner padding so nodes spread to pitch edges
    INNER_PAD_X=20   # inset from pitch border for player text
    INNER_PAD_Y=12
    
    # Adjust x position for LWB/RWB in Canva mode only
    adjusted_x = slot_x
    if slot_id in ("LWB", "RWB"):
        # Push wingbacks to the absolute edges (only in Canva)
        if slot_x < 50:  # LWB (left side)
            adjusted_x = 1  # almost touching left edge
        else:  # RWB (right side)
            adjusted_x = 99  # almost touching right edge
    
    lx_pct = 1.0 - (slot_y - Y_MIN) / (Y_MAX - Y_MIN)  # 0=GK-side,1=ST-side
    lx = CPX + INNER_PAD_X + lx_pct * (CPW - 2*INNER_PAD_X)
    ly_pct = adjusted_x / 100.0
    ly = CPY + INNER_PAD_Y + ly_pct * (CPH - 2*INNER_PAD_Y)
    # Smart anchor: keep nodes inside pitch boundaries
    # Horizontal: GK side → text grows right; ST side → text grows left; else centre
    if lx_pct < 0.12:   tx="translate(0,-50%)";   ta="left"   # GK: anchor left edge
    elif lx_pct > 0.88: tx="translate(-100%,-50%)"; ta="right"  # ST: anchor right edge
    else:               tx="translate(-50%,-50%)"; ta="center"
    # Vertical: top edge → text grows down; bottom → text grows up
    if ly_pct < 0.12:   tx=tx.replace("-50%)",  "0)")            # top: grow down
    elif ly_pct > 0.88: tx=tx.replace("-50%)",  "-100%)")        # bottom: grow up
    return round(lx), round(ly), tx, ta

# ── Render pitch ───────────────────────────────────────────────────────────────
def render_pitch(
    team:str, league:str, formation:str,
    slots:list, slot_map:dict, depth:list, df_sc,
    show_mins:bool, show_goals:bool, show_assists:bool,
    show_positions:bool, show_roles:bool, xi_only:bool, canva:bool,
    pitch_width_px:int=560,
    white_names:bool=False,
    show_contracts:bool=True,
    best_role_only:bool=False,
    esc_blue:bool=False,
)->str:
    BG="#0a0f1c"

    # ── shared node builder ────────────────────────────────────────────────────
    def make_node(slot, pos_style:str, bsz:str, nsz:str, ssz:str, rsz:str)->str:
        ps_all=slot_map.get(slot["id"],[])
        ps=ps_all[:1] if xi_only else ps_all
        badge=(f'<div style="display:inline-block;padding:2px 8px;border:2px solid #ef4444;'
               f'color:#ef4444;font-size:{bsz};font-weight:900;letter-spacing:.1em;'
               f'margin-bottom:3px;background:rgba(10,15,28,.97);">{slot["label"]}</div>')
        rows=""
        real_shown=0
        for i,p in enumerate(ps):
            if p.get("_is_ns"):
                _sn_lbl=p.get("_ns_label","NEW SIGNING") or "NEW SIGNING"
                _sn_sub=p.get("_ns_sub","")
                _sn_col=p.get("_ns_color","#ef4444")
                mt_ns="margin-top:4px;" if rows else ""
                rows+=(f'<div style="color:{_sn_col};font-size:{nsz};font-weight:800;{mt_ns}'
                        f'letter-spacing:.08em;line-height:1.4;text-transform:uppercase;'
                        f'text-shadow:0 0 8px rgba(0,0,0,1);">{_sn_lbl}</div>')
                if _sn_sub:
                    rows+=(f'<div style="color:{_sn_col};font-size:{rsz};font-weight:400;'
                            f'line-height:1.3;">{_sn_sub}</div>')
                continue
            ri=real_shown; real_shown+=1
            yrs=contract_years(p.get("Contract expires",""))
            yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); fw="800" if ri==0 else "500"
            _lo=is_loaned_out(p); _yt=is_youth(p); _esc=is_esc(p)
            col=("#ffffff" if white_names else player_css_color(yrs,loan,_lo,_yt,_esc,esc_blue))
            multi=" \U0001f501" if _multi_role(p.get("Position","")) else ""
            _hpo=st.session_state.get('hide_pos_override',set())
            _hop=st.session_state.get('hide_oop_players',set())
            oop_s=f" ({p['_primary_pos']})" if (p.get('_show_pos') and p.get('_key','') not in _hpo and p.get('_key','') not in _hop) else ''
            lo=is_loaned_out(p); yt=is_youth(p)
            if loan:
                suffix=f" L{oop_s}{multi}" if show_contracts else f"{oop_s}{multi}"
            else:
                suffix=f"{(yr_str if show_contracts else '')}{oop_s}{multi}"
            stat_parts=[]
            if show_mins:   stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}\u2032")
            if show_goals:
                g=float(p.get("Goals") or 0)
                if g>0: stat_parts.append(f"{int(g)}\u26bd")
            if show_assists:
                a=float(p.get("Assists") or 0)
                if a>0: stat_parts.append(f"{int(a)}\U0001f170")
            stat_html=(f'<div style="color:#fff;font-size:{ssz};line-height:1.2;opacity:.9;">'
                       f'{" ".join(stat_parts)}</div>') if stat_parts else ""
            all_pos=", ".join(_all_toks(p.get("Position","")))
            pos_html=(f'<div style="color:#9ca3af;font-size:{ssz};line-height:1.2;">{all_pos}</div>'
                      ) if (show_positions and all_pos) else ""
            rs_html=(best_role_html(p,df_sc,rsz) if (show_roles and best_role_only)
                     else all_roles_html(p,df_sc,rsz) if (ri==0 and show_roles)
                     else best_role_html(p,df_sc,rsz) if (ri>0 and show_roles) else "")
            mt="margin-top:5px;" if rows else ""
            rows+=(f'<div style="color:{col};font-size:{nsz};line-height:1.45;font-weight:{fw};{mt}'
                   f'white-space:nowrap;text-shadow:0 0 8px rgba(0,0,0,1),0 0 4px rgba(0,0,0,1);">'
                   f'{p["Player"]} {suffix}</div>{pos_html}{stat_html}{rs_html}')
        if not rows:
            rows=f'<div style="color:#1f2937;font-size:{ssz};">&#8212;</div>'
        sx=float(slot.get("x",50))
        is_edge=(sx<20 or sx>80)
        if canva:
            # Canva: generous width, text-align toward pitch centre
            mw="160px"; mxw="220px"
            talign="left" if sx<20 else ("right" if sx>80 else "center")
        else:
            # Portrait: edge nodes get a max-width cap so very long names wrap
            # naturally; short names (J. Key) are never affected since they fit fine.
            mw="80px"
            mxw="115px" if is_edge else "none"
            talign="center"
        return (f'<div style="position:absolute;{pos_style}'
                f'transform:translate(-50%,-50%);text-align:{talign};'
                f'min-width:{mw};max-width:{mxw};z-index:10;">'
                f'{badge}<div>{rows}</div></div>')

    # ── legend text ───────────────────────────────────────────────────────────
    def legend_text()->str:
        s=""
        if show_mins:    s+=" \u00b7 \u2032=mins"
        if show_goals:   s+=" \u00b7 \u26bd=goals"
        if show_assists: s+=" \u00b7 \U0001f170=assists"
        return s

    # ── CANVA mode (1920×1080 landscape) ──────────────────────────────────────
    # Landscape pitch: GK left → ST right, full-width, smart node anchoring.
    if canva:
        bsz="32px"; nsz="29px"; ssz="21px"; rsz="20px"

        def make_canva_node_ls(slot)->str:
            lx,ly,tx,ta=canva_slot_px(float(slot["x"]),float(slot["y"]),slot["id"])
            ps_all=slot_map.get(slot["id"],[])
            ps=ps_all[:1] if xi_only else ps_all
            badge=(f'<div style="display:inline-block;padding:3px 12px;'
                   f'border-radius:8px;background:#b8bfc9;'
                   f'color:#1f2937;font-size:{bsz};font-weight:900;letter-spacing:.07em;'
                   f'margin-bottom:5px;white-space:nowrap;">{slot["label"]}</div>')
            rows=""
            _slot_ns_list=st.session_state.get("new_signing",{}).get(slot["id"],[])
            if isinstance(_slot_ns_list,dict): _slot_ns_list=[_slot_ns_list]
            real_shown=0
            for i,p in enumerate(ps):
                if p.get("_is_ns"):
                    _sn_lbl=p.get("_ns_label","NEW SIGNING") or "NEW SIGNING"
                    _sn_sub=p.get("_ns_sub","")
                    _sn_col=p.get("_ns_color","#ef4444")
                    mt_ns="margin-top:4px;" if rows else ""
                    rows+=(f'<div style="color:{_sn_col};font-size:{nsz};font-weight:800;{mt_ns}'
                            f'letter-spacing:.08em;line-height:1.4;text-transform:uppercase;">{_sn_lbl}</div>')
                    if _sn_sub:
                        rows+=(f'<div style="color:{_sn_col};font-size:{rsz};font-weight:400;'
                                f'line-height:1.3;">{_sn_sub}</div>')
                    continue
                ri=real_shown; real_shown+=1
                yrs=contract_years(p.get("Contract expires",""))
                yr_str=f"+{yrs}" if yrs>=0 else "+?"
                loan=is_loan(p); fw="700" if ri==0 else "400"
                _lo=is_loaned_out(p); _yt=is_youth(p); _esc=is_esc(p)
                col=("#ffffff" if white_names else player_css_color(yrs,loan,_lo,_yt,_esc,esc_blue))
                multi=" 🔁" if _multi_role(p.get("Position","")) else ""
                _hpo=st.session_state.get("hide_pos_override",set())
                _hop=st.session_state.get("hide_oop_players",set())
                oop_s=f" ({p['_primary_pos']})" if (p.get('_show_pos') and p.get('_key','') not in _hpo and p.get('_key','') not in _hop) else ''
                lo=is_loaned_out(p); yt=is_youth(p)
                if loan:
                    suffix=f" L{oop_s}{multi}" if show_contracts else f"{oop_s}{multi}"
                else:
                    suffix=f"{(yr_str if show_contracts else '')}{oop_s}{multi}"
                mt="margin-top:5px;" if rows else ""
                rs_html=(best_role_html(p,df_sc,rsz,flip=(ta=="right")) if (show_roles and best_role_only)
                         else all_roles_html(p,df_sc,rsz,flip=(ta=="right")) if (ri==0 and show_roles)
                         else best_role_html(p,df_sc,rsz,flip=(ta=="right")) if (ri>0 and show_roles) else "")
                rows+=(f'<div style="color:{col};font-size:{nsz};line-height:1.4;font-weight:{fw};{mt}'
                       f'white-space:nowrap;text-shadow:0 0 6px rgba(0,0,0,1);">'
                       f'{p["Player"]}{suffix}</div>{rs_html}')
            for _sn in _slot_ns_list:
                _sn_lbl=_sn.get("label","NEW SIGNING") or "NEW SIGNING"
                _sn_sub=_sn.get("sub","")
                _sn_col=_sn.get("color","#ef4444")
                mt_ns="margin-top:4px;" if rows else ""
                rows+=(f'<div style="color:{_sn_col};font-size:{nsz};font-weight:800;{mt_ns}'
                        f'letter-spacing:.08em;line-height:1.4;text-transform:uppercase;">{_sn_lbl}</div>')
                if _sn_sub:
                    rows+=(f'<div style="color:{_sn_col};font-size:{rsz};font-weight:400;'
                            f'line-height:1.3;">{_sn_sub}</div>')
            if not rows:
                rows=f'<div style="color:#4b5563;font-size:{ssz};">&#8212;</div>'
            return (f'<div style="position:absolute;left:{lx}px;top:{ly}px;'
                    f'transform:{tx};text-align:{ta};z-index:10;">'
                    f'{badge}<div>{rows}</div></div>')

        nodes="".join(make_canva_node_ls(s) for s in slots)

        # Legend bar — sits above the pitch (top strip)
        esc_legend=f'<span style="color:#60a5fa;font-weight:700;">ESC</span>&ensp;' if esc_blue else ""
        header=(f'<div style="position:absolute;top:16px;left:{CPX}px;right:{CANVA_W-CPX-CPW}px;'
                f'display:flex;justify-content:space-between;align-items:center;z-index:20;'
                f'font-size:21px;color:#6b7280;letter-spacing:.03em;width:{CPW}px;">'
                f'<span>Name + contract years{legend_text()} &nbsp;·&nbsp; 🔁=4+ positions</span>'
                f'<span>'
                f'<span style="color:#ffffff;font-weight:700;">Under Contract</span>&ensp;'
                f'<span style="color:#ef4444;font-weight:700;">Out of Contract</span>&ensp;'
                f'<span style="color:#f59e0b;font-weight:700;">Final Year</span>&ensp;'
                f'<span style="color:#22c55e;font-weight:700;">On Loan</span>&ensp;'
                f'<span style="color:#c084fc;font-weight:700;">Loaned Out</span>&ensp;'
                f'<span style="color:#9ca3af;font-weight:700;">Youth</span>&ensp;'
                f'{esc_legend}'
                f'<span style="color:#6b7280;">{league} · {formation}</span>'
                f'</span></div>')

        return (f'<div id="pitch-root" style="font-family:Montserrat,sans-serif;color:#fff;'
                f'background:{BG};width:{CANVA_W}px;height:{CANVA_H}px;position:relative;'
                f'overflow:hidden;">'
                f'{canva_landscape_svg()}{header}{nodes}</div>')

    # ── PORTRAIT mode ─────────────────────────────────────────────────────────
    bsz="15px"; nsz="14px"; ssz="9px"; rsz="8px"
    nodes="".join(make_node(s,f'left:{s["x"]}%;top:{s["y"]}%;',bsz,nsz,ssz,rsz) for s in slots)

    # Portrait SVG — very faint so it never overpowers player text
    portrait_svg=(
        '<svg style="position:absolute;inset:0;width:100%;height:100%;'
        'pointer-events:none;z-index:1;" viewBox="0 0 100 142" preserveAspectRatio="none">'
        + PORTRAIT_SVG + '</svg>')

    depth_html=""
    if not xi_only and depth:
        cards=""
        for p in depth:
            yrs=contract_years(p.get("Contract expires","")); yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p)
            _lo=is_loaned_out(p); _yt=is_youth(p); _esc=is_esc(p)
            col=("#ffffff" if white_names else player_css_color(yrs,loan,_lo,_yt,_esc,esc_blue))
            multi="\U0001f501" if _multi_role(p.get("Position","")) else ""
            pos_t=_tok(p.get("Position",""))
            br=best_role_html(p,df_sc,"8px") if show_roles else ""
            dep_yr = "L" if loan else (f"+{yrs}" if yrs>=0 else "+?")
            cards+=(f'<div style="background:#0d1220;border:1px solid #1f2937;'
                    f'padding:5px 9px;min-width:100px;text-align:center;flex-shrink:0;">'
                    f'<div style="color:{col};font-size:11px;font-weight:700;">'
                    f'{p["Player"]} {dep_yr} {multi}</div>'
                    f'<div style="color:#6b7280;font-size:7px;">{pos_t}</div>{br}</div>')
        depth_html=(f'<div style="margin-top:10px;border-top:1px solid #1f2937;padding-top:8px;">'
                    f'<div style="font-size:9px;font-weight:800;letter-spacing:.18em;color:#6b7280;'
                    f'margin-bottom:6px;text-align:center;">DEPTH</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">'
                    f'{cards}</div></div>')

    title_html=(f'<div style="font-weight:900;font-size:20px;letter-spacing:.05em;'
                f'text-transform:uppercase;text-align:center;margin-bottom:4px;">'
                f'{team} Squad Depth</div>')
    header_html=(f'<div style="display:flex;justify-content:space-between;'
                 f'align-items:baseline;margin-bottom:4px;font-size:9px;color:#6b7280;">'
                 f'<span>{league}</span><span>{formation}</span></div>')
    esc_legend_p=f'<span style="color:#60a5fa;">ESC</span>' if esc_blue else ""
    legend_bar=(f'<div style="text-align:center;font-size:8px;color:#6b7280;margin-top:6px;">'
                f'Name + contract years{legend_text()} \u00b7 \U0001f501=4+ positions</div>'
                f'<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'
                f'font-size:9px;font-weight:700;margin-top:4px;">'
                f'<span style="color:#fff;">Contracted</span>'
                f'<span style="color:#f59e0b;">Final Year</span>'
                f'<span style="color:#ef4444;">Out of Contract</span>'
                f'<span style="color:#22c55e;">On Loan</span>'
                f'<span style="color:#c084fc;">Loaned Out</span>'
                f'<span style="color:#9ca3af;">Youth</span>{esc_legend_p}</div>')

    # The pitch uses padding-bottom:142% to maintain aspect ratio.
    # For PNG capture we need an EXPLICIT pixel height.
    # We embed a data-width attribute that the PNG capture script can use
    # to work out the real rendered height.
    return (f'<div id="pitch-root" data-pitch-w="{pitch_width_px}" '
            f'style="font-family:Montserrat,sans-serif;color:#fff;background:{BG};padding:0 4px 10px;">'
            f'{title_html}{header_html}'
            f'<div id="pitch-field" style="position:relative;background:{BG};padding-bottom:142%;'
            f'overflow:hidden;border:1px solid #1a2540;">'
            f'{portrait_svg}{nodes}</div>'
            f'{depth_html}{legend_bar}</div>')

# ── HTML wrapper for standalone download ─────────────────────────────────────
FONT_URL="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap"
def make_mobile_html_page(pitch_html:str, team:str)->str:
    """Full-size pitch optimised for iPhone Safari — viewport meta, full-width."""
    BG="#0a0f1c"
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>{team} Squad Depth</title>
<style>
@import url('{FONT_URL}');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};font-family:Montserrat,sans-serif;overflow-x:hidden;}}
#pitch-root{{width:100vw!important;padding:0!important;}}
#pitch-field{{height:calc(100vw * 1.42)!important;padding-bottom:0!important;}}
</style></head>
<body>{pitch_html}</body></html>"""

def make_html_page(pitch_html:str, team:str, canva:bool, pitch_w:int=560)->str:
    """Standalone HTML page that renders identically to Streamlit."""
    BG="#0a0f1c"
    if canva:
        body_style=(f"margin:0;background:{BG};font-family:Montserrat,sans-serif;"
                    f"display:flex;justify-content:center;align-items:flex-start;")
        wrap_style="display:inline-block;"
    else:
        body_style=f"margin:0;background:{BG};font-family:Montserrat,sans-serif;"
        # Fix pitch-field: replace padding-bottom trick with explicit height for standalone
        wrap_style=f"width:{pitch_w}px;margin:0 auto;padding:8px;"
    page_fix_css=""
    if not canva:
        # Force pitch-field to explicit height so it renders correctly in browsers
        page_fix_css=f"#pitch-field{{height:{round(pitch_w*1.45)}px!important;padding-bottom:0!important;}}"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{team} Squad Depth</title>
<style>
@import url('{FONT_URL}');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{{body_style}}}
{page_fix_css}
</style></head>
<body><div style="{wrap_style}">{pitch_html}</div></body></html>"""

def make_png_page(pitch_html:str, team:str, canva:bool, pitch_w:int=560)->str:
    """HTML page that auto-captures itself as PNG using html2canvas."""
    BG="#0a0f1c"
    # For portrait: capture element has explicit px dimensions
    # For canva: element is already fixed 1920×1080
    if canva:
        cap_w="1920"; cap_h="1080"
        wrap_style="display:inline-block;"
        extra_cfg=""
    else:
        # pitch aspect = 142%, so height = width * 1.42 approximately
        # Add ~120px for title + legend areas
        est_h = round(pitch_w * 1.42) + 160
        cap_w=str(pitch_w); cap_h=str(est_h)
        wrap_style=f"width:{pitch_w}px;"
        # Force the pitch-field div to actual pixels (removes padding-bottom hack)
        extra_cfg=f"""
  // Fix padding-bottom aspect-ratio trick for html2canvas
  var pf = el.querySelector('#pitch-field');
  if(pf){{ pf.style.paddingBottom='0'; pf.style.height='{round(pitch_w*1.42)}px'; }}"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Saving PNG\u2026</title>
<style>
@import url('{FONT_URL}');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};font-family:Montserrat,sans-serif;}}
#msg{{color:#fff;font-size:13px;text-align:center;padding:10px;letter-spacing:.12em;
      font-family:Montserrat,sans-serif;font-weight:700;}}
</style></head>
<body>
<div id="msg">GENERATING PNG \u2014 PLEASE WAIT\u2026</div>
<div id="capture" style="{wrap_style}">{pitch_html}</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
document.fonts.ready.then(function(){{
  setTimeout(function(){{
    var el = document.getElementById('capture');
    {extra_cfg}
    html2canvas(el, {{
      backgroundColor: '{BG}',
      scale: 2,
      useCORS: true,
      allowTaint: false,
      logging: false,
      width: {cap_w if canva else "el.offsetWidth"},
      height: {cap_h if canva else "el.offsetHeight"},
      windowWidth: {cap_w if canva else "el.offsetWidth"},
      windowHeight: {cap_h if canva else "el.offsetHeight"}
    }}).then(function(canvas){{
      var a = document.createElement('a');
      a.download = '{team.replace(" ","_")}_squad_depth.png';
      a.href = canvas.toDataURL('image/png');
      a.click();
      document.getElementById('msg').textContent = '\u2713 PNG SAVED \u2014 YOU CAN CLOSE THIS TAB';
    }}).catch(function(e){{
      document.getElementById('msg').textContent = 'ERROR: ' + e;
    }});
  }}, 1500);
}});
</script></body></html>"""

# ── Server-side PNG render (Mobile view only) ────────────────────────────────
# The HTML downloads above need a desktop browser: html2canvas has to run, and on
# iPad/iPhone Safari just shows an "Open with" prompt with nothing to open. So when
# Mobile view is on we re-render the very same pitch layout server-side with
# matplotlib and hand back a real PNG. Desktop's HTML/html2canvas route is untouched.
from io import BytesIO

_MPL_OK=True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Rectangle as _Rect, Ellipse as _Ell, FancyBboxPatch as _FBox
    from matplotlib import font_manager as _fm
except Exception:                      # matplotlib missing → PNG button hides itself
    _MPL_OK=False

def _png_font()->str:
    """Montserrat if the host has it, otherwise whatever sans is installed."""
    if not _MPL_OK: return "sans-serif"
    try: have={f.name for f in _fm.fontManager.ttflist}
    except Exception: return "sans-serif"
    for n in ("Montserrat","DejaVu Sans","Liberation Sans","Arial"):
        if n in have: return n
    return "sans-serif"

def _mpl_col(c:str)->str:
    """score_to_color() emits rgb(r,g,b); matplotlib wants hex."""
    m=re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)",str(c))
    if m: return "#{:02x}{:02x}{:02x}".format(*[int(g) for g in m.groups()])
    return c

def _mpl_w(v):
    """CSS numeric weight → a weight DejaVu/Montserrat can actually render."""
    try: return "bold" if int(v)>=600 else "normal"
    except Exception: return v

class _PngCanvas:
    """CSS-pixel drawing surface: origin top-left, 1 unit = 1 CSS px, saved at `scale`×."""
    def __init__(self,w,h,scale,bg,font):
        self.w=float(w); self.h=float(h); self.S=float(scale); self.font=font; self.bg=bg
        self.fig=_plt.figure(figsize=(self.w/100.0,self.h/100.0),dpi=100*self.S)
        self.fig.patch.set_facecolor(bg)
        self.ax=self.fig.add_axes([0,0,1,1])
        self.ax.set_xlim(0,self.w); self.ax.set_ylim(self.h,0)
        self.ax.set_facecolor(bg); self.ax.axis("off")
        self.r=self.fig.canvas.get_renderer()

    def set_height(self,h):
        self.h=float(h)
        self.fig.set_size_inches(self.w/100.0,self.h/100.0)
        self.ax.set_ylim(self.h,0)
        self.r=self.fig.canvas.get_renderer()

    def text(self,x,y,s,px,color="#ffffff",weight="normal",ha="center",va="center",z=10,alpha=1.0):
        return self.ax.text(x,y,s,fontsize=px*0.72,color=_mpl_col(color),fontweight=_mpl_w(weight),
                            ha=ha,va=va,family=self.font,zorder=z,alpha=alpha)

    def tw(self,s,px,weight="normal")->float:
        """Rendered width of `s` in CSS px (font metrics don't depend on figure size)."""
        if not s: return 0.0
        t=self.ax.text(0,0,s,fontsize=px*0.72,fontweight=_mpl_w(weight),family=self.font)
        try: w=t.get_window_extent(self.r).width/self.S
        except Exception: w=len(str(s))*px*0.55
        t.remove(); return float(w)

    def rect(self,x,y,w,h,fc="none",ec="none",lw=1.0,z=1,alpha=1.0,radius=0):
        if radius>0:
            p=_FBox((x+radius,y+radius),max(w-2*radius,0.1),max(h-2*radius,0.1),
                    boxstyle=f"round,pad={radius},rounding_size={radius}",
                    facecolor=_mpl_col(fc) if fc!="none" else "none",
                    edgecolor=_mpl_col(ec) if ec!="none" else "none",
                    linewidth=lw*0.72,alpha=alpha,zorder=z)
        else:
            p=_Rect((x,y),w,h,facecolor=_mpl_col(fc) if fc!="none" else "none",
                    edgecolor=_mpl_col(ec) if ec!="none" else "none",
                    linewidth=lw*0.72,alpha=alpha,zorder=z)
        self.ax.add_patch(p); return p

    def line(self,x1,y1,x2,y2,color="#9ca3af",lw=1.0,z=1,alpha=1.0):
        self.ax.plot([x1,x2],[y1,y2],color=_mpl_col(color),linewidth=lw*0.72,
                     alpha=alpha,zorder=z,solid_capstyle="butt")

    def ellipse(self,cx,cy,rx,ry,ec="#9ca3af",fc="none",lw=1.0,z=1,alpha=1.0):
        e=_Ell((cx,cy),2*rx,2*ry,facecolor=_mpl_col(fc) if fc!="none" else "none",
               edgecolor=_mpl_col(ec) if ec!="none" else "none",
               linewidth=lw*0.72,alpha=alpha,zorder=z)
        self.ax.add_patch(e); return e

    def png(self)->bytes:
        buf=BytesIO()
        self.fig.savefig(buf,format="png",facecolor=self.bg)
        _plt.close(self.fig); buf.seek(0); return buf.getvalue()

def render_pitch_png(
    team:str, league:str, formation:str,
    slots:list, slot_map:dict, depth:list, df_sc,
    show_mins:bool, show_goals:bool, show_assists:bool,
    show_positions:bool, show_roles:bool, xi_only:bool, canva:bool,
    pitch_width_px:int=700,
    white_names:bool=False,
    show_contracts:bool=True,
    best_role_only:bool=False,
    esc_blue:bool=False,
    scale:float=2.0,
)->bytes:
    """Matplotlib twin of render_pitch(): same slots, same players, same colours.

    Emoji are swapped for ASCII (⚽→G, 🅰→A, 🔁→*) because the fonts available on
    Streamlit Cloud have no emoji glyphs — the legend says the same thing.
    """
    BG="#0a0f1c"; FONT=_png_font()
    _hpo=st.session_state.get("hide_pos_override",set())
    _hop=st.session_state.get("hide_oop_players",set())

    # ── per-player bits (mirrors make_node) ───────────────────────────────────
    def p_color(p)->str:
        if white_names: return "#ffffff"
        return player_css_color(contract_years(p.get("Contract expires","")),is_loan(p),
                                is_loaned_out(p),is_youth(p),is_esc(p),esc_blue)

    def p_suffix(p)->str:
        yrs=contract_years(p.get("Contract expires","")); yr_str=f"+{yrs}" if yrs>=0 else "+?"
        multi=" *" if _multi_role(p.get("Position","")) else ""
        oop=(f" ({p['_primary_pos']})" if (p.get("_show_pos") and p.get("_key","") not in _hpo
             and p.get("_key","") not in _hop) else "")
        if is_loan(p):
            return f" L{oop}{multi}" if show_contracts else f"{oop}{multi}"
        return f"{(yr_str if show_contracts else '')}{oop}{multi}"

    def p_stats(p)->str:
        parts=[]
        if show_mins: parts.append(f"{int(float(p.get('Minutes played') or 0))}′")
        if show_goals:
            g=float(p.get("Goals") or 0)
            if g>0: parts.append(f"{int(g)}G")
        if show_assists:
            a=float(p.get("Assists") or 0)
            if a>0: parts.append(f"{int(a)}A")
        return " ".join(parts)

    def p_roles(p,best_only:bool)->list:
        """[(role, score, name_colour, score_colour, is_best)] — as all/best_role_html."""
        if df_sc is None or getattr(df_sc,"empty",True): return []
        rows=df_sc[df_sc["Player"]==p.get("Player","")]
        if rows.empty: return []
        row=rows.iloc[0]; rk=_role_key(p.get("Position",""))
        scores={}
        for rn in ROLE_BUCKETS.get(rk,{}):
            v=row.get(f"_rs_{rn}",np.nan)
            if isinstance(v,(int,float)) and not np.isnan(float(v)): scores[rn]=float(v)
        if not scores: return []
        best=max(scores,key=scores.get)
        if best_only:
            return [(best,scores[best],"#7a8494",score_to_color(scores[best]),True)]
        out=[]
        for rn,sc in sorted(scores.items(),key=lambda x:-x[1]):
            c=score_to_color(sc); b=(rn==best)
            out.append((rn,sc,c if b else "#7a8494",c,b))
        return out

    def wrap(cv,s,px,weight,maxw)->list:
        """Word-wrap like the CSS max-width cap on edge nodes."""
        if maxw<=0 or cv.tw(s,px,weight)<=maxw: return [s]
        words=str(s).split(" "); out=[]; cur=""
        for w in words:
            trial=(cur+" "+w).strip()
            if cur and cv.tw(trial,px,weight)>maxw: out.append(cur); cur=w
            else: cur=trial
        if cur: out.append(cur)
        return out or [s]

    # ── node block: list of (height, draw(cv, x_anchor, y_top)) ───────────────
    def node_block(cv,slot,bsz,nsz,ssz,rsz,ta,badge_style,name_maxw=0.0,zbase=10):
        """Build one slot's stack as [(height, draw(cv, x_anchor, y_top)), ...].

        Role lines share the node's content width so the scores line up in a column,
        exactly like the flex `justify-content:space-between` rows do in the HTML.
        """
        lines=[]; widths=[]; nodew=[0.0]
        def gap(h): lines.append((h,None))
        def txt(s,px,color,weight,lh=1.45,align=None):
            h=px*lh; a=align or ta; widths.append(cv.tw(s,px,weight))
            def d(c,x,y,_s=s,_px=px,_c=color,_w=weight,_h=h,_a=a):
                c.text(x,y+_h/2.0,_s,_px,_c,_w,ha=_a,va="center",z=zbase+1)
            lines.append((h,d))
        def role(rn,sc,cn,cs,px,minw,lh=1.4):
            h=px*lh; wt="bold" if cn!="#7a8494" else "normal"; sv=str(int(sc))
            widths.append(cv.tw(rn,px,wt)+6+cv.tw(sv,px,"bold"))
            def d(c,x,y,_rn=rn,_sv=sv,_cn=cn,_cs=cs,_px=px,_h=h,_mw=minw,_wt=wt):
                tot=max(_mw,nodew[0])
                left=x if ta=="left" else (x-tot if ta=="right" else x-tot/2.0)
                c.text(left,y+_h/2.0,_rn,_px,_cn,_wt,ha="left",va="center",z=zbase+1)
                c.text(left+tot,y+_h/2.0,_sv,_px,_cs,"bold",ha="right",va="center",z=zbase+1)
            lines.append((h,d))

        # slot badge
        if badge_style=="portrait":
            bh=bsz*1.2+4+4                                   # padding 2px + border 2px
            bw=cv.tw(slot["label"],bsz,"bold")+16+4
            def dbadge(c,x,y,_l=slot["label"],_px=bsz,_h=bh,_w=bw):
                left=x-_w/2.0 if ta=="center" else (x if ta=="left" else x-_w)
                c.rect(left,y,_w,_h,fc="#0a0f1c",ec="#ef4444",lw=2,z=zbase,alpha=.97)
                c.text(left+_w/2.0,y+_h/2.0,_l,_px,"#ef4444","bold",ha="center",va="center",z=zbase+1)
            lines.append((bh,dbadge)); gap(3)
        else:
            bh=bsz*1.2+6
            bw=cv.tw(slot["label"],bsz,"bold")+24
            def dbadge(c,x,y,_l=slot["label"],_px=bsz,_h=bh,_w=bw):
                left=x-_w/2.0 if ta=="center" else (x if ta=="left" else x-_w)
                c.rect(left,y,_w,_h,fc="#b8bfc9",lw=0,z=zbase,radius=8)
                c.text(left+_w/2.0,y+_h/2.0,_l,_px,"#1f2937","bold",ha="center",va="center",z=zbase+1)
            lines.append((bh,dbadge)); gap(5)
        widths.append(bw)

        ps_all=slot_map.get(slot["id"],[])
        ps=ps_all[:1] if xi_only else ps_all
        started=False; real=0
        for p in ps:
            if p.get("_is_ns"):
                col=p.get("_ns_color","#ef4444")
                if started: gap(4)
                txt((p.get("_ns_label") or "NEW SIGNING").upper(),nsz,col,"bold",1.4)
                if p.get("_ns_sub"): txt(p["_ns_sub"],rsz,col,"normal",1.3)
                started=True
                continue
            ri=real; real+=1
            if started: gap(5)
            col=p_color(p); fw="bold" if ri==0 else "normal"
            full=f'{p["Player"]} {p_suffix(p)}' if badge_style=="portrait" else f'{p["Player"]}{p_suffix(p)}'
            for ln in wrap(cv,full,nsz,fw,name_maxw):
                txt(ln,nsz,col,fw,1.45)
            if badge_style=="portrait":
                allp=", ".join(_all_toks(p.get("Position","")))
                if show_positions and allp: txt(allp,ssz,"#9ca3af","normal",1.2)
                sh=p_stats(p)
                if sh: txt(sh,ssz,"#ffffff","normal",1.2)
            if show_roles:
                rl=p_roles(p,best_role_only or ri>0)
                if rl: gap(2)
                for rn,sc,cn,cs,_b in rl:
                    role(rn,sc,cn,cs,rsz,90 if badge_style=="portrait" else 110)
            started=True
        if not started:
            txt("—",ssz,"#1f2937" if badge_style=="portrait" else "#4b5563","normal",1.2)
        # node width = widest line, min-width 80 / max-width 115 on portrait edge nodes
        w=max(widths) if widths else 0.0
        if badge_style=="portrait":
            w=max(80.0,w)
            if name_maxw>0: w=min(w,name_maxw)
        nodew[0]=w
        return lines

    def draw_block(cv,lines,x,y_top):
        y=y_top
        for h,d in lines:
            if d is not None: d(cv,x,y)
            y+=h

    def legend_tail()->str:
        s=""
        if show_mins:    s+=" · ′=mins"
        if show_goals:   s+=" · G=goals"
        if show_assists: s+=" · A=assists"
        return s

    LEGEND=[("Under Contract","#ffffff"),("Out of Contract","#ef4444"),("Final Year","#f59e0b"),
            ("On Loan","#22c55e"),("Loaned Out","#c084fc"),("Youth","#9ca3af")]
    if esc_blue: LEGEND=LEGEND+[("ESC","#60a5fa")]

    # ── CANVA 1920×1080 ───────────────────────────────────────────────────────
    if canva:
        cv=_PngCanvas(CANVA_W,CANVA_H,scale,BG,FONT)
        cv.rect(CPX,CPY,CPW,CPH,fc="#0d1820",lw=0,z=1,alpha=.6)
        cv.rect(CPX,CPY,CPW,CPH,ec="#374151",lw=2,z=2)
        ccx=CPX+CPW//2; ccy=CPY+CPH//2
        cv.line(ccx,CPY,ccx,CPY+CPH,"#374151",1.5,z=2)
        cv.ellipse(ccx,ccy,CP_CR,CP_CR,"#374151",lw=1.5,z=2)
        cv.ellipse(ccx,ccy,5,5,"none","#374151",z=2)
        pa_y=CPY+round((CPH-CP_PAH)/2); ga_y=CPY+round((CPH-CP_GAH)/2)
        cv.rect(CPX,pa_y,CP_PAW,CP_PAH,ec="#374151",lw=1.5,z=2)
        cv.rect(CPX,ga_y,CP_GAW,CP_GAH,ec="#374151",lw=1,z=2)
        cv.ellipse(CPX+round(CPW*0.08),ccy,4,4,"none","#374151",z=2)
        cv.rect(CPX+CPW-CP_PAW,pa_y,CP_PAW,CP_PAH,ec="#374151",lw=1.5,z=2)
        cv.rect(CPX+CPW-CP_GAW,ga_y,CP_GAW,CP_GAH,ec="#374151",lw=1,z=2)
        cv.ellipse(CPX+CPW-round(CPW*0.08),ccy,4,4,"none","#374151",z=2)

        hy=16+21*0.65
        cv.text(CPX,hy,f"Name + contract years{legend_tail()} · *=4+ positions",
                21,"#6b7280","normal",ha="left",va="center",z=20)
        cur=CPX+CPW
        tail=[(f"{league} · {formation}","#6b7280","normal")]+ \
             [(t,c,"bold") for t,c in reversed(LEGEND)]
        for t,c,w in tail:
            cv.text(cur,hy,t,21,c,w,ha="right",va="center",z=20)
            cur-=cv.tw(t,21,w)+10

        for si,s in enumerate(slots):
            lx,ly,tx,ta=canva_slot_px(float(s["x"]),float(s["y"]),s["id"])
            lines=node_block(cv,s,32,29,21,20,ta,"canva",zbase=20+si*4)
            H=sum(h for h,_ in lines)
            y=ly if tx.endswith(",0)") else (ly-H if tx.endswith(",-100%)") else ly-H/2.0)
            draw_block(cv,lines,lx,y)
        return cv.png()

    # ── PORTRAIT ──────────────────────────────────────────────────────────────
    W=float(pitch_width_px); PADX=4.0
    fw=W-2*PADX; fh=fw*1.42
    cv=_PngCanvas(W,fh+400,scale,BG,FONT)          # height fixed up once measured

    title_h=20*1.25; head_h=9*1.3
    field_y=title_h+4+head_h+4
    y=field_y+fh

    # depth strip
    depth_rows=[]; card_h=0.0
    if not xi_only and depth:
        cards=[]
        for p in depth:
            yrs=contract_years(p.get("Contract expires","")); loan=is_loan(p)
            dep_yr="L" if loan else (f"+{yrs}" if yrs>=0 else "+?")
            multi="*" if _multi_role(p.get("Position","")) else ""
            name=f'{p["Player"]} {dep_yr} {multi}'.strip()
            pos=_tok(p.get("Position",""))
            rl=p_roles(p,True) if show_roles else []
            wneed=max(cv.tw(name,11,"bold"),cv.tw(pos,7,"normal"),
                      (cv.tw(rl[0][0],8,"bold")+6+cv.tw(str(int(rl[0][1])),8,"bold")) if rl else 0)
            cards.append({"name":name,"col":p_color(p),"pos":pos,"role":rl,
                          "w":max(100.0,wneed+18)})
        card_h=5+11*1.2+7*1.2+(2+8*1.4 if show_roles else 0)+5
        row=[]; rw=0.0
        for c in cards:
            add=c["w"]+(6 if row else 0)
            if row and rw+add>fw: depth_rows.append((row,rw)); row=[]; rw=0.0; add=c["w"]
            row.append(c); rw+=add
        if row: depth_rows.append((row,rw))
        y+=10+1+8+9*1.25+6+len(depth_rows)*card_h+max(0,len(depth_rows)-1)*6

    # legend bar
    leg_line=f"Name + contract years{legend_tail()} · *=4+ positions"
    leg_rows=[]; row=[]; rw=0.0
    for t,c in LEGEND:
        w=cv.tw(t,9,"bold"); add=w+(12 if row else 0)
        if row and rw+add>fw: leg_rows.append((row,rw)); row=[]; rw=0.0; add=w
        row.append((t,c,w)); rw+=add
    if row: leg_rows.append((row,rw))
    y+=6+8*1.25+4+len(leg_rows)*9*1.25
    H=y+10
    cv.set_height(H)

    # header
    cv.text(W/2.0,title_h/2.0,f"{team} Squad Depth".upper(),20,"#ffffff","bold",ha="center",va="center",z=20)
    hy=title_h+4+head_h/2.0
    cv.text(PADX,hy,str(league),9,"#6b7280","normal",ha="left",va="center",z=20)
    cv.text(W-PADX,hy,str(formation),9,"#6b7280","normal",ha="right",va="center",z=20)

    # pitch field + markings (PORTRAIT_SVG, viewBox 0 0 100 142, stretched)
    cv.rect(PADX,field_y,fw,fh,fc=BG,ec="#1a2540",lw=1,z=1)
    sx=fw/100.0; sy=fh/142.0; avg=(sx+sy)/2.0
    X=lambda v:PADX+v*sx; Y=lambda v:field_y+v*sy
    LC="#9ca3af"; OP=0.18
    cv.rect(X(2),Y(2),96*sx,138*sy,ec=LC,lw=1.2*avg,z=2,alpha=OP)
    cv.line(X(2),Y(71),X(98),Y(71),LC,0.8*avg,z=2,alpha=OP)
    cv.ellipse(X(50),Y(71),10*sx,10*sy,LC,lw=0.8*avg,z=2,alpha=OP)
    cv.ellipse(X(50),Y(71),1.2*sx,1.2*sy,"none",LC,z=2,alpha=OP)
    cv.rect(X(22),Y(2),56*sx,18*sy,ec=LC,lw=0.8*avg,z=2,alpha=OP)
    cv.rect(X(36),Y(2),28*sx,7*sy,ec=LC,lw=0.6*avg,z=2,alpha=OP)
    cv.ellipse(X(50),Y(14),0.9*sx,0.9*sy,"none",LC,z=2,alpha=OP)
    cv.rect(X(22),Y(122),56*sx,18*sy,ec=LC,lw=0.8*avg,z=2,alpha=OP)
    cv.rect(X(36),Y(133),28*sx,7*sy,ec=LC,lw=0.6*avg,z=2,alpha=OP)
    cv.ellipse(X(50),Y(126),0.9*sx,0.9*sy,"none",LC,z=2,alpha=OP)

    # nodes
    for si,s in enumerate(slots):
        sxp=float(s.get("x",50)); is_edge=(sxp<20 or sxp>80)
        lines=node_block(cv,s,15,14,9,8,"center","portrait",
                         name_maxw=115.0 if is_edge else 0.0,zbase=20+si*4)
        Hb=sum(h for h,_ in lines)
        cy=field_y+float(s.get("y",50))/100.0*fh
        draw_block(cv,lines,X(sxp),cy-Hb/2.0)

    # depth strip
    y=field_y+fh
    if depth_rows:
        y+=10
        cv.line(PADX,y,W-PADX,y,"#1f2937",1,z=3)
        y+=8
        cv.text(W/2.0,y+9*1.25/2.0,"DEPTH",9,"#6b7280","bold",ha="center",va="center",z=10)
        y+=9*1.25+6
        for row,rw in depth_rows:
            x=PADX+(fw-rw)/2.0
            for c in row:
                cv.rect(x,y,c["w"],card_h,fc="#0d1220",ec="#1f2937",lw=1,z=4)
                cxm=x+c["w"]/2.0; ty=y+5
                cv.text(cxm,ty+11*1.2/2.0,c["name"],11,c["col"],"bold",ha="center",va="center",z=6)
                ty+=11*1.2
                cv.text(cxm,ty+7*1.2/2.0,c["pos"],7,"#6b7280","normal",ha="center",va="center",z=6)
                ty+=7*1.2
                if c["role"]:
                    rn,sc,cn,cs,_b=c["role"][0]; ty+=2
                    tot=c["w"]-18; left=cxm-tot/2.0   # card inner width, as the flex row
                    cv.text(left,ty+8*1.4/2.0,rn,8,cn,"normal",ha="left",va="center",z=6)
                    cv.text(left+tot,ty+8*1.4/2.0,str(int(sc)),8,cs,"bold",ha="right",va="center",z=6)
                x+=c["w"]+6
            y+=card_h+6
        y-=6

    # legend bar
    y+=6
    cv.text(W/2.0,y+8*1.25/2.0,leg_line,8,"#6b7280","normal",ha="center",va="center",z=10)
    y+=8*1.25+4
    for row,rw in leg_rows:
        x=PADX+(fw-rw)/2.0
        for t,c,w in row:
            cv.text(x,y+9*1.25/2.0,t,9,c,"bold",ha="left",va="center",z=10)
            x+=w+12
        y+=9*1.25
    return cv.png()

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in {"slot_map":{},"depth":[],"move_player":None,"df":None,"df_sc":None,
             "last_team":None,"last_formation":None,"edit_contract_player":None,
             "hide_pos_override":set(),"new_signing":{},"esc_players":set(),
             "hide_oop_players":set()}.items():
    if k not in st.session_state: st.session_state[k]=v

def _tog(k,d=False): return st.session_state.get(k,d)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## \u26bd Squad Chart")
    st.markdown("---")
    st.markdown("**DATA**")

    # ── CSV selector — any CSV in repo, ordered by SEASON (not mtime) ────────
    import os
    from pathlib import Path as _Path

    def _season_key(name):
        """First 4-digit year anywhere in the filename: 2026-27WORLDFULL.csv -> 2026.
        Duplicated per repo rather than imported — separate repos, no shared
        package. Mirrors season_utils.py in Scouting-Hub."""
        m = re.search(r"(\d{4})", str(name))
        return int(m.group(1)) if m else -1

    # Player-only: filtered to *WORLDFULL.csv specifically — this app crashes
    # with an unguarded KeyError (df["Position"], df["Player"]) if a
    # WORLDTEAMS*.csv file loads by default, which has neither column.
    #
    # Season from the filename, not st_mtime. mtime was inverted: split_seasons.py
    # writes seasons newest-first and copy2 preserves those timestamps, so
    # 2026-27WORLDFULL.csv had the OLDEST mtime and this defaulted to 2024-25.
    _repo_csvs = sorted(_Path.cwd().glob("*WORLDFULL.csv"), key=lambda f: _season_key(f.name), reverse=True)
    _csv_names = [f.name for f in _repo_csvs]
    if _csv_names:
        preset_choice = st.selectbox("Select dataset", _csv_names, index=0, key="preset_choice")
        PRELOADED = {n: n for n in _csv_names}
    else:
        preset_choice = None
        PRELOADED = {}
    st.markdown("<div style='text-align:center;font-size:9px;color:#4b5563;margin:4px 0;'>— or upload —</div>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    @st.cache_data(show_spinner=False)
    def _load_path(path: str) -> pd.DataFrame:
        df = pd.read_csv(path); df.columns = df.columns.str.strip()
        for c in ["Player","Team","Position","League"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip()
        for c in ["Minutes played","Goals","Assists","Age","xG","xA"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["_ftok"] = df["Position"].apply(_tok); df["_key"] = df["Player"]
        return df

    @st.cache_data(show_spinner=False)
    def _load(f) -> pd.DataFrame:
        df = pd.read_csv(f); df.columns = df.columns.str.strip()
        for c in ["Player","Team","Position","League"]:
            if c in df.columns: df[c] = df[c].astype(str).str.strip()
        for c in ["Minutes played","Goals","Assists","Age","xG","xA"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["_ftok"] = df["Position"].apply(_tok); df["_key"] = df["Player"]
        return df

    # Determine which source to load — upload takes priority over preset
    _active_source = None
    if uploaded:
        _active_source = ("upload", uploaded)
    elif preset_choice and PRELOADED.get(preset_choice):
        _csv_path = PRELOADED[preset_choice]
        _full_path = str(_Path.cwd() / _csv_path)
        if os.path.exists(_full_path):
            _active_source = ("preset", _full_path)
        else:
            st.warning(f"⚠ {_csv_path} not found in repo root")

    # Track which source is loaded so we reset scores when it changes
    _src_key = (uploaded.name if uploaded else None) or preset_choice
    if _active_source:
        if st.session_state.get("_src_key") != _src_key:
            st.session_state.df = None
            st.session_state.df_sc = None
            st.session_state["_src_key"] = _src_key
        if st.session_state.df is None:
            with st.spinner("Loading…"):
                if _active_source[0] == "upload":
                    raw = _load(_active_source[1])
                else:
                    raw = _load_path(_active_source[1])
            st.session_state.df = raw
            st.session_state.df_sc = None
        if st.session_state.df_sc is None:
            with st.spinner("Computing role scores\u2026"):
                st.session_state.df_sc = compute_role_scores(st.session_state.df)
        _lbl = uploaded.name if uploaded else preset_choice
        st.success(f"\u2713 {len(st.session_state.df):,} players \u00b7 {_lbl}")

    st.markdown("---")
    if st.session_state.df is not None:
        df=st.session_state.df
        lgs=["All"]+sorted(df["League"].dropna().unique())
        lg=st.selectbox("League",lgs)
        fdf=df if lg=="All" else df[df["League"]==lg]
        max_mins=int(df["Minutes played"].max()) if "Minutes played" in df.columns else 5000
        min_mins=st.slider("Min minutes played",0,max_mins,0,50)
        fdf=fdf[fdf["Minutes played"]>=min_mins]
        sel_team=st.selectbox("Team",sorted(fdf["Team"].dropna().unique()))
        formation=st.selectbox("Formation",list(FORMATIONS.keys()))

        st.markdown("---")
        st.markdown("**DISPLAY**")
        st.toggle("Minutes played",  True,  key="show_mins")
        st.toggle("Goals",           True,  key="show_goals")
        st.toggle("Assists",         True,  key="show_assists")
        st.toggle("Show positions",  False, key="show_positions")
        st.toggle("Role scores",     True,  key="show_roles")
        st.toggle("Best role only",  False, key="best_role_only")
        st.toggle("XI only",         False, key="xi_only")
        st.toggle("White names",     False, key="white_names")
        st.toggle("Show contracts",  True,  key="show_contracts")
        st.toggle("ESC blue names",  False, key="esc_blue")
        st.toggle("Canva 1920\u00d71080", False, key="canva_mode")
        st.toggle("Mobile view 📱",  False, key="mobile_mode")

        st.markdown("---")
        changed=(sel_team!=st.session_state.last_team or
                 formation!=st.session_state.last_formation)
        if st.button("\U0001f504 Build / Rebuild") or changed:
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
            if st.button("\u2705 Confirm Move"):
                p=mp["player"]; fid=mp["from_slot"]; did=opts[dest_lbl]
                if fid=="_depth":
                    st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=p["_key"]]
                elif fid in st.session_state.slot_map:
                    st.session_state.slot_map[fid]=[x for x in st.session_state.slot_map[fid] if x["_key"]!=p["_key"]]
                st.session_state.slot_map.setdefault(did,[]).append(p)
                st.session_state.move_player=None; st.rerun()
            if st.button("\u274c Cancel Move"):
                st.session_state.move_player=None; st.rerun()

        if st.session_state.edit_contract_player:
            ec=st.session_state.edit_contract_player
            st.markdown(f"**EDIT CONTRACT:** {ec['player']['Player']}")
            new_exp=st.text_input("Expires (YYYY-MM-DD)",
                                  value=ec["player"].get("Contract expires",""),key="new_exp")
            if st.button("\U0001f4be Save Contract"):
                pk=ec["player"]["_key"]
                for sid,ps in st.session_state.slot_map.items():
                    for p in ps:
                        if p["_key"]==pk: p["Contract expires"]=new_exp
                for p in st.session_state.depth:
                    if p["_key"]==pk: p["Contract expires"]=new_exp
                st.session_state.edit_contract_player=None; st.rerun()
            if st.button("\u2716 Cancel Edit"):
                st.session_state.edit_contract_player=None; st.rerun()

        st.markdown("---")
        st.markdown("**ADD PLAYER**")
        nn=st.text_input("Name",key="nn")
        np_=st.selectbox("Position",list(CANONICAL.keys()),key="np_")
        extra_pos=st.text_input("Extra positions (e.g. LCMF,AMF)",key="extra_pos",
                                 help="Comma-separated. 4+ positions = \U0001f501 emoji")
        nm_=st.number_input("Minutes",0,5000,0,10,key="nm_")
        ng_=st.number_input("Goals",0,100,0,key="ng_")
        na_=st.number_input("Assists",0,100,0,key="na_")
        ne_=st.text_input("Contract expires","2026-06-30",key="ne_")
        nl_=st.checkbox("On Loan? (incoming, green)",key="nl_")
        nlo_=st.checkbox("Loaned Out? (purple)",key="nlo_")
        nyt_=st.checkbox("Youth Player? (grey)",key="nyt_")
        sl_opts={f"{s['label']} ({s['id']})":s["id"] for s in FORMATIONS.get(formation,[])}
        ns_=st.selectbox("Add to slot",list(sl_opts.keys()),key="ns_")
        if st.button("\u2795 Add Player") and nn.strip():
            pos_str=np_
            if extra_pos.strip(): pos_str+=","+extra_pos.strip()
            new_p={"Player":nn.strip(),"Position":pos_str,"_key":f"custom_{nn}",
                   "Minutes played":nm_,"Goals":ng_,"Assists":na_,
                   "Contract expires":ne_,"On Loan":"yes" if nl_ else "no",
                   "Loaned Out":"yes" if nlo_ else "no",
                   "Youth Player":"yes" if nyt_ else "no",
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

# Estimate the portrait pitch pixel width from Streamlit's main column
# Streamlit wide layout main area ≈ 1140px; subtracting sidebar (300px) → ~840px usable
# We use 560px as a conservative portrait width (matches typical Streamlit narrow render)
PORTRAIT_W=700

slots=layout_slots(formation, PORTRAIT_W); slot_map=st.session_state.slot_map
depth=st.session_state.depth; df_sc=st.session_state.df_sc
canva=_tog("canva_mode")

pitch=render_pitch(
    team_name,league_nm,formation,slots,slot_map,depth,df_sc,
    _tog("show_mins",True),_tog("show_goals",True),_tog("show_assists",True),
    _tog("show_positions"),_tog("show_roles",True),_tog("xi_only"),canva,
    pitch_width_px=PORTRAIT_W,
    white_names=_tog("white_names"),
    show_contracts=_tog("show_contracts",True),
    best_role_only=_tog("best_role_only"),
    esc_blue=_tog("esc_blue"),
)

_mobile = _tog("mobile_mode")
if canva:
    # Scale 1920×1080 down to fit browser using CSS transform
    st.markdown(
        f'<div id="canva-scaler" style="width:100%;background:#0a0f1c;overflow:hidden;">'
        f'<div style="transform-origin:top left;" id="canva-inner">{pitch}</div></div>'
        f'<script>'
        f'(function(){{'
        f'  var wrap=document.getElementById("canva-scaler");'
        f'  var inner=document.getElementById("canva-inner");'
        f'  function scl(){{'
        f'    var s=wrap.offsetWidth/1920;'
        f'    inner.style.transform="scale("+s+")";'
        f'    wrap.style.height=(1080*s)+"px";'
        f'  }}'
        f'  scl(); window.addEventListener("resize",scl);'
        f'}})()'
        f'</script>',
        unsafe_allow_html=True)
elif _mobile:
    # Same pitch HTML — CSS scale(0.5) halves it on screen only.
    # Container height forced to 50% so no whitespace gap below.
    st.markdown(
        f'<div id="mob-wrap" style="width:100%;overflow:hidden;">'
        f'<div id="mob-inner" style="transform-origin:top left;'
        f'width:200%;transform:scale(0.5);">{pitch}</div></div>'
        f'<script>(function(){{'
        f'  var w=document.getElementById("mob-wrap");'
        f'  var n=document.getElementById("mob-inner");'
        f'  function sz(){{w.style.height=(n.scrollHeight*0.5)+"px";}}'
        f'  setTimeout(sz,200); window.addEventListener("resize",sz);'
        f'}})()</script>',
        unsafe_allow_html=True)
else:
    st.markdown(pitch, unsafe_allow_html=True)

# ── Downloads ─────────────────────────────────────────────────────────────────
html_dl = make_html_page(pitch, team_name, canva, PORTRAIT_W)
png_dl  = make_png_page(pitch, team_name, canva, PORTRAIT_W)

if _mobile:
    mob_dl = make_mobile_html_page(pitch, team_name)
    dl1,dl2,dl3,_=st.columns([1,1,1,1])
else:
    dl1,dl2,_=st.columns([1,1,4])
with dl1:
    st.download_button("\u2b07 HTML", html_dl.encode("utf-8"),
        f"{team_name.replace(' ','_')}_squad_depth.html","text/html")
with dl2:
    st.download_button("\u2b07 PNG",  png_dl.encode("utf-8"),
        f"{team_name.replace(' ','_')}_OPEN_TO_SAVE_PNG.html","text/html",
        help="Download \u2192 open in Chrome/Edge \u2192 PNG auto-saves")
if _mobile:
    with dl3:
        st.download_button("\u2b07 Mobile HTML \U0001f4f1", mob_dl.encode("utf-8"),
            f"{team_name.replace(' ','_')}_mobile.html","text/html",
            help="Full-size pitch — open in Safari on iPhone")
    # ── Real PNG, rendered server-side (mobile only) ───────────────────────
    # Safari can't run the html2canvas page, so mobile gets actual image bytes.
    # The rendered pitch HTML is a perfect cache key: it changes whenever the squad,
    # the formation or any toggle changes, and nothing else does — so we only pay
    # for a re-render when something actually moved.
    #
    # The image, not the download button, is the primary way out on iPad: tapping
    # a download navigates the whole tab to Safari's file preview, and coming back
    # reloads the app and loses your place. Long-pressing the image below saves
    # straight to Photos without ever leaving the page. Same bytes for both.
    if not _MPL_OK:
        st.caption("PNG needs matplotlib")
    else:
        _png_sig=hashlib.md5((pitch+str(canva)).encode("utf-8")).hexdigest()
        if st.session_state.get("_png_sig")!=_png_sig:
            try:
                st.session_state["_png_bytes"]=render_pitch_png(
                    team_name,league_nm,formation,slots,slot_map,depth,df_sc,
                    _tog("show_mins",True),_tog("show_goals",True),_tog("show_assists",True),
                    _tog("show_positions"),_tog("show_roles",True),_tog("xi_only"),canva,
                    pitch_width_px=PORTRAIT_W,
                    white_names=_tog("white_names"),
                    show_contracts=_tog("show_contracts",True),
                    best_role_only=_tog("best_role_only"),
                    esc_blue=_tog("esc_blue"),
                    scale=2.0,
                )
                st.session_state["_png_err"]=None
            except Exception as _pe:
                st.session_state["_png_bytes"]=None
                st.session_state["_png_err"]=str(_pe)
            st.session_state["_png_sig"]=_png_sig
        _png_bytes=st.session_state.get("_png_bytes")
        if _png_bytes:
            with st.expander("📥 Get PNG — long-press the image to save", expanded=False):
                st.image(_png_bytes, width="stretch")
                st.caption("Long-press the image → **Add to Photos**. Stays on this page, "
                           "so you keep your squad exactly as it is.")
                st.download_button("⬇ Download the file instead", _png_bytes,
                    f"{team_name.replace(' ','_')}_squad_depth.png","image/png",
                    help="Safari opens this in its own preview tab — going back reloads the app")
        else:
            st.caption(f"PNG failed: {st.session_state.get('_png_err')}")

# ── Move / Remove / Edit Contract / Reorder ───────────────────────────────────
st.markdown("---")
all_on=[]
for sl in slots:
    for p in slot_map.get(sl["id"],[]):
        _disp=p.get("_ns_label","NEW SIGNING") if p.get("_is_ns") else p["Player"]
        all_on.append({"sid":sl["id"],"lbl":sl["label"],"player":p,"disp":_disp})
for p in depth:
    all_on.append({"sid":"_depth","lbl":"DEPTH","player":p,"disp":p["Player"]})
# Real players only (no NS) for move/edit/ESC
real_on=[e for e in all_on if not e["player"].get("_is_ns")]

if all_on:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>MOVE</div>",
                    unsafe_allow_html=True)
        mv_opts={f"{e['disp']} ({e['lbl']})":e for e in real_on}
        if mv_opts:
            mv_sel=st.selectbox("",list(mv_opts.keys()),key="mv_sel",label_visibility="collapsed")
            if st.button("Select for Move"):
                e=mv_opts[mv_sel]
                st.session_state.move_player={"player":e["player"],"from_slot":e["sid"]}; st.rerun()
    with c2:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>REMOVE</div>",
                    unsafe_allow_html=True)
        rm_opts={f"{e['disp']} ({e['lbl']})":e for e in all_on}
        rm_sel=st.selectbox("",list(rm_opts.keys()),key="rm_sel",label_visibility="collapsed")
        if st.button("\U0001f5d1 Remove"):
            e=rm_opts[rm_sel]; sid=e["sid"]; pk=e["player"]["_key"]
            if sid=="_depth":
                st.session_state.depth=[x for x in st.session_state.depth if x["_key"]!=pk]
            else:
                st.session_state.slot_map[sid]=[x for x in st.session_state.slot_map.get(sid,[]) if x["_key"]!=pk]
            st.rerun()
    with c3:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>EDIT CONTRACT</div>",
                    unsafe_allow_html=True)
        ec_opts={f"{e['disp']} ({e['lbl']})":e for e in real_on}
        if ec_opts:
            ec_sel=st.selectbox("",list(ec_opts.keys()),key="ec_sel",label_visibility="collapsed")
            if st.button("\u270f\ufe0f Edit Contract"):
                e=ec_opts[ec_sel]
                st.session_state.edit_contract_player={"player":e["player"],"sid":e["sid"]}; st.rerun()
    with c4:
        # ESC Toggle for players
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-bottom:3px;'>TOGGLE ESC</div>",
                    unsafe_allow_html=True)
        esc_opts={f"{e['disp']} ({e['lbl']})":e["player"]["_key"] for e in real_on}
        if esc_opts:
            esc_sel=st.selectbox("",list(esc_opts.keys()),key="esc_sel",label_visibility="collapsed")
            pk_esc=esc_opts[esc_sel]
        esc_set=st.session_state.setdefault("esc_players",set())
        is_esc_player=pk_esc in esc_set
        btn_lbl="\u2705 ESC ON" if is_esc_player else "\u274c ESC OFF"
        if st.button(btn_lbl,key="esc_btn"):
            if is_esc_player: esc_set.discard(pk_esc)
            else: esc_set.add(pk_esc)
            st.session_state.esc_players=esc_set; st.rerun()

    # ── Hide out-of-position label for individual players ─────────────────────
    st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-top:14px;margin-bottom:4px;'>TOGGLE POSITION LABEL (e.g. hide DMF tag)</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:8px;color:#374151;margin-bottom:4px;'>Hides the (POS) suffix shown when a player is playing out of position</div>",
                unsafe_allow_html=True)
    hop_c1,hop_c2=st.columns([3,1])
    with hop_c1:
        oop_eligible=[e for e in real_on if e["player"].get("_show_pos")]
        if oop_eligible:
            hop_opts={f"{e['disp']} ({e['lbl']})":e["player"]["_key"] for e in oop_eligible}
            hop_sel=st.selectbox("",list(hop_opts.keys()),key="hop_sel",label_visibility="collapsed")
            pk_hop=hop_opts[hop_sel]
            hop_set=st.session_state.setdefault("hide_oop_players",set())
            is_hidden=pk_hop in hop_set
            hop_btn_lbl="\U0001f441 Showing tag" if not is_hidden else "\U0001f6ab Tag hidden"
            with hop_c2:
                st.markdown("<div style='margin-top:4px;'></div>",unsafe_allow_html=True)
                if st.button(hop_btn_lbl,key="hop_btn"):
                    if is_hidden: hop_set.discard(pk_hop)
                    else: hop_set.add(pk_hop)
                    st.session_state.hide_oop_players=hop_set; st.rerun()
        else:
            st.markdown("<div style='font-size:9px;color:#374151;'>No out-of-position players to toggle</div>",unsafe_allow_html=True)

    # ── Reorder players within a slot ─────────────────────────────────────────
    st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-top:14px;margin-bottom:6px;'>REORDER PLAYERS IN SLOT</div>",
                unsafe_allow_html=True)
    reorder_slots={}
    for sl in slots:
        ps=slot_map.get(sl["id"],[])
        if len(ps)>1:
            reorder_slots[f"{sl['label']} ({sl['id']}) — {len(ps)} entries"]=sl["id"]
    if len(depth)>1:
        reorder_slots[f"DEPTH — {len(depth)} players"]="_depth"

    if reorder_slots:
        ro_c1,ro_c2=st.columns([2,2])
        with ro_c1:
            ro_slot_lbl=st.selectbox("Slot",list(reorder_slots.keys()),key="ro_slot",label_visibility="visible")
            ro_sid=reorder_slots[ro_slot_lbl]
            if ro_sid=="_depth":
                cur_list=list(st.session_state.depth)
            else:
                cur_list=list(st.session_state.slot_map.get(ro_sid,[]))
            ro_player_opts=[f"#{i+1} {'[NS] '+p.get('_ns_label','NEW SIGNING') if p.get('_is_ns') else p['Player']}" for i,p in enumerate(cur_list)]
            ro_player_sel=st.selectbox("Player to move",ro_player_opts,key="ro_player",label_visibility="visible")
            ro_idx=ro_player_opts.index(ro_player_sel)
        with ro_c2:
            st.markdown("<div style='margin-top:24px;'></div>",unsafe_allow_html=True)
            rc1,rc2=st.columns(2)
            with rc1:
                if st.button("⬆ Move Up",key="ro_up") and ro_idx>0:
                    cur_list[ro_idx-1],cur_list[ro_idx]=cur_list[ro_idx],cur_list[ro_idx-1]
                    if ro_sid=="_depth": st.session_state.depth=cur_list
                    else: st.session_state.slot_map[ro_sid]=cur_list
                    st.rerun()
            with rc2:
                if st.button("⬇ Move Down",key="ro_dn") and ro_idx<len(cur_list)-1:
                    cur_list[ro_idx],cur_list[ro_idx+1]=cur_list[ro_idx+1],cur_list[ro_idx]
                    if ro_sid=="_depth": st.session_state.depth=cur_list
                    else: st.session_state.slot_map[ro_sid]=cur_list
                    st.rerun()
            st.markdown(f"<div style='font-size:9px;color:#4b5563;margin-top:6px;'>Position {ro_idx+1} of {len(cur_list)}<br>1st player = starter shown bold</div>",
                        unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:9px;color:#374151;'>No slots with multiple players to reorder</div>",unsafe_allow_html=True)

    # ── New Signing slot label ────────────────────────────────────────────────
    st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-top:14px;margin-bottom:6px;'>NEW SIGNING / TARGET — add to slot</div>",
                unsafe_allow_html=True)
    ns_slot_opts={f"{sl['label']} ({sl['id']})":sl["id"] for sl in slots}
    ns_c1,ns_c2=st.columns([2,2])
    with ns_c1:
        ns_slot_sel=st.selectbox("Slot",list(ns_slot_opts.keys()),key="ns_slot_sel",label_visibility="visible")
        ns_sid=ns_slot_opts[ns_slot_sel]
        # Count existing NS entries in this slot
        _cur_ns_in_slot=[p for p in st.session_state.slot_map.get(ns_sid,[]) if p.get("_is_ns")]
        st.markdown(f"<div style='font-size:9px;color:#4b5563;margin-top:4px;'>{len(_cur_ns_in_slot)} signing(s) in this slot</div>",unsafe_allow_html=True)
    with ns_c2:
        ns_lbl_val=st.text_input("Label (caps)","NEW SIGNING",key="ns_lbl_val",
                                  help="e.g. NEW SIGNING, TARGET, TRIALIST")
        ns_sub_val=st.text_input("Subtitle (optional)","",key="ns_sub_val",
                                  help="e.g. Wide Creator U23")
        ns_col_val=st.selectbox("Colour",["#ef4444","#f97316","#eab308"],
                                 index=0,
                                 format_func=lambda x:{"#ef4444":"Red","#f97316":"Orange","#eab308":"Yellow"}[x],
                                 key="ns_col_val")
    if st.button("🟠 Add to slot",key="ns_add_btn"):
        ns_entry={"_is_ns":True,"Player":"","_key":f"_ns_{ns_sid}_{len(_cur_ns_in_slot)}",
                  "_ns_label":ns_lbl_val.strip() or "NEW SIGNING",
                  "_ns_sub":ns_sub_val.strip(),"_ns_color":ns_col_val,
                  "Position":"","Minutes played":0,"Goals":0,"Assists":0,
                  "Contract expires":"","On Loan":"no","Loaned Out":"no","Youth Player":"no"}
        st.session_state.slot_map.setdefault(ns_sid,[]).append(ns_entry); st.rerun()

    # Show existing NS entries for the selected slot — they're now reorderable via REORDER section above
    _cur_ns_display=[p for p in st.session_state.slot_map.get(ns_sid,[]) if p.get("_is_ns")]
    if _cur_ns_display:
        st.markdown("<div style='font-size:9px;color:#6b7280;letter-spacing:.1em;margin-top:8px;margin-bottom:4px;'>SIGNINGS IN THIS SLOT — remove</div>",unsafe_allow_html=True)
        st.markdown("<div style='font-size:8px;color:#374151;margin-bottom:4px;'>Use REORDER section above to move signings up/down among players</div>",unsafe_allow_html=True)
        for _ni,_sn in enumerate(_cur_ns_display):
            _lbl=_sn.get("_ns_label","NEW SIGNING")
            _sub=_sn.get("_ns_sub","")
            _disp=f"{_ni+1}. {_lbl}" + (f" — {_sub}" if _sub else "")
            _nc1,_nc2=st.columns([4,1])
            with _nc1:
                st.markdown(f"<div style='font-size:10px;color:#c084fc;padding-top:6px;'>{_disp}</div>",unsafe_allow_html=True)
            with _nc2:
                if st.button("🗑",key=f"ns_rm_{ns_sid}_{_ni}"):
                    pk=_sn.get("_key")
                    st.session_state.slot_map[ns_sid]=[x for x in st.session_state.slot_map[ns_sid] if x.get("_key")!=pk]
                    st.rerun()


# ── Full squad ─────────────────────────────────────────────────────────────────
if st.session_state.df is not None and team_name:
    with st.expander("\U0001f4cb Full Squad"):
        tdf3=st.session_state.df[st.session_state.df["Team"]==team_name]
        show_c=[c for c in ["Player","Position","Minutes played","Goals","Assists",
                             "Market value","Contract expires","Age"] if c in tdf3.columns]
        st.dataframe(tdf3[show_c].sort_values("Minutes played",ascending=False).reset_index(drop=True),
                     use_container_width=True)
