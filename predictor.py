# predictor.py
# requirements: pandas, numpy
import pandas as pd, numpy as np
from math import exp, factorial, isfinite
from itertools import product
import re
import tkinter as tk
from tkinter import ttk, messagebox

# ==== CONFIG ====
LEAGUE_FILES = [
    # Serie A
    "data/serie-a/season-2324.csv",
    "data/serie-a/season-2425.csv",
    "data/serie-a/season-2526.csv",
    "data/serie-a/season-2223.csv",
    "data/serie-a/season-2122.csv",
    "data/serie-a/season-2021.csv",
    # LaLiga
    "data/laliga/SP1.csv","data/laliga/SP2.csv","data/laliga/SP3.csv",
    "data/laliga/SP4.csv","data/laliga/SP5.csv","data/laliga/SP6.csv",
    # Premier
    "data/premier/E0.csv","data/premier/E1.csv","data/premier/E2.csv",
    "data/premier/E3.csv","data/premier/E4.csv","data/premier/E5.csv",
    # Serie B
    "data/serie-b/I2.csv", 
    "data/serie-b/I2 (1).csv",
    "data/serie-b/I2 (2).csv",
    "data/serie-b/I2 (3).csv",
    "data/serie-b/I2 (4).csv",
    "data/serie-b/I2 (5).csv", 
]

DC_TAU = 0.0075
MAX_GOALS = 10

# Pesi (quanto le feature extra modificano λ gol)
W_SOT = 0.35
W_SHOTS = 0.15
W_CORNERS = 0.10

# Limiti quota primo tempo
HT_SHARE_MIN = 0.35
HT_SHARE_MAX = 0.60

# Decadimento temporale (mezzo-vita in mesi)
HALF_LIFE_MONTHS = 24.0

# Stelline estetiche
BADGE_STARS = [(0.80, "⭐⭐⭐"), (0.70, "⭐⭐"), (0.60, "⭐")]

# Quanti pronostici stampare per sezione per match
TOP_PER_SECTION = 5

# ====== Utility ======
def pct(x): return f"{x*100:.1f}%"
def poisson_pmf(k, lam): return (lam**k) * exp(-lam) / factorial(k)
def poisson_cdf(k, lam): return sum(poisson_pmf(i, lam) for i in range(k+1))
def dc_corr(i,j,tau=DC_TAU):
    if i==0 and j==0: return 1 - tau
    if i==0 and j==1: return 1 + tau
    if i==1 and j==0: return 1 + tau
    if i==1 and j==1: return 1 - tau
    return 1.0
def safe_ratio(a, b, default=1.0):
    try:
        if b is None or b == 0 or not isfinite(b): return default
        if a is None or not isfinite(a): return default
        return max(1e-9, a / b)
    except Exception:
        return default
def clamp(x, lo, hi): return max(lo, min(hi, x))
def badge(p: float) -> str:
    for thr, b in BADGE_STARS:
        if p >= thr: return b
    return ""
def months_between(late: pd.Timestamp, early: pd.Timestamp) -> float:
    return max(0.0, (late - early).days / 30.4375)
def wmean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float); w = weights.astype(float)
    mask = v.notna() & w.notna()
    if not mask.any(): return np.nan
    sw = w[mask].sum()
    if sw <= 0: return np.nan
    return (v[mask] * w[mask]).sum() / sw

# ====== Loader & standardizzazione ======
ALIAS_RENAME = {
    "Home": "HomeTeam", "Away": "AwayTeam",
    "HomeGoals": "FTHG", "AwayGoals": "FTAG",
    "Result": "FTR", "HTHomeGoals": "HTHG", "HTAwayGoals": "HTAG", "HTResult": "HTR",
}
REQUIRED_BASE = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
OPTIONAL_BASE = ["FTR","HTHG","HTAG","HTR"]
EXTRA_COLS = ["HS","AS","HST","AST","HC","AC","HY","AY","HR","AR","Referee"]

def infer_league_key(path: str) -> str:
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if "data" in parts:
        i = parts.index("data")
        if i+1 < len(parts):
            return parts[i+1].lower()
    return parts[-2].lower() if len(parts) >= 2 else "unknown"

def load_all_with_league(files):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = df.rename(columns=ALIAS_RENAME)
            df["__LeagueKey__"] = infer_league_key(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Non riesco a leggere {f}: {e}")
    if not frames:
        raise SystemExit("Nessun CSV valido trovato in LEAGUE_FILES.")
    df = pd.concat(frames, ignore_index=True)
    missing_min = [c for c in REQUIRED_BASE if c not in df.columns]
    if missing_min:
        raise SystemExit(f"Mancano colonne indispensabili: {missing_min}")
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"]>df["FTAG"], "H", np.where(df["FTHG"]<df["FTAG"], "A", "D"))
    if "HTR" not in df.columns and "HTHG" in df.columns and "HTAG" in df.columns:
        df["HTR"] = np.where(df["HTHG"]>df["HTAG"], "H", np.where(df["HTHG"]<df["HTAG"], "A", "D"))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"]).sort_values("Date")
    keep = list(dict.fromkeys(REQUIRED_BASE + OPTIONAL_BASE + [c for c in EXTRA_COLS if c in df.columns] + ["__LeagueKey__"]))
    return df[keep]

FULL = load_all_with_league(LEAGUE_FILES)

# ===== Costruzione contesti per lega (con pesi temporali) =====
def build_league_context(df_league: pd.DataFrame):
    last_date = df_league["Date"].max()
    months_ago = df_league["Date"].apply(lambda d: months_between(last_date, d))
    df_league = df_league.copy()
    df_league["__w__"] = np.power(0.5, months_ago / HALF_LIFE_MONTHS)

    teams = pd.unique(pd.concat([df_league["HomeTeam"], df_league["AwayTeam"]]))

    avg_home = wmean(df_league["FTHG"], df_league["__w__"])
    avg_away = wmean(df_league["FTAG"], df_league["__w__"])
    home_adv = max(0.0, (avg_home if isfinite(avg_home) else 1.3) - (avg_away if isfinite(avg_away) else 1.2))

    # per team: goal fatti/subiti (weighted)
    g_home  = df_league.groupby("HomeTeam")[["FTHG","__w__"]].apply(lambda g: wmean(g["FTHG"], g["__w__"]))
    ga_home = df_league.groupby("HomeTeam")[["FTAG","__w__"]].apply(lambda g: wmean(g["FTAG"], g["__w__"]))
    g_away  = df_league.groupby("AwayTeam")[["FTAG","__w__"]].apply(lambda g: wmean(g["FTAG"], g["__w__"]))
    ga_away = df_league.groupby("AwayTeam")[["FTHG","__w__"]].apply(lambda g: wmean(g["FTHG"], g["__w__"]))

    attack, defense = {}, {}
    for t in teams:
        att_home = (g_home.get(t, avg_home) / avg_home) if isfinite(avg_home) and avg_home>0 else 1.0
        att_away = (g_away.get(t, avg_away) / avg_away) if isfinite(avg_away) and avg_away>0 else 1.0
        attack[t] = 0.55*att_home + 0.45*att_away

        def_home = (ga_home.get(t, avg_away) / avg_away) if isfinite(avg_away) and avg_away>0 else 1.0
        def_away = (ga_away.get(t, avg_home) / avg_home) if isfinite(avg_home) and avg_home>0 else 1.0
        defense[t] = 0.55*def_home + 0.45*def_away

    def league_mean(colH, colA):
        mH = wmean(df_league[colH], df_league["__w__"]) if colH in df_league.columns else np.nan
        mA = wmean(df_league[colA], df_league["__w__"]) if colA in df_league.columns else np.nan
        return np.nanmean([mH, mA])

    league_means = {
        "shots_for":     league_mean("HS","AS"),
        "sot_for":       league_mean("HST","AST"),
        "corners_for":   league_mean("HC","AC"),
        "shots_conc":    league_mean("AS","HS"),
        "sot_conc":      league_mean("AST","HST"),
        "corners_conc":  league_mean("AC","HC"),
    }
    for k,v in league_means.items():
        if v!=v or v is None or not isfinite(v): league_means[k] = 1.0

    profiles = {}
    corner_for, corner_conc, y_for, y_conc = {}, {}, {}, {}

    by_home = df_league.groupby("HomeTeam")
    by_away = df_league.groupby("AwayTeam")

    def team_weighted_mean(grouped, team, col):
        if col not in df_league.columns: return np.nan
        try: g = grouped.get_group(team)
        except KeyError: return np.nan
        return wmean(g[col], g["__w__"])

    for t in teams:
        gf_home = team_weighted_mean(by_home, t, "FTHG")
        gf_away = team_weighted_mean(by_away, t, "FTAG")
        ga_home = team_weighted_mean(by_home, t, "FTAG")
        ga_away = team_weighted_mean(by_away, t, "FTHG")

        gf_per = np.nanmean([gf_home, gf_away])
        ga_per = np.nanmean([ga_home, ga_away])

        shots_for = np.nanmean([team_weighted_mean(by_home, t, "HS"),  team_weighted_mean(by_away, t, "AS") ])
        sot_for   = np.nanmean([team_weighted_mean(by_home, t, "HST"), team_weighted_mean(by_away, t, "AST")])
        corners_f = np.nanmean([team_weighted_mean(by_home, t, "HC"),  team_weighted_mean(by_away, t, "AC") ])

        shots_con = np.nanmean([team_weighted_mean(by_home, t, "AS"),  team_weighted_mean(by_away, t, "HS") ])
        sot_con   = np.nanmean([team_weighted_mean(by_home, t, "AST"), team_weighted_mean(by_away, t, "HST")])
        corners_c = np.nanmean([team_weighted_mean(by_home, t, "AC"),  team_weighted_mean(by_away, t, "HC") ])

        if not isfinite(shots_for):   shots_for   = league_means["shots_for"]
        if not isfinite(sot_for):     sot_for     = league_means["sot_for"]
        if not isfinite(corners_f):   corners_f   = league_means["corners_for"]
        if not isfinite(shots_con):   shots_con   = league_means["shots_conc"]
        if not isfinite(sot_con):     sot_con     = league_means["sot_conc"]
        if not isfinite(corners_c):   corners_c   = league_means["corners_conc"]

        if "HTHG" in df_league.columns and "HTAG" in df_league.columns:
            hthg = team_weighted_mean(by_home, t, "HTHG")
            htag = team_weighted_mean(by_away, t, "HTAG")
            gf_ht = np.nansum([hthg, htag]); gf_tot = np.nansum([gf_home, gf_away])
            share_ht = (gf_ht / gf_tot) if (isfinite(gf_tot) and gf_tot>0) else 0.45
            share_ht = clamp(share_ht, HT_SHARE_MIN, HT_SHARE_MAX)
        else:
            share_ht = 0.45

        y_f = np.nanmean([team_weighted_mean(by_home, t, "HY"), team_weighted_mean(by_away, t, "AY")])
        y_c = np.nanmean([team_weighted_mean(by_home, t, "AY"), team_weighted_mean(by_away, t, "HY")])
        y_for[t]  = y_f if isfinite(y_f) else 2.0
        y_conc[t] = y_c if isfinite(y_c) else 2.0

        corner_for[t] = corners_f
        corner_conc[t]= corners_c

        profiles[t] = {
            "gf_per": gf_per, "ga_per": ga_per,
            "shots_for": shots_for, "sot_for": sot_for, "corners_for": corners_f,
            "shots_conc": shots_con, "sot_conc": sot_con, "corners_conc": corners_c,
            "ht_share": share_ht,
        }

    context = {
        "teams": set(teams),
        "avg_home": avg_home, "avg_away": avg_away, "home_adv": home_adv,
        "attack": attack, "defense": defense,
        "profiles": profiles, "league_means": league_means,
        "corner_for": corner_for, "corner_conc": corner_conc,
        "y_for": y_for, "y_conc": y_conc,
        "df": df_league
    }
    return context

LEAGUE_CONTEXTS = {}
for league_key, dfL in FULL.groupby("__LeagueKey__"):
    try:
        LEAGUE_CONTEXTS[league_key] = build_league_context(dfL)
    except Exception as e:
        print(f"[WARN] Contesto non costruito per lega {league_key}: {e}")

# ===== Helpers per nomi mercati e alias =====
MARKET_MAP = {
    "1 (Casa)": "Esito finale: 1", "X (Pareggio)": "Esito finale: X", "2 (Ospite)": "Esito finale: 2",
    "U 1.5": "Under 1.5", "O 1.5": "Over 1.5", "U 2.5": "Under 2.5", "O 2.5": "Over 2.5",
    "U 3.5": "Under 3.5", "O 3.5": "Over 3.5",
    "BTTS (Goal)": "Goal (GG)", "No BTTS": "No Goal (NG)",
    "DC 1X": "Doppia Chance: 1X", "DC 12": "Doppia Chance: 12", "DC X2": "Doppia Chance: X2",
    "AH Casa -0.5": "Handicap Asiatico: Casa -0.5 (≃ 1 secco)", "AH Ospite +0.5": "Doppia Chance: X2 (≃ AH Ospite +0.5)",
    "DNB Casa": "Draw No Bet: Casa (0)", "DNB Ospite": "Draw No Bet: Ospite (0)",
    "Multigol 0-2": "Multigol 0–2", "Multigol 1-3": "Multigol 1–3", "Multigol 2-4": "Multigol 2–4",
    "HT O 0.5": "1° Tempo Over 0.5", "HT U 1.5": "1° Tempo Under 1.5", "HT O 1.5": "1° Tempo Over 1.5",
    "HT/FT": "Parziale/Finale",
    "Corners O 8.5": "Angoli Over 8.5", "Corners O 9.5": "Angoli Over 9.5", "Corners U 10.5": "Angoli Under 10.5",
    "Cards O 3.5": "Cartellini (gialli) Over 3.5", "Cards O 4.5": "Cartellini (gialli) Over 4.5",
    "1X + U3.5": "Doppia Chance 1X + Under 3.5", "X2 + U3.5": "Doppia Chance X2 + Under 3.5",
    "12 + O1.5": "12 + Over 1.5", "BTTS + O2.5": "Goal (GG) + Over 2.5", "NoBTTS + U2.5": "No Goal (NG) + Under 2.5",
}
def _map_htft(label: str) -> str:
    if label.startswith("HT/FT "):
        return f"Parziale/Finale {label.replace('HT/FT ','')}"
    return None
def _map_correct_score(label: str) -> str:
    if label.startswith("Ris.Esatto "):
        return f"Risultato esatto {label.replace('Ris.Esatto ','')}"
    return None
def to_planetwin_name(name: str) -> str:
    if name in MARKET_MAP: return MARKET_MAP[name]
    alt = _map_htft(name)
    if alt: return alt
    alt = _map_correct_score(name)
    if alt: return alt
    return name

def market_family(n: str) -> str:
    if n.startswith("Esito finale: 1"): return "HomeWin"
    if n.startswith("Esito finale: 2"): return "AwayWin"
    if n.startswith("Esito finale: X"): return "Draw"
    if "Doppia Chance: X2" in n or "DC X2" in n: return "AwayNotLose"
    if "Doppia Chance: 1X" in n or "DC 1X" in n: return "HomeNotLose"
    if "Doppia Chance: 12" in n or "DC 12" in n: return "NoDraw"
    if "AH Casa -0.5" in n or "DNB Casa" in n: return "HomeWin"
    if "DNB Ospite" in n or "AH Ospite +0.5" in n: return "AwayNotLose"
    if n.startswith("Under"): return "Under"
    if n.startswith("Over"): return "Over"
    if n.startswith("Multigol"): return "Multigol"
    if n.startswith("Goal (GG)") or n.startswith("No Goal"): return "BTTS"
    if n.startswith("Parziale/Finale"): return "HTFT"
    if n.startswith("1° Tempo"): return "HalfTotals"
    if n.startswith("Angoli"): return "Corners"
    if n.startswith("Cartellini"): return "Cards"
    if n.startswith("Risultato esatto"): return "CorrectScore"
    if " + " in n: return "Combo"
    return "Other"

def canon(s: str) -> str:
    s = s.lower()
    s = s.replace("°", "")
    s = s.replace("primo tempo", "1t").replace("1 tempo", "1t").replace("1° tempo", "1t").replace("ht ", "1t ")
    s = s.replace("over", "o ").replace("under", "u ")
    s = s.replace("goal (gg)", "gg").replace("goal", "gg").replace("btts", "gg").replace("no goal", "ng").replace("nobtts","ng")
    s = s.replace("doppia chance", "dc").replace("parziale/finale", "htft").replace("parziale / finale","htft")
    s = s.replace("angoli", "corners").replace("cartellini (gialli)", "cards").replace("cartellini", "cards")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = s.replace("×", "x")
    return s

def market_aliases(pretty: str):
    cands = set()
    p = pretty; c = canon(p)
    cands.add(c)
    if p.startswith("Esito finale: "):
        v = p.split(":")[1].strip()
        if v == "1": cands.update({"1","esitofinale1","1secco"})
        elif v == "X": cands.update({"x","pareggio","esitofinalex"})
        elif v == "2": cands.update({"2","esitofinale2"})
    if p.startswith("Doppia Chance: "):
        v = p.split(":")[1].strip()
        key = canon(v)
        cands.update({key, "dc"+key, "dc "+key, "dc"+key.replace(" ", "")})
    if "AH Ospite +0.5" in p: cands.update({"x2","dcx2","ahaway+0.5","ahospite+0.5"})
    if "AH Casa -0.5" in p:  cands.update({"1","ahhome-0.5","ahcasa-0.5"})
    if "Draw No Bet: Casa" in p:   cands.update({"dnbcasa","dnbhome","dnb1"})
    if "Draw No Bet: Ospite" in p: cands.update({"dnbospite","dnbaway","dnb2"})
    m = re.match(r"^(Under|Over) (\d+\.\d)$", p)
    if m:
        side = "u" if m.group(1)=="Under" else "o"
        num = m.group(2).replace(".","")
        cands.update({f"{side}{num}", f"{side} {num}"})
    m = re.match(r"^Multigol (\d)–(\d)$", p)
    if m:
        a,b = m.group(1), m.group(2)
        cands.update({f"mg{a}-{b}", f"multigol{a}-{b}", f"{a}-{b}gol"})
    if p.startswith("1° Tempo Over "):
        num = p.split("Over ")[1].replace(".","").strip()
        cands.update({f"hto{num}", f"1to{num}"})
    if p.startswith("1° Tempo Under "):
        num = p.split("Under ")[1].replace(".","").strip()
        cands.update({f"htu{num}", f"1tu{num}"})
    if p.startswith("Goal (GG)"): cands.update({"gg","btts"})
    if p.startswith("No Goal"):   cands.update({"ng","nobtts"})
    if p.startswith("Parziale/Finale "):
        pair = p.split("Parziale/Finale ")[1].strip()
        cands.update({f"htft{pair}", f"htft{pair.replace('/','')}"})
    m = re.match(r"^Angoli (Over|Under) (\d+\.\d)$", p)
    if m:
        side = "o" if m.group(1)=="Over" else "u"
        num = m.group(2).replace(".","")
        cands.add(f"corners{side}{num}")
    m = re.match(r"^Cartellini.* (Over) (\d+\.\d)$", p)
    if m:
        num = m.group(2).replace(".","")
        cands.add(f"cardso{num}")
    if " + " in p:
        left, right = p.split(" + ", 1)
        cands.add(canon(left) + "+" + canon(right))
    return {canon(x) for x in cands}

# ===== Scelta lega e predizione =====
def pick_league_for_match(home: str, away: str):
    hl = home.lower().strip(); al = away.lower().strip()
    candidates = []
    for key, ctx in LEAGUE_CONTEXTS.items():
        tset = {t.lower() for t in ctx["teams"]}
        if hl in tset and al in tset:
            max_date = ctx["df"]["Date"].max()
            candidates.append((max_date, key))
    if not candidates:
        raise ValueError(f"Nessuna lega contiene entrambe le squadre: '{home}' e '{away}'.")
    candidates.sort(reverse=True)
    return candidates[0][1]

def predict_for_match(home_input: str, away_input: str):
    league_key = pick_league_for_match(home_input, away_input)
    ctx = LEAGUE_CONTEXTS[league_key]

    def _resolve(team_str):
        for t in ctx["teams"]:
            if t == team_str: return t
        low = team_str.lower()
        for t in ctx["teams"]:
            if t.lower() == low: return t
        raise ValueError(f"Team non trovato nella lega {league_key}: {team_str}")

    home = _resolve(home_input.strip()); away = _resolve(away_input.strip())
    if home == away: raise ValueError("Le due squadre coincidono.")

    avg_home = ctx["avg_home"]; avg_away = ctx["avg_away"]; home_adv = ctx["home_adv"]
    attack = ctx["attack"]; defense = ctx["defense"]
    profiles = ctx["profiles"]; league_means = ctx["league_means"]
    corner_for = ctx["corner_for"]; corner_conc = ctx["corner_conc"]
    y_for = ctx["y_for"]; y_conc = ctx["y_conc"]

    base_home = avg_home * attack.get(home,1.0) * defense.get(away,1.0) + home_adv*0.5
    base_away = avg_away * attack.get(away,1.0) * defense.get(home,1.0)

    ph, pa = profiles[home], profiles[away]
    f_sot = (safe_ratio(ph["sot_for"], league_means["sot_for"])**W_SOT) * (safe_ratio(pa["sot_conc"], league_means["sot_conc"])**W_SOT)
    f_sho = (safe_ratio(ph["shots_for"], league_means["shots_for"])**W_SHOTS) * (safe_ratio(pa["shots_conc"], league_means["shots_conc"])**W_SHOTS)
    f_cor = (safe_ratio(ph["corners_for"], league_means["corners_for"])**W_CORNERS) * (safe_ratio(pa["corners_conc"], league_means["corners_conc"])**W_CORNERS)
    mult_home = f_sot * f_sho * f_cor

    g_sot = (safe_ratio(pa["sot_for"], league_means["sot_for"])**W_SOT) * (safe_ratio(ph["sot_conc"], league_means["sot_conc"])**W_SOT)
    g_sho = (safe_ratio(pa["shots_for"], league_means["shots_for"])**W_SHOTS) * (safe_ratio(ph["shots_conc"], league_means["shots_conc"])**W_SHOTS)
    g_cor = (safe_ratio(pa["corners_for"], league_means["corners_for"])**W_CORNERS) * (safe_ratio(ph["corners_conc"], league_means["corners_conc"])**W_CORNERS)
    mult_away = g_sot * g_sho * g_cor

    lambda_home = max(0.05, base_home * mult_home)
    lambda_away = max(0.05, base_away * mult_away)

    base = np.zeros((MAX_GOALS+1, MAX_GOALS+1))
    for i in range(MAX_GOALS+1):
        for j in range(MAX_GOALS+1):
            base[i,j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
    P = np.zeros_like(base)
    for i in range(MAX_GOALS+1):
        for j in range(MAX_GOALS+1):
            P[i,j] = base[i,j]*dc_corr(i,j,DC_TAU)
    P /= P.sum()

    prob_H = sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if i>j)
    prob_A = sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if j>i)
    prob_D = sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if i==j)

    def total_prob_leq(K): return sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if i+j<=K)
    prob_U15 = total_prob_leq(1); prob_O15 = 1 - prob_U15
    prob_U25 = total_prob_leq(2); prob_O25 = 1 - prob_U25
    prob_U35 = total_prob_leq(3); prob_O35 = 1 - prob_U35
    prob_BTTS = 1 - (sum(P[i,0] for i in range(MAX_GOALS+1)) + sum(P[0,j] for j in range(MAX_GOALS+1)) - P[0,0])

    prob_DC_1X = 1 - prob_A; prob_DC_12 = 1 - prob_D; prob_DC_X2 = 1 - prob_H
    prob_AH_home_m05 = prob_H; prob_AH_away_p05 = 1 - prob_H
    prob_DNB_home_win = prob_H; prob_DNB_away_win = prob_A

    def multigol_prob(low, high): return sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if low <= i+j <= high)
    mg_0_2 = multigol_prob(0,2); mg_1_3 = multigol_prob(1,3); mg_2_4 = multigol_prob(2,4)

    lam_ht_home = max(0.05, lambda_home * profiles[home]["ht_share"])
    lam_ht_away = max(0.05, lambda_away * profiles[away]["ht_share"])
    Pht = np.array([[poisson_pmf(i,lam_ht_home)*poisson_pmf(j,lam_ht_away) for j in range(6)] for i in range(6)])
    Pht /= Pht.sum()
    def ht_total_prob_leq(K): return sum(Pht[i,j] for i in range(6) for j in range(6) if i+j<=K)
    ht_O05 = 1 - ht_total_prob_leq(0); ht_U15 = ht_total_prob_leq(1); ht_O15 = 1 - ht_U15
    pHT_H = sum(Pht[i,j] for i in range(6) for j in range(6) if i>j)
    pHT_D = sum(Pht[i,j] for i in range(6) for j in range(6) if i==j)
    pHT_A = 1 - pHT_H - pHT_D
    HTFT = {
        "1/1": pHT_H * prob_H, "1/X": pHT_H * prob_D, "1/2": pHT_H * prob_A,
        "X/1": pHT_D * prob_H, "X/X": pHT_D * prob_D, "X/2": pHT_D * prob_A,
        "2/1": pHT_A * prob_H, "2/X": pHT_A * prob_D, "2/2": pHT_A * prob_A,
    }

    lch = ((corner_for.get(home,2.5)*corner_conc.get(away,2.5))**0.5) if (isfinite(corner_for.get(home,2.5)) and isfinite(corner_conc.get(away,2.5))) else 2.5
    lca = ((corner_for.get(away,2.5)*corner_conc.get(home,2.5))**0.5) if (isfinite(corner_for.get(away,2.5)) and isfinite(corner_conc.get(home,2.5))) else 2.5
    lam_corners_tot = max(1.0, lch + lca)
    corners_O85 = 1 - poisson_cdf(8, lam_corners_tot)
    corners_O95 = 1 - poisson_cdf(9, lam_corners_tot)
    corners_U105 = poisson_cdf(10, lam_corners_tot)

    lyh = ((y_for.get(home,2.0)*y_conc.get(away,2.0))**0.5) if (isfinite(y_for.get(home,2.0)) and isfinite(y_conc.get(away,2.0))) else 2.0
    lya = ((y_for.get(away,2.0)*y_conc.get(home,2.0))**0.5) if (isfinite(y_for.get(away,2.0)) and isfinite(y_conc.get(home,2.0))) else 2.0
    lam_cards_tot = max(0.5, lyh + lya)
    cards_O35 = 1 - poisson_cdf(3, lam_cards_tot)
    cards_O45 = 1 - poisson_cdf(4, lam_cards_tot)

    def prob_event(filter_fn): return sum(P[i,j] for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1) if filter_fn(i,j))
    p_1X_U35   = prob_event(lambda i,j: (i>=j) and (i+j<=3))
    p_X2_U35   = prob_event(lambda i,j: (j>=i) and (i+j<=3))
    p_12_O15   = prob_event(lambda i,j: (i!=j) and (i+j>=2))
    p_BTTS_O25 = prob_event(lambda i,j: (i>0 and j>0 and i+j>=3))
    p_NoBTTS_U25 = prob_event(lambda i,j: (i==0 or j==0) and (i+j<=2))

    all_scores = [((i,j), P[i,j]) for i in range(MAX_GOALS+1) for j in range(MAX_GOALS+1)]
    top_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)[:3]

    candidates = {
        "1 (Casa)": prob_H, "X (Pareggio)": prob_D, "2 (Ospite)": prob_A,
        "DC 1X": prob_DC_1X, "DC 12": prob_DC_12, "DC X2": prob_DC_X2,
        "AH Casa -0.5": prob_AH_home_m05, "AH Ospite +0.5": prob_AH_away_p05,
        "DNB Casa": prob_DNB_home_win, "DNB Ospite": prob_DNB_away_win,
        "U 1.5": prob_U15, "O 1.5": prob_O15, "U 2.5": prob_U25, "O 2.5": prob_O25, "U 3.5": prob_U35, "O 3.5": prob_O35,
        "Multigol 0-2": mg_0_2, "Multigol 1-3": mg_1_3, "Multigol 2-4": mg_2_4,
        "BTTS (Goal)": prob_BTTS, "No BTTS": 1 - prob_BTTS,
        "HT O 0.5": ht_O05, "HT U 1.5": ht_U15, "HT O 1.5": ht_O15,
        "Corners O 8.5": corners_O85, "Corners O 9.5": corners_O95, "Corners U 10.5": corners_U105,
        "Cards O 3.5": cards_O35, "Cards O 4.5": cards_O45,
        "1X + U3.5": p_1X_U35, "X2 + U3.5": p_X2_U35, "12 + O1.5": p_12_O15, "BTTS + O2.5": p_BTTS_O25, "NoBTTS + U2.5": p_NoBTTS_U25,
    }
    for k,v in sorted(HTFT.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        candidates[f"HT/FT {k}"] = v
    for (i,j),p in top_scores:
        candidates[f"Ris.Esatto {i}-{j}"] = p

    items = sorted(((to_planetwin_name(n), p) for n,p in candidates.items()), key=lambda kv: kv[1], reverse=True)

    def pick_from_range(lo, hi, already_used_names, max_k=TOP_PER_SECTION):
        picked = []; used_families = set()
        for nice, p in items:
            if not (lo <= p < hi): continue
            if nice in already_used_names: continue
            fam = market_family(nice)
            if fam in used_families: continue
            picked.append((nice, p))
            used_families.add(fam)
            already_used_names.add(nice)
            if len(picked) == max_k: break
        return picked

    already = set()
    buckets = {
        "Sicurissimi":             pick_from_range(0.80, 1.01, already, TOP_PER_SECTION),
        "Quasi sicuri":            pick_from_range(0.75, 0.80, already, TOP_PER_SECTION),
        "Rischiosi ma possibili":  pick_from_range(0.65, 0.75, already, TOP_PER_SECTION),
        "Medio-alte (50–70%)":     pick_from_range(0.50, 0.70, already, TOP_PER_SECTION),
        "Rischiosissime":          pick_from_range(0.30, 0.40, already, TOP_PER_SECTION),
        "Quasi impossibile":       pick_from_range(0.15, 0.30, already, TOP_PER_SECTION),
        "Impossibile":             pick_from_range(0.00, 0.15, already, TOP_PER_SECTION),
    }
    for key in buckets:
        buckets[key] = [(f"{badge(p)} {nice}".strip(), p) for (nice, p) in buckets[key]]

    catalog = {nice: p for nice, p in items}
    alias_map = {}
    for nice, p in catalog.items():
        for a in market_aliases(nice):
            alias_map[a] = (nice, p)

    return {
        "league": league_key,
        "match": f"{home} vs {away}",
        "home": home, "away": away,
        "lambda_home": lambda_home, "lambda_away": lambda_away,
        "buckets": buckets,
        "catalog": catalog,
        "alias_map": alias_map
    }

# ===== GUI =====
GROUPS = [
    "Sicurissimi",
    "Quasi sicuri",
    "Rischiosi ma possibili",
    "Medio-alte (50–70%)",
    "Rischiosissime",
    "Quasi impossibile",
    "Impossibile",
]

class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Predictor — Poisson + Feature Boost (leghe separate, pesi temporali)")
        self.summaries = []  # risultati ultimi calcoli

        # Input panel
        frm_in = ttk.Frame(root, padding=10)
        frm_in.pack(fill="x")

        ttk.Label(frm_in, text="Inserisci i match (uno per riga, formato: Casa - Ospite):").pack(anchor="w")
        self.txt_matches = tk.Text(frm_in, height=5)
        self.txt_matches.pack(fill="x", pady=5)
        self.txt_matches.insert("1.0", "Juventus - Inter\nReal Madrid - Barcelona")

        self.btn_calc = ttk.Button(frm_in, text="Calcola pronostici", command=self.on_calculate)
        self.btn_calc.pack(anchor="e", pady=(0,5))

        # Notebook per gruppi
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=5)
        self.tree_by_group = {}
        for g in GROUPS:
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text=g)
            tree = ttk.Treeview(tab, columns=("match","market","prob"), show="headings", height=12)
            tree.heading("match", text="Match [Lega]")
            tree.heading("market", text="Pronostico")
            tree.heading("prob", text="Prob%")
            tree.column("match", width=260)
            tree.column("market", width=280)
            tree.column("prob", width=80, anchor="e")
            tree.pack(fill="both", expand=True)
            self.tree_by_group[g] = tree

        # Query panel
        frm_q = ttk.Frame(root, padding=10)
        frm_q.pack(fill="x")
        ttk.Label(frm_q, text="Controlla percentuale di un pronostico specifico").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(frm_q, text="Match:").grid(row=1, column=0, sticky="e")
        self.cmb_match = ttk.Combobox(frm_q, state="readonly", width=40)
        self.cmb_match.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(frm_q, text="Pronostico:").grid(row=1, column=2, sticky="e")
        self.ent_market = ttk.Entry(frm_q, width=35)
        self.ent_market.grid(row=1, column=3, sticky="w", padx=5)
        ttk.Button(frm_q, text="Mostra %", command=self.on_query_market).grid(row=1, column=4, sticky="w", padx=8)

        for i in range(5):
            frm_q.grid_columnconfigure(i, weight=1)

    def on_calculate(self):
        raw = self.txt_matches.get("1.0", "end").strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            messagebox.showwarning("Input vuoto", "Inserisci almeno un match.")
            return

        pairs = []
        for ln in lines:
            sep = None
            for cand in [" - ", "-", "–", "—"]:
                if cand in ln:
                    sep = cand; break
            if not sep:
                messagebox.showerror("Formato non valido", f"Riga: '{ln}' — usa 'Casa - Ospite'")
                return
            a,b = [x.strip() for x in ln.split(sep,1)]
            if not a or not b:
                messagebox.showerror("Formato non valido", f"Riga: '{ln}' — manca una squadra")
                return
            pairs.append((a,b))

        self.summaries = []
        errors = []
        for a,b in pairs:
            try:
                res = predict_for_match(a,b)
                self.summaries.append(res)
            except Exception as e:
                errors.append(f"{a} vs {b}: {e}")

        # pulisci tabelle
        for g, tree in self.tree_by_group.items():
            for item in tree.get_children():
                tree.delete(item)

        if not self.summaries:
            if errors:
                messagebox.showerror("Errore", "\n".join(errors))
            else:
                messagebox.showwarning("Nessun risultato", "Nessun match valido.")
            return

        # riempi tab per gruppi
        for s in self.summaries:
            match_disp = f"{s['match']} [{s['league']}]"
            for g in GROUPS:
                for (nice, p) in s["buckets"][g]:
                    prob = pct(p)
                    self.tree_by_group[g].insert("", "end", values=(match_disp, nice, prob))

        # riempi combobox match per query ad-hoc
        self.cmb_match["values"] = [f"{s['match']} [{s['league']}]" for s in self.summaries]
        if self.cmb_match["values"]:
            self.cmb_match.current(0)

        if errors:
            messagebox.showwarning("Alcuni match saltati", "\n".join(errors))

    def on_query_market(self):
        if not self.summaries:
            messagebox.showwarning("Nessun calcolo", "Calcola prima qualche match.")
            return
        sel = self.cmb_match.get().strip()
        if not sel:
            messagebox.showwarning("Seleziona match", "Scegli il match dal menu.")
            return
        # trova summary scelto
        chosen = None
        for s in self.summaries:
            if sel.startswith(s["match"]):
                chosen = s; break
        if not chosen:
            messagebox.showerror("Errore", "Match non trovato nella memoria.")
            return

        q = self.ent_market.get().strip()
        if not q:
            messagebox.showinfo("Suggerimento", "Scrivi un pronostico (es. 'x2', 'o2.5', 'gg', 'ht o0.5', '1x + u3.5').")
            return

        key = canon(q)
        alias_map = chosen["alias_map"]
        hit = alias_map.get(key) or alias_map.get(key.replace(" ", "")) \
              or alias_map.get(key.replace("over","o").replace("under","u"))
        if hit:
            nice, p = hit
            messagebox.showinfo("Probabilità", f"{chosen['match']} [{chosen['league']}]\n\n{nice}: {pct(p)}")
        else:
            # mostra esempi top come guida
            examples = sorted(chosen["catalog"].items(), key=lambda kv: kv[1], reverse=True)[:8]
            msg = "Pronostico non riconosciuto.\nEsempi validi per questo match:\n\n" + \
                  "\n".join([f"- {nm}: {pct(pr)}" for nm,pr in examples])
            messagebox.showwarning("Non trovato", msg)

def main_gui():
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Avvia direttamente la GUI
    main_gui()
