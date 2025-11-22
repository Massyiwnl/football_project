import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from math import exp, factorial
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ================= CONFIGURAZIONE PROFESSIONALE =================
# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Percorsi e parametri del modello
CONFIG = {
    "LEAGUE_FILES": [
        # Inserisci qui tutti i tuoi percorsi come nel tuo file originale
        "data/serie-a/season-2324.csv", "data/serie-a/season-2425.csv",
        "data/serie-a/season-2223.csv", "data/serie-a/season-2122.csv",
        "data/laliga/SP1.csv", "data/laliga/SP2.csv",
        "data/premier/E0.csv", "data/premier/E1.csv",
        # ... aggiungi gli altri file qui
    ],
    "DECAY_HALFLIFE_DAYS": 365.0,  # Peso dimezza ogni anno (più stabile di 24 mesi)
    "MIN_GAMES_PLAYED": 5,         # Minimo partite per avere stats affidabili
    "MAX_GOALS": 10,
    
    # Pesi del "Feature Boosting" (Quanto contano tiri/corner rispetto ai gol puri)
    "WEIGHTS": {
        "SOT": 0.25,    # Shots on Target (molto predittivo)
        "SHOTS": 0.10,  # Total Shots
        "CORNERS": 0.05 # Corners (meno correlati ai gol, peso ridotto)
    },
    
    # Parametro Dixon-Coles (Rho) - Correlazione bassi punteggi
    "RHO": -0.13, 
}

# Mappatura colonne standard per compatibilità diversi dataset (es. Football-Data.co.uk)
COL_MAP = {
    "HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam", "Date": "Date",
    "FTHG": "FTHG", "FTAG": "FTAG",
    "HST": "HST", "AST": "AST",     # Tiri in porta
    "HS": "HS", "AS": "AS",         # Tiri totali
    "HC": "HC", "AC": "AC"          # Corner
}

# ================= LOGICA MATEMATICA & STATISTICA =================

class MathUtils:
    @staticmethod
    def poisson_pmf(mu: float, k: int) -> float:
        """Probability Mass Function per Poisson."""
        return (mu**k * exp(-mu)) / factorial(k)

    @staticmethod
    def dixon_coles_matrix(lambda_h: float, lambda_a: float, rho: float, max_goals: int) -> np.ndarray:
        """
        Genera la matrice di probabilità con correzione Dixon-Coles.
        Questa correzione è fondamentale per gestire la sottostima dei pareggi (0-0, 1-1)
        nei modelli Poisson puri.
        """
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        # Matrice Poisson base (indipendente)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_h = MathUtils.poisson_pmf(lambda_h, i)
                p_a = MathUtils.poisson_pmf(lambda_a, j)
                matrix[i, j] = p_h * p_a

        # Applicazione fattore di correzione Dixon-Coles (Tau function)
        # Modifica probabilità solo per 0-0, 0-1, 1-0, 1-1
        if lambda_h > 0 and lambda_a > 0:
            # 0-0
            matrix[0, 0] *= (1 - (lambda_h * lambda_a * rho))
            # 0-1
            matrix[0, 1] *= (1 + (lambda_h * rho))
            # 1-0
            matrix[1, 0] *= (1 + (lambda_a * rho))
            # 1-1
            matrix[1, 1] *= (1 - rho)

        # Normalizzazione (la somma deve fare 1.0)
        matrix /= matrix.sum()
        return matrix

class DataManager:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = None
        self.teams_stats = {}
        self.league_avgs = {}

    def load_data(self):
        """Carica e pulisce i dati con gestione errori robusta."""
        frames = []
        required_cols = list(COL_MAP.values())
        
        for f in self.file_paths:
            try:
                # Tenta diversi encoding e separatori se necessario
                df = pd.read_csv(f, on_bad_lines='skip')
                
                # Normalizza nomi colonne
                if 'Home' in df.columns and 'HomeTeam' not in df.columns:
                    df.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'}, inplace=True)
                
                # Verifica colonne minime
                if not all(col in df.columns for col in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]):
                    continue

                # Conversione data intelligente
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
                
                # Aggiungi colonna lega inferita dal path
                df['League'] = f.split('/')[-2] if '/' in f else 'Unknown'
                frames.append(df)
                
            except Exception as e:
                logger.warning(f"Impossibile leggere {f}: {e}")

        if not frames:
            raise ValueError("Nessun dataset valido caricato.")

        self.data = pd.concat(frames, ignore_index=True).sort_values('Date')
        logger.info(f"Caricate {len(self.data)} partite totali.")
        self._calculate_weighted_stats()

    def _calculate_weighted_stats(self):
        """
        Calcola la forza offensiva/difensiva usando Media Ponderata Esponenziale.
        Più la partita è recente, più pesa.
        """
        ref_date = self.data['Date'].max()
        # Calcolo pesi temporali
        time_diff = (ref_date - self.data['Date']).dt.days
        self.data['Weight'] = np.exp(-time_diff * (np.log(2) / CONFIG['DECAY_HALFLIFE_DAYS']))

        # Medie della lega (Globali) per normalizzazione
        self.global_avg_home_goals = np.average(self.data['FTHG'], weights=self.data['Weight'])
        self.global_avg_away_goals = np.average(self.data['FTAG'], weights=self.data['Weight'])

        # Statistiche per Team
        teams = set(self.data['HomeTeam']).union(set(self.data['AwayTeam']))
        
        # Raggruppa per calcoli veloci
        home_stats = self.data.groupby('HomeTeam').apply(
            lambda x: pd.Series({
                'GF': np.average(x['FTHG'], weights=x['Weight']), # Gol Fatti Casa
                'GA': np.average(x['FTAG'], weights=x['Weight']), # Gol Subiti Casa
                'SOT_F': np.average(x['HST'], weights=x['Weight']) if 'HST' in x else np.nan,
                'SOT_A': np.average(x['AST'], weights=x['Weight']) if 'AST' in x else np.nan,
                'CORN_F': np.average(x['HC'], weights=x['Weight']) if 'HC' in x else np.nan,
                'Games': len(x)
            })
        )
        
        away_stats = self.data.groupby('AwayTeam').apply(
            lambda x: pd.Series({
                'GF': np.average(x['FTAG'], weights=x['Weight']), # Gol Fatti Fuori
                'GA': np.average(x['FTHG'], weights=x['Weight']), # Gol Subiti Fuori
                'SOT_F': np.average(x['AST'], weights=x['Weight']) if 'AST' in x else np.nan,
                'SOT_A': np.average(x['HST'], weights=x['Weight']) if 'HST' in x else np.nan,
                'CORN_F': np.average(x['AC'], weights=x['Weight']) if 'AC' in x else np.nan,
                'Games': len(x)
            })
        )

        # Uniamo in un dizionario facile da consultare
        self.teams_stats = {}
        for team in teams:
            h_s = home_stats.loc[team] if team in home_stats.index else None
            a_s = away_stats.loc[team] if team in away_stats.index else None
            
            if h_s is not None and a_s is not None:
                self.teams_stats[team] = {
                    'Home': h_s,
                    'Away': a_s
                }

    def get_team_stats(self, team_name):
        return self.teams_stats.get(team_name)

class MatchPredictor:
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def predict(self, home_team, away_team):
        # Fuzzy matching o risoluzione nomi
        h_stats = self.dm.get_team_stats(home_team)
        a_stats = self.dm.get_team_stats(away_team)

        if not h_stats or not a_stats:
            return None, "Dati insufficienti per una o entrambe le squadre."

        # 1. Calcolo Forza Attacco e Difesa (Rispetto alla media lega)
        # Attacco Casa vs Difesa Ospite
        att_h = h_stats['Home']['GF'] / self.dm.global_avg_home_goals
        def_a = a_stats['Away']['GA'] / self.dm.global_avg_home_goals # La difesa away subisce gol "home"
        
        # Attacco Ospite vs Difesa Casa
        att_a = a_stats['Away']['GF'] / self.dm.global_avg_away_goals
        def_h = h_stats['Home']['GA'] / self.dm.global_avg_away_goals

        # Lambda base (Gol attesi)
        lambda_h = self.dm.global_avg_home_goals * att_h * def_a
        lambda_a = self.dm.global_avg_away_goals * att_a * def_h

        # 2. Feature Boosting (Correzione basata su statistiche secondarie)
        # Se abbiamo dati su tiri/corner, raffinamo lambda
        # Logica: Se una squadra fa molti tiri in porta ma segna poco, è sfortunata -> aumentiamo lambda
        
        boost_h = 1.0
        boost_a = 1.0
        
        # Esempio implementazione correttore SOT (Tiri in porta)
        if not pd.isna(h_stats['Home']['SOT_F']):
             # Rapporto SOT fatti in casa / Media Gol fatti in casa
             eff_h = h_stats['Home']['GF'] / (h_stats['Home']['SOT_F'] + 0.1) 
             # Se efficienza bassa, potrebbe esserci regressione positiva, ma qui usiamo la pura creazione
             # Semplifichiamo: Moltiplicatore basato sul volume di gioco
             volume_h = h_stats['Home']['SOT_F'] * a_stats['Away']['SOT_A'] # Tiri fatti casa * Tiri subiti ospite
             # Normalizziamo approssimativamente intorno a 1.0 (valori medi ~4-5 tiri)
             boost_h *= (volume_h / 20.0) ** CONFIG['WEIGHTS']['SOT']

        if not pd.isna(a_stats['Away']['SOT_F']):
             volume_a = a_stats['Away']['SOT_F'] * h_stats['Home']['SOT_A']
             boost_a *= (volume_a / 20.0) ** CONFIG['WEIGHTS']['SOT']

        # Applica boost (con limiti per evitare valori assurdi)
        lambda_h = max(0.1, lambda_h * np.clip(boost_h, 0.8, 1.2))
        lambda_a = max(0.1, lambda_a * np.clip(boost_a, 0.8, 1.2))

        # 3. Calcolo Matrice Probabilità
        matrix = MathUtils.dixon_coles_matrix(lambda_h, lambda_a, CONFIG['RHO'], CONFIG['MAX_GOALS'])
        
        # 4. Estrazione Quote/Probabilità
        probs = self._extract_markets(matrix)
        probs['lambdas'] = (lambda_h, lambda_a)
        return probs, None

    def _extract_markets(self, M):
        """Estrae probabilità leggibili dalla matrice."""
        # Esiti finali
        p_1 = np.sum(np.tril(M, -1)) # Home (triangolo inferiore)
        p_X = np.trace(M)            # Draw (diagonale)
        p_2 = np.sum(np.triu(M, 1))  # Away (triangolo superiore)

        # Gol totali
        p_o15 = 1 - np.sum([M[i,j] for i in range(len(M)) for j in range(len(M)) if i+j <= 1])
        p_o25 = 1 - np.sum([M[i,j] for i in range(len(M)) for j in range(len(M)) if i+j <= 2])
        p_u25 = 1 - p_o25
        
        # Goal / NoGoal
        p_ng = sum(M[i, 0] for i in range(len(M))) + sum(M[0, j] for j in range(len(M))) - M[0,0]
        p_gg = 1 - p_ng

        return {
            "1": p_1, "X": p_X, "2": p_2,
            "1X": p_1 + p_X, "X2": p_X + p_2, "12": p_1 + p_2,
            "Over 1.5": p_o15, "Over 2.5": p_o25, "Under 2.5": p_u25,
            "GG": p_gg, "NG": p_ng
        }

# ================= GUI MODERNA =================

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pro Football Predictor v2.0")
        self.root.geometry("900x600")
        
        # Stile
        style = ttk.Style()
        style.theme_use('clam')
        
        # Data Loading
        self.dm = DataManager(CONFIG['LEAGUE_FILES'])
        try:
            self.dm.load_data()
        except Exception as e:
            messagebox.showerror("Errore Dati", str(e))
        
        self.predictor = MatchPredictor(self.dm)
        
        self._setup_ui()

    def _setup_ui(self):
        # Frame Input
        frame_top = ttk.Frame(self.root, padding=20)
        frame_top.pack(fill="x")
        
        ttk.Label(frame_top, text="Inserisci Match (Casa - Ospite):", font=("Arial", 12, "bold")).pack(anchor="w")
        self.txt_input = tk.Text(frame_top, height=4, font=("Consolas", 10))
        self.txt_input.pack(fill="x", pady=5)
        self.txt_input.insert("1.0", "AC Milan - Inter\nJuventus - Napoli")
        
        btn_calc = ttk.Button(frame_top, text="ANALIZZA MATCH", command=self.on_calculate)
        btn_calc.pack(pady=5, anchor="e")
        
        # Frame Risultati
        self.tree = ttk.Treeview(self.root, columns=("Match", "Pick", "Conf", "1", "X", "2", "O2.5", "GG"), show="headings")
        
        cols = {
            "Match": 200, "Pick": 150, "Conf": 80,
            "1": 60, "X": 60, "2": 60, "O2.5": 60, "GG": 60
        }
        for c, w in cols.items():
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
            
        self.tree.pack(fill="both", expand=True, padx=20, pady=10)

    def on_calculate(self):
        # Pulisci tabella
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        raw_text = self.txt_input.get("1.0", "end").strip()
        lines = [l for l in raw_text.split('\n') if l.strip()]
        
        for line in lines:
            if '-' not in line: continue
            home, away = line.split('-')
            home, away = home.strip(), away.strip()
            
            probs, error = self.predictor.predict(home, away)
            
            if error:
                self.tree.insert("", "end", values=(f"{home}-{away}", "ERR", error, "-", "-", "-", "-", "-"))
                continue
                
            # Trova la "Value Bet" (Probabilità più alta)
            markets = {
                "1": probs["1"], "X": probs["X"], "2": probs["2"],
                "1X": probs["1X"], "X2": probs["X2"],
                "O 2.5": probs["Over 2.5"], "U 2.5": probs["Under 2.5"],
                "GG": probs["GG"]
            }
            
            # Filtra quote "sicure"
            best_pick = max(markets, key=markets.get)
            best_prob = markets[best_pick]
            
            # Formattazione percentuali
            def fp(x): return f"{x*100:.1f}%"
            
            # Badge sicurezza
            stars = ""
            if best_prob > 0.75: stars = "⭐⭐⭐"
            elif best_prob > 0.65: stars = "⭐⭐"
            elif best_prob > 0.55: stars = "⭐"
            
            self.tree.insert("", "end", values=(
                f"{home} - {away}",
                f"{best_pick} {stars}",
                fp(best_prob),
                fp(probs["1"]), fp(probs["X"]), fp(probs["2"]),
                fp(probs["Over 2.5"]), fp(probs["GG"])
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()