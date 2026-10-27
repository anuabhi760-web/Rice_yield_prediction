"""
chatbot.py — Rice Yield Assistant
----------------------------------------------------------------------
A chatbot that answers questions about district-level rice yield using
THIS PROJECT'S OWN data and models. Two layers:

  1. Built-in layer (always on, no API key, no internet):
     understands a question (district / state / year / intent), looks the
     answer up in the data and saved model outputs, and replies in plain
     English with a small table / chart. Because every number comes from
     the dataset or the trained models, it cannot make numbers up.

  2. Optional LLM layer (Claude or Gemini API key needed):
     takes the built-in answer as FACTS and rewrites it conversationally,
     and lets people ask in Hindi / Telugu / Bengali / Tamil (the question is
     translated to English for lookup, the answer is written back in the
     user's language). The LLM is told to use ONLY the supplied facts.
     If no key is set, or a call fails, the built-in answer is shown.

This file has no Streamlit dependency, so it can be tested on its own.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "RICE YIELD (Kg per ha)"

# ----------------------------------------------------------------------------
# DATA-TRUST SWITCH — read this one.
# Checks on the project dataset showed that from 2016 onward BOTH the climate
# columns and the yields behave very differently from the years before:
#   * year-to-year wiggle around a straight line drops to ~25% of its 2006-15
#     level (yield: ~230 -> ~57 kg/ha), national mean yield rises every year
#     2016-2025 with no dips, rainfall slides smoothly 1035 -> 954 mm, and a few
#     Gujarat/Rajasthan districts have NEGATIVE rainfall in those years only.
# That pattern looks like extrapolated / smoothed values rather than observed
# ones. Until you have traced where the 2016+ rows came from, the chatbot
# labels anything based on those years as indicative.
#   -> Set to None once you have verified the 2016-2025 data are real.
# ----------------------------------------------------------------------------
DATA_TRUSTED_THROUGH = 2015

# "Normal year" scenario for next-season forecasts = district median of the
# most recent N years of each climate feature.
SCENARIO_YEARS = 10

CLIMATE_FEATURES = [
    "TEMP_ANNUAL_C", "TEMP_JUN_SEP_C", "TEMP_OCT_DEC_C", "RAIN_ANNUAL_mm", "RAIN_JUN_SEP_mm",
    "RAIN_JUN_mm", "RAIN_JUL_mm", "RAIN_AUG_mm", "RAIN_SEP_mm", "ETo_Kharif_seasonal",
    "ARIDITY_INDEX", "SPI_Kharif", "SPI_Annual", "RAIN_CONCENTRATION", "HEAT_STRESS",
    "COLD_STRESS", "TEMP_OPTIMALITY", "GDD_Kharif",
]
RAIN_FEATURES = ["RAIN_ANNUAL_mm", "RAIN_JUN_SEP_mm", "RAIN_JUN_mm", "RAIN_JUL_mm",
                 "RAIN_AUG_mm", "RAIN_SEP_mm", "ARIDITY_INDEX"]

# Simple, transparent what-if knobs (assumptions, not forecasts of weather).
SCENARIOS = {
    "drought":  dict(label="a drought year (rain -30%, SPI -1.5)", rain=0.70, spi=-1.5, temp=0.0, heat=1.0),
    "wet":      dict(label="a very wet monsoon (rain +25%, SPI +1.0)", rain=1.25, spi=1.0, temp=0.0, heat=1.0),
    "heatwave": dict(label="a heatwave year (+1.5 °C, heat stress x1.3)", rain=1.0, spi=None, temp=1.5, heat=1.3),
}

FEATURE_LABELS = {
    "YIELD_ROLLING_5Y": "average yield of the previous 5 years (recent productivity level)",
    "YIELD_LAG1": "last year's yield",
    "YEAR_TREND": "long-run time trend (seeds, irrigation, technology)",
    "SPI_Kharif": "Kharif-season drought index (SPI; negative = dry)",
    "SPI_Annual": "annual drought index (SPI; negative = dry)",
    "RAIN_ANNUAL_mm": "annual rainfall",
    "RAIN_JUN_mm": "June rainfall", "RAIN_JUL_mm": "July rainfall",
    "RAIN_AUG_mm": "August rainfall", "RAIN_SEP_mm": "September rainfall",
    "RAIN_CONCENTRATION": "how concentrated the year's rain is",
    "ARIDITY_INDEX": "aridity index", "HEAT_STRESS": "heat-stress index",
    "COLD_STRESS": "cold-stress index", "TEMP_ANNUAL_C": "annual mean temperature",
    "TEMP_OCT_DEC_C": "Oct-Dec temperature", "TEMP_JUN_SEP_C": "Jun-Sep temperature",
    "SOIL_PHH2O": "soil pH", "SOIL_SOC": "soil organic carbon", "SOIL_NITROGEN": "soil nitrogen",
    "SOIL_CLAY": "soil clay content", "SOIL_SAND": "soil sand content", "SOIL_SILT": "soil silt content",
    "SOIL_BDOD": "soil bulk density", "SOIL_CEC": "soil cation-exchange capacity",
    "PH_RAIN_INTERACTION": "soil pH x rainfall interaction", "N_TEMP_INTERACTION": "soil N x temperature interaction",
    "Lat": "latitude (location)", "Lon": "longitude (location)", "STATE_ENC": "state",
}

STATE_ALIASES = {"odisha": "Orissa", "bengal": "West Bengal", "tamilnadu": "Tamil Nadu",
                 "andhra": "Andhra Pradesh", "uttaranchal": "Uttarakhand"}
DISTRICT_ALIASES = {"vizag": "Visakhapatnam", "bengaluru": "Bangalore", "mysuru": "Mysore",
                    "prayagraj": "Allahabad", "tanjore": "Thanjavur", "nellore": "S.P.S. Nellore",
                    "kadapa": "Kadapa YSR", "cuddapah": "Kadapa YSR", "nilgiris": "The Nilgiris",
                    "bulandshahr": "Buland Shahar", "calcutta": "24 Parganas", "burdwan": "Burdwan"}

STOPWORDS = {
    "yield", "yields", "rice", "paddy", "what", "when", "which", "where", "will", "this", "that", "kharif",
    "district", "state", "forecast", "predict", "prediction", "compare", "trend", "advice", "farmers",
    "farmer", "about", "tell", "show", "give", "please", "year", "years", "next", "last", "top", "best",
    "highest", "lowest", "average", "india", "model", "models", "accurate", "accuracy", "factors",
    "should", "could", "would", "there", "their", "these", "those", "between", "versus", "drought",
    "rainfall", "monsoon", "temperature", "irrigation", "fertiliser", "fertilizer", "production",
    "data", "dataset", "source", "sources", "reliable", "regions", "region", "area", "areas", "farming",
    "forecasts", "performing", "performance", "values", "number", "numbers",
}

HELP_TEXT = (
    "I can answer questions about **district-level rice yield in India (1966–2025)** using this project's "
    "data and models. Try:\n\n"
    "- *What was the rice yield in Patna in 2010?*\n"
    "- *What will rice yield be in Patna this Kharif?*  /  *What if drought hits Guntur?*\n"
    "- *Compare Patna and Gaya*\n"
    "- *Top 5 districts in Punjab*  /  *Lowest yield districts in 2015*\n"
    "- *Yield trend in West Bengal since 2000*\n"
    "- *What factors drive rice yield?*  /  *How accurate are the models?*\n"
    "- *How sure are you about the Ludhiana forecast?*\n"
    "- *Advice for farmers in Raichur*\n\n"
    "Yields are in kg/ha. Forecasts are **indicative** — they are not a guarantee."
)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def kg(v: float) -> str:
    return f"{v:,.0f} kg/ha ({v / 1000:.2f} t/ha)"


def find_file(base_dir: str, names) -> str | None:
    """Look for the first existing file among `names` in the usual places."""
    for folder in (base_dir, os.path.join(base_dir, "data"),
                   os.path.join(base_dir, "deployment"), os.path.dirname(base_dir)):
        for n in names:
            p = os.path.join(folder, n)
            if os.path.isfile(p):
                return p
    return None


def is_non_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ord(c) > 0x024F) / len(letters) > 0.3


def _aliases(name: str) -> set[str]:
    """'Tiruchirapalli / Trichy' -> {'tiruchirapalli trichy', 'tiruchirapalli', 'trichy'} etc."""
    out = {name}
    inner = re.findall(r"\((.*?)\)", name)
    base = re.sub(r"\(.*?\)", " ", name)
    out.update(p for p in base.split("/"))
    out.update(i for i in inner if "now part" not in i.lower())
    return {a for a in (_norm(x) for x in out) if len(a) >= 3}


@dataclass
class Entities:
    districts: list = field(default_factory=list)
    states: list = field(default_factory=list)
    years: list = field(default_factory=list)
    scenario: str | None = None
    rain_change: float | None = None      # e.g. -0.2 for "20% less rain"
    topn: int = 5
    national: bool = False


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    chart: pd.DataFrame | None = None      # index = year (as str)
    suggestions: list = field(default_factory=list)
    intent: str = ""
    polish: bool = True                    # let the LLM rewrite this answer?
    offline_text: str = ""
    llm_used: bool = False
    llm_error: str = ""
    llm_calls: int = 0

    def facts(self) -> str:
        t = self.text
        if self.table is not None and len(self.table):
            t += "\n\nTABLE:\n" + self.table.to_string(index=False)
        return t


# ----------------------------------------------------------------------------
# knowledge base
# ----------------------------------------------------------------------------
class RiceKnowledge:
    def __init__(self, history, test_predictions=None, model_summary=None, shap_importance=None,
                 ssm_forecasts=None, ssm_diagnostics=None, ssm_next_year=None):
        h = history.copy()
        h = h[h[TARGET].notna() & (h[TARGET] > 0)]
        self.h = h.sort_values(["Dist Name", "Year"]).reset_index(drop=True)
        self.pred = test_predictions
        self.summary = model_summary
        self.shap = shap_importance
        self.ssm = ssm_forecasts
        self.diag = ssm_diagnostics
        self.ssm_next = ssm_next_year

        self.districts = sorted(self.h["Dist Name"].unique())
        self.states = sorted(self.h["State Name"].unique())
        self.first_year = int(self.h["Year"].min())
        self.last_year = int(self.h["Year"].max())
        self.state_of = self.h.drop_duplicates("Dist Name").set_index("Dist Name")["State Name"].to_dict()
        self._build_alias_index()

    @classmethod
    def from_dir(cls, base_dir: str) -> "RiceKnowledge":
        hist_path = find_file(base_dir, ["rice_model_ready_full.csv", "rice_dataset_final.csv",
                                         "rice_model_ready_baseline.csv"])
        if hist_path is None:
            raise FileNotFoundError("Could not find the yield dataset (rice_model_ready_full.csv or "
                                    "rice_dataset_final.csv) next to the app or in a data/ folder.")

        def opt(name, **kw):
            p = find_file(base_dir, [name])
            return pd.read_csv(p, **kw) if p else None

        return cls(pd.read_csv(hist_path), opt("test_predictions.csv"), opt("model_summary.csv"),
                   opt("shap_global_importance.csv", index_col=0), opt("ssm_forecasts.csv"),
                   opt("ssm_diagnostics.csv"), opt("ssm_next_year.csv"))

    # -- alias index ---------------------------------------------------------
    def _build_alias_index(self):
        d = []
        for name in self.districts:
            for a in _aliases(name):
                d.append((a, name))
        for a, name in DISTRICT_ALIASES.items():
            if name in self.state_of:
                d.append((_norm(a), name))
        d.sort(key=lambda x: -len(x[0]))                  # longest first
        self._dist_alias = d
        self._alias_keys = sorted({a for a, _ in d})
        self._alias_to_name = {}
        for a, n in d:
            self._alias_to_name.setdefault(a, n)

        s = [(_norm(x), x) for x in self.states]
        for a, name in STATE_ALIASES.items():
            if name in self.states:
                s.append((_norm(a), name))
        s.sort(key=lambda x: -len(x[0]))
        self._state_alias = s

    # -- entity extraction ---------------------------------------------------
    def find_districts(self, text: str) -> list:
        t = " " + _norm(text) + " "
        hits = []
        for alias, canon in self._dist_alias:
            key = f" {alias} "
            pos = t.find(key)
            if pos >= 0 and canon not in [c for _, c in hits]:
                hits.append((pos, canon))
                t = t[:pos] + " " * len(key) + t[pos + len(key):]
        if not hits:                                       # typo-tolerant pass
            toks = [w for w in _norm(text).split() if len(w) >= 5 and w not in STOPWORDS]
            cands = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
            for c in cands:
                m = difflib.get_close_matches(c, self._alias_keys, n=1, cutoff=0.86)
                if m and self._alias_to_name[m[0]] not in [x for _, x in hits]:
                    hits.append((text.lower().find(c.split()[0]), self._alias_to_name[m[0]]))
        hits.sort()
        return [c for _, c in hits]

    def find_states(self, text: str, consumed_districts=()) -> list:
        t = " " + _norm(text) + " "
        for dname in consumed_districts:                   # "West Dinajpur" must not count as a state hit
            for a in _aliases(dname):
                t = t.replace(f" {a} ", " ")
        hits = []
        for alias, canon in self._state_alias:
            key = f" {alias} "
            pos = t.find(key)
            if pos >= 0 and canon not in [c for _, c in hits]:
                hits.append((pos, canon))
                t = t[:pos] + " " * len(key) + t[pos + len(key):]
        hits.sort()
        return [c for _, c in hits]

    def extract(self, question: str) -> Entities:
        ql = question.lower()
        e = Entities()
        e.districts = self.find_districts(question)
        e.states = self.find_states(question, e.districts)
        e.years = [int(y) for y in re.findall(r"\b(19[6-9]\d|20[0-4]\d)\b", ql)]
        m = re.search(r"\b(?:top|best|highest|lowest|worst|bottom)\s*(\d{1,2})\b|\b(\d{1,2})\s+(?:best|top|highest|lowest|worst|districts)", ql)
        if m:
            e.topn = max(1, min(15, int(m.group(1) or m.group(2))))
        e.national = bool(re.search(r"\b(india|national|nationwide|country|all districts|overall)\b", ql))

        if re.search(r"\b(drought|dry spell|dry year|monsoon fail|deficit|less rain|low rain|poor monsoon)", ql):
            e.scenario = "drought"
        elif re.search(r"heat ?wave|very hot|extreme heat|hotter|heat stress|heat spell", ql):
            e.scenario = "heatwave"
        elif re.search(r"\b(flood|very wet|excess rain|heavy rain|good monsoon|more rain|wet year)", ql):
            e.scenario = "wet"
        m = re.search(r"(\d{1,2})\s*%", ql)
        if m and re.search(r"\b(rain|rainfall|monsoon)\b", ql):
            neg = re.search(r"\b(less|lower|below|reduc\w*|fewer|drop\w*|fall\w*|declin\w*|decreas\w*|down|deficit|shortfall)\b", ql)
            pos = re.search(r"\b(more|higher|above|extra|excess|increas\w*|rise\w*|rising|up)\b", ql)
            if bool(neg) != bool(pos):
                e.rain_change = (-1 if neg else 1) * min(int(m.group(1)), 60) / 100.0
        return e

    def unmatched_place(self, question: str) -> str | None:
        """A place-like phrase after in/for/of/at that is NOT a known district or state (e.g. 'Springfield')."""
        pat = (r"\b(?:in|for|of|at|near|around)\s+([A-Za-z][A-Za-z\.\- ]{2,30}?)"
               r"(?=\s+(?:in|for|during|this|next|last|district|state|kharif|rabi|season|year)\b|\s+\d{4}|[?.!,]|$)")
        for m in re.finditer(pat, question, flags=re.I):
            cand = m.group(1).strip()
            words = _norm(cand).split()
            if not words or all(w in STOPWORDS or w in {"the", "a", "an", "my", "our", "general", "detail", "future"} for w in words):
                continue
            if self.find_states(cand):
                continue
            return cand
        return None

    # -- data access ---------------------------------------------------------
    def series(self, district: str) -> pd.Series:
        g = self.h[self.h["Dist Name"] == district]
        return g.set_index("Year")[TARGET]

    def national_series(self) -> pd.Series:
        return self.h.groupby("Year")[TARGET].mean()

    def state_series(self, state: str) -> pd.Series:
        return self.h[self.h["State Name"] == state].groupby("Year")[TARGET].mean()

    def metric(self, name_part: str, col: str = "test_r2"):
        if self.summary is None:
            return None
        m = self.summary[self.summary["model"].str.contains(name_part, case=False, regex=False)]
        return float(m[col].iloc[0]) if len(m) else None

    def ssm_coverage(self):
        if self.diag is None or "ci_coverage" not in self.diag:
            return None
        ok = self.diag[self.diag["status"] == "ok"]["ci_coverage"].dropna()
        return float(ok.mean()) if len(ok) else None

    def trust_note(self, year: int | None) -> str:
        if DATA_TRUSTED_THROUGH and year and year > DATA_TRUSTED_THROUGH:
            return (f"\n\n⚠️ *Data note: values after {DATA_TRUSTED_THROUGH} in this dataset look unusually smooth "
                    f"(possibly extrapolated rather than observed), so treat them as indicative until the source is verified.*")
        return ""

    # -- next-season forecast machinery -------------------------------------
    def build_next_row(self, district: str, feature_cols, scenario=None, rain_change=None) -> pd.DataFrame:
        g = self.h[self.h["Dist Name"] == district].sort_values("Year")
        last = g.iloc[-1].copy()
        row = last.copy()
        row["YIELD_LAG1"] = last[TARGET]                               # next year's "last year" = latest yield
        row["YIELD_ROLLING_5Y"] = g[TARGET].tail(5).mean()             # mean of the 5 most recent yields
        if len(g) >= 2 and "YEAR_TREND" in g:                          # continue the column's own convention
            dy = max(1, int(g["Year"].iloc[-1] - g["Year"].iloc[-2]))
            row["YEAR_TREND"] = last["YEAR_TREND"] + (g["YEAR_TREND"].iloc[-1] - g["YEAR_TREND"].iloc[-2]) / dy
        recent = g.tail(SCENARIO_YEARS)
        for c in CLIMATE_FEATURES:                                     # "normal year" = recent median
            if c in g.columns:
                row[c] = recent[c].median()
        for c in ("SPI_Kharif", "SPI_Annual"):
            if c in g.columns:
                row[c] = 0.0                                           # SPI 0 = median rainfall by definition
        rain, spi, temp, heat = 1.0, None, 0.0, 1.0
        if scenario in SCENARIOS:
            k = SCENARIOS[scenario]
            rain, spi, temp, heat = k["rain"], k["spi"], k["temp"], k["heat"]
        if rain_change is not None:
            rain = 1.0 + rain_change
            spi = float(np.clip(rain_change * 5, -2.5, 2.5))           # rough: -20% rain ~ SPI -1
        for c in RAIN_FEATURES:
            if c in row.index and pd.notna(row[c]):
                row[c] = row[c] * rain
        if spi is not None:
            for c in ("SPI_Kharif", "SPI_Annual"):
                if c in row.index:
                    row[c] = spi
        for c in ("TEMP_ANNUAL_C", "TEMP_OCT_DEC_C", "TEMP_JUN_SEP_C"):
            if c in row.index and pd.notna(row[c]):
                row[c] = row[c] + temp
        if "HEAT_STRESS" in row.index:
            row["HEAT_STRESS"] = row["HEAT_STRESS"] * heat
        if "PH_RAIN_INTERACTION" in row.index:
            row["PH_RAIN_INTERACTION"] = row["SOIL_PHH2O"] * row["RAIN_ANNUAL_mm"]
        if "N_TEMP_INTERACTION" in row.index:
            row["N_TEMP_INTERACTION"] = row["SOIL_NITROGEN"] * row["TEMP_ANNUAL_C"]
        return pd.DataFrame([row])[list(feature_cols)].astype(float)

    def predict_next(self, district, models, feature_cols, scenario=None, rain_change=None) -> dict:
        X = self.build_next_row(district, feature_cols, scenario, rain_change)
        return {name: float(m.predict(X)[0]) for name, m in models.items()}

    # -- history + model chart ------------------------------------------------
    def history_chart(self, district: str, extra: pd.DataFrame | None = None, years_back: int = 25) -> pd.DataFrame:
        s = self.series(district)
        df = pd.DataFrame({"Actual": s})
        if self.pred is not None:
            p = self.pred[self.pred["Dist Name"] == district].set_index("Year")
            for col, lab in (("rf_pred", "Random Forest"), ("xgb_pred", "XGBoost")):
                if col in p:
                    df[lab] = p[col]
        if self.ssm is not None:
            q = self.ssm[self.ssm["Dist Name"] == district].set_index("Year")
            if "ssm_forecast" in q:
                df["SSM"] = q["ssm_forecast"]
        if extra is not None:
            df = df.join(extra, how="outer")
        df = df.sort_index().tail(years_back)
        df.index = df.index.astype(int).astype(str)
        return df


# ----------------------------------------------------------------------------
# intent detection
# ----------------------------------------------------------------------------
def detect_intent(q: str, e: Entities) -> str:
    ql = q.lower().strip()
    has = lambda pat: re.search(pat, ql) is not None
    if has(r"^(hi+|hello|hey|namaste|good (morning|afternoon|evening)|thanks|thank you)\b") and len(ql.split()) <= 4:
        return "greeting"
    if has(r"\b(help|what can you do|examples?|how do i use|what do you know)\b"):
        return "help"
    if has(r"\b(accura\w*|r2|r²|rmse|which model|best model|how good|how reliable|reliab\w*|compare (the )?models|model comparison|can i trust|how well|how correct)\b"):
        return "model_perf"
    if has(r"\bperform\w*\b") and has(r"\b(models?|algorithms?|forecasts?|predictions?|xgboost|random forest|ssm)\b"):
        return "model_perf"
    if has(r"\b(features?|shap|importan\w*|drivers?|drives?|driving|factors?|influenc\w*|determin\w*|matters?)\b") and not has(r"\b(advice|advise|should)\b"):
        return "drivers"
    if has(r"\b(uncertain\w*|confidence|how sure|how certain|margin of error|error bars?|interval|how confident)\b"):
        return "uncertainty"
    whatif = has(r"\b(what if|what happens|suppose|scenario|if there is|if the|if it)\b")
    if (e.scenario or e.rain_change is not None) and (whatif or has(r"\b(predict|forecast|expect|yield|will)\b")) and not has(r"\b(advice|advise|should|recommend)\b"):
        return "forecast"
    if has(r"\b(advice|advis\w*|recommend\w*|suggest\w*|what should|should (i|we|farmers)|how (to|can|do|should) .*(improve|increase|protect|manage|cope)|irrigat\w*|fertili[sz]\w*|tips?|practices?)\b"):
        return "advisory"
    if has(r"\b(predict\w*|forecast\w*|expect\w*|will|going to|next (year|season|kharif)|upcoming|coming (year|season|kharif)|projected|projection|outlook|estimate\w*|this (year|kharif|season))\b") \
            or any(y > 2025 for y in e.years):
        return "forecast"
    if (len(e.districts) >= 2 or len(e.states) >= 2) and has(r"\b(compar\w*|vs\.?|versus|between|better|difference|and|or)\b"):
        return "compare"
    if has(r"\b(top|best|highest|lowest|worst|bottom|rank\w*|most|least|leading|lagging)\b") and not e.districts:
        return "ranking"
    if has(r"\b(trend|growth|grown|grew|increas\w*|decreas\w*|declin\w*|change[ds]?|over the years|since|history|historical|progress|improv\w*)\b"):
        return "trend"
    if e.districts:
        return "yield_lookup"
    if e.states or e.national:
        return "state_summary" if e.states else "trend"
    if has(r"\b(yield|production|productivity|harvest)\b"):
        return "unknown_scope"
    return "unknown"


NEEDS_DISTRICT = {"yield_lookup", "forecast", "uncertainty", "advisory"}


# ----------------------------------------------------------------------------
# answer builders
# ----------------------------------------------------------------------------
def _pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


def ans_yield(kb: RiceKnowledge, d: str, year: int | None) -> Answer:
    s, st = kb.series(d), kb.state_of[d]
    year = int(year) if year else int(s.index.max())
    if year not in s.index:
        return Answer(f"I don't have a {year} yield for **{d}** ({st}). Its data covers "
                      f"{int(s.index.min())}–{int(s.index.max())}.", intent="yield_lookup",
                      suggestions=[f"Yield in {d} in {int(s.index.max())}", f"Yield trend in {d}"])
    v = float(s[year])
    lines = [f"**{d}, {st} — {year}:** {kg(v)}."]
    sy = kb.h[(kb.h["State Name"] == st) & (kb.h["Year"] == year)][TARGET]
    if len(sy) > 1:
        rank = int((sy > v).sum() + 1)
        lines.append(f"That is {abs(_pct(v, sy.mean())):.0f}% {'above' if v >= sy.mean() else 'below'} the "
                     f"{st} district average ({sy.mean():,.0f} kg/ha) — rank {rank} of {len(sy)} districts.")
    prior = s[(s.index < year) & (s.index >= year - 5)]
    if len(prior) >= 3:
        lines.append(f"Versus its previous 5-year average ({prior.mean():,.0f} kg/ha): {_pct(v, prior.mean()):+.0f}%.")

    table = None
    if kb.pred is not None:
        p = kb.pred[(kb.pred["Dist Name"] == d) & (kb.pred["Year"] == year)]
        if len(p):
            r = p.iloc[0]
            rows = [("Actual", v)]
            for col, lab in (("naive_pred", "Naive (last year's yield)"), ("rf_pred", "Random Forest"), ("xgb_pred", "XGBoost")):
                if col in p:
                    rows.append((lab, float(r[col])))
            if kb.ssm is not None:
                q = kb.ssm[(kb.ssm["Dist Name"] == d) & (kb.ssm["Year"] == year)]
                if len(q):
                    rows.append(("State Space Model", float(q.iloc[0]["ssm_forecast"])))
            table = pd.DataFrame(rows, columns=["Source", "kg/ha"])
            table["Error vs actual"] = [("—" if i == 0 else f"{x - v:+,.0f}") for i, x in enumerate(table["kg/ha"])]
            table["kg/ha"] = table["kg/ha"].round(0).astype(int)
            lines.append(f"What the models estimated for {year} (test period) is shown below.")
    text = " ".join(lines[:1]) + "\n\n" + "\n\n".join(lines[1:]) + kb.trust_note(year)
    return Answer(text, table=table, chart=kb.history_chart(d), intent="yield_lookup",
                  suggestions=[f"What will rice yield be in {d} next year?", f"Advice for farmers in {d}",
                               f"Yield trend in {d}"])


def ans_forecast(kb: RiceKnowledge, d: str, e: Entities, models, feature_cols, models_note: str = "") -> Answer:
    s, st = kb.series(d), kb.state_of[d]
    last_y, last_v = int(s.index.max()), float(s.iloc[-1])
    year = e.years[-1] if e.years else last_y + 1
    if year <= last_y:
        return ans_yield(kb, d, year)
    notes = []
    if year > last_y + 1:
        notes.append(f"I only forecast one season ahead, so here is the {last_y + 1} outlook "
                     f"(longer horizons compound the uncertainty).")
    year = last_y + 1

    preds, err = {}, ""
    if models and feature_cols:
        try:
            preds = kb.predict_next(d, models, feature_cols, e.scenario, e.rain_change)
        except Exception as ex:                                       # never crash the chat
            err = f"(model prediction unavailable: {type(ex).__name__})"
    scen_label = None
    if e.scenario in SCENARIOS:
        scen_label = SCENARIOS[e.scenario]["label"]
    elif e.rain_change is not None:
        scen_label = f"rainfall {e.rain_change * 100:+.0f}% vs normal"

    rows = [("Naive (same as last year)", last_v)]
    for name, v in preds.items():
        rows.append((name, v))
    ssm_txt = ""
    if kb.ssm_next is not None and not scen_label:
        q = kb.ssm_next[kb.ssm_next["Dist Name"] == d]
        if len(q):
            r = q.iloc[0]
            rows.append(("State Space Model", float(r["ssm_forecast"])))
            ssm_txt = f"The State Space Model's 95% range is **{r['ssm_ci_lower']:,.0f}–{r['ssm_ci_upper']:,.0f} kg/ha**."
    table = pd.DataFrame({"Source": [r[0] for r in rows],
                          f"{year} estimate (kg/ha)": [int(round(r[1])) for r in rows],
                          f"vs {last_y} actual": [f"{_pct(r[1], last_v):+.1f}%" for r in rows]})

    ml = {k: v for k, v in preds.items()}
    head_name = None
    if ml:
        best = max(ml, key=lambda n: (kb.metric(n) if kb.metric(n) is not None else -9))
        head_name, head = best, ml[best]
    elif ssm_txt:
        head_name, head = "State Space Model", float(rows[-1][1])
    else:
        head_name, head = "Naive (same as last year)", last_v

    title = f"**{d}, {st} — outlook for {year}**" + (f" under {scen_label}" if scen_label else "")
    lines = [title,
             f"Indicative yield: **{head:,.0f} kg/ha** ({head_name}), {_pct(head, last_v):+.1f}% vs {last_y} ({last_v:,.0f} kg/ha)."]
    if ssm_txt:
        lines.append(ssm_txt)
    rmse = kb.metric(head_name, "test_rmse") if head_name else None
    if rmse:
        lines.append(f"On past test years the {head_name} was typically off by about ±{rmse:,.0f} kg/ha (RMSE).")
    if scen_label and ml:
        try:
            base = kb.predict_next(d, models, feature_cols)
            deltas = {n: ml[n] - base[n] for n in ml}
            dl = ", ".join(f"{n}: {v:+,.0f} kg/ha" for n, v in deltas.items())
            lines.append(f"Versus a normal-weather year this scenario changes the estimate by {dl}. "
                         f"(The models lean mostly on recent yield history, so weather scenarios move the estimate only modestly.)")
            bad_dir = e.scenario in ("drought", "heatwave") and any(v > 0 for v in deltas.values())
            if bad_dir or len({v > 0 for v in deltas.values()}) > 1:
                lines.append("⚠️ The models disagree on the direction (or move the 'wrong' way), which means weather has little "
                             "reliable influence on their estimates here — use this as a rough sensitivity check, not a forecast.")
        except Exception:
            pass
    lines.append(f"*Assumptions: a normal-weather year (this district's recent median climate), {last_y} yield and "
                 f"the 5-year average as momentum. No live weather forecast is used.*")
    if err:
        lines.append(err)
    if models_note:
        lines.append(models_note)
    text = "\n\n".join(notes + lines) + kb.trust_note(last_y)

    extra = pd.DataFrame(index=[last_y, year])
    extra.loc[last_y, "Forecast (best model)"] = last_v
    extra.loc[year, "Forecast (best model)"] = head
    return Answer(text, table=table, chart=kb.history_chart(d, extra, years_back=15), intent="forecast",
                  suggestions=[f"How sure are you about {d} {year}?", f"Advice for farmers in {d}",
                               f"What if drought hits {d}?", "How accurate are the models?"])


def ans_uncertainty(kb: RiceKnowledge, d: str, e: Entities) -> Answer:
    st, s = kb.state_of[d], kb.series(d)
    last_y = int(s.index.max())
    year = e.years[-1] if e.years else last_y + 1
    cov = kb.ssm_coverage()
    cov_txt = (f" Across districts, these intervals contained the real value {cov * 100:.0f}% of the time "
               f"(a perfectly calibrated 95% interval would be 95%).") if cov is not None else ""
    if year > last_y and kb.ssm_next is not None:
        q = kb.ssm_next[kb.ssm_next["Dist Name"] == d]
        if len(q):
            r = q.iloc[0]
            w = r["ssm_ci_upper"] - r["ssm_ci_lower"]
            return Answer(f"**{d}, {st} — {int(r['Year'])} forecast uncertainty (State Space Model):** "
                          f"{r['ssm_forecast']:,.0f} kg/ha, 95% range **{r['ssm_ci_lower']:,.0f}–{r['ssm_ci_upper']:,.0f}** "
                          f"(width {w:,.0f} kg/ha).{cov_txt}\n\nRead it as: if the model's assumptions hold, the real yield "
                          f"should fall inside this band about 19 times out of 20." + kb.trust_note(last_y),
                          intent="uncertainty", suggestions=[f"What will rice yield be in {d} next year?"])
    if kb.ssm is not None and year <= last_y:
        q = kb.ssm[(kb.ssm["Dist Name"] == d) & (kb.ssm["Year"] == year)]
        if len(q):
            r = q.iloc[0]
            inside = r["ssm_ci_lower"] <= r["actual"] <= r["ssm_ci_upper"]
            return Answer(f"**{d}, {st} — {year} (State Space Model):** forecast {r['ssm_forecast']:,.0f} kg/ha, "
                          f"95% range {r['ssm_ci_lower']:,.0f}–{r['ssm_ci_upper']:,.0f}. The actual yield was "
                          f"{r['actual']:,.0f} kg/ha, which was **{'inside' if inside else 'outside'}** the range.{cov_txt}"
                          + kb.trust_note(year), intent="uncertainty", chart=kb.history_chart(d))
    return Answer("I don't have State Space Model intervals for that district/year yet. Run "
                  "`02_train_state_space_model.py` (it writes `ssm_forecasts.csv` and `ssm_next_year.csv`) "
                  "and redeploy.", intent="uncertainty")


def ans_ranking(kb: RiceKnowledge, q: str, e: Entities) -> Answer:
    ql = q.lower()
    year = e.years[-1] if e.years else kb.last_year
    sub = kb.h[kb.h["Year"] == year]
    scope = "India (all districts in the dataset)"
    if e.states:
        sub, scope = sub[sub["State Name"].isin(e.states)], " & ".join(e.states)
    if sub.empty:
        return Answer(f"No yield data for {year}.", intent="ranking")
    low = bool(re.search(r"\b(lowest|worst|bottom|least|lagging|poorest)\b", ql))
    sub = sub.sort_values(TARGET, ascending=low).head(e.topn)
    table = pd.DataFrame({"Rank": range(1, len(sub) + 1), "District": sub["Dist Name"].values,
                          "State": sub["State Name"].values, "Yield (kg/ha)": sub[TARGET].round(0).astype(int).values})
    text = (f"**{'Lowest' if low else 'Highest'}-yielding districts in {scope}, {year}:**")
    if low and (table["Yield (kg/ha)"] < 300).any():
        text += ("\n\n*Values under ~300 kg/ha are implausibly low for rice and may be data-quality problems "
                 "(e.g. partial reporting) rather than real harvests.*")
    text += kb.trust_note(year)
    return Answer(text, table=table, intent="ranking",
                  suggestions=[f"Yield in {table.iloc[0]['District']} in {year}", "Yield trend in India"])


def _trend_text(label: str, s: pd.Series, e: Entities, kb: RiceKnowledge) -> Answer:
    lo = e.years[0] if e.years else int(s.index.min())
    hi = e.years[1] if len(e.years) > 1 else int(s.index.max())
    w = s[(s.index >= lo) & (s.index <= hi)]
    if len(w) < 6:
        return Answer(f"Not enough data for {label} between {lo} and {hi}.", intent="trend")
    start, end = w.iloc[:3].mean(), w.iloc[-3:].mean()
    n = int(w.index.max() - w.index.min())
    cagr = ((end / start) ** (1 / max(n - 2, 1)) - 1) * 100
    slope = np.polyfit(w.index.values, w.values, 1)[0]
    dec = (w.groupby((w.index // 10) * 10).mean().round(0).astype(int))
    table = pd.DataFrame({"Decade": [f"{k}s" for k in dec.index], "Average yield (kg/ha)": dec.values})
    text = (f"**{label}, {int(w.index.min())}–{int(w.index.max())}:** yield moved from about {start:,.0f} kg/ha "
            f"(first 3 years) to {end:,.0f} kg/ha (last 3 years), {_pct(end, start):+.0f}% overall — roughly "
            f"{cagr:.1f}% per year, or {slope:+,.0f} kg/ha each year on a straight-line fit." + kb.trust_note(int(w.index.max())))
    chart = pd.DataFrame({"Yield": s.loc[w.index]})
    chart.index = chart.index.astype(int).astype(str)
    return Answer(text, table=table, chart=chart, intent="trend")


def ans_state(kb: RiceKnowledge, st: str, year: int | None) -> Answer:
    year = year or kb.last_year
    sub = kb.h[(kb.h["State Name"] == st) & (kb.h["Year"] == year)].sort_values(TARGET, ascending=False)
    if sub.empty:
        return Answer(f"No {st} data for {year}.", intent="state_summary")
    ss = kb.state_series(st)
    t10 = ss.get(year - 10)
    lines = [f"**{st}, {year}:** average district yield {kg(sub[TARGET].mean())} across {len(sub)} districts.",
             f"Best: {sub.iloc[0]['Dist Name']} ({sub.iloc[0][TARGET]:,.0f} kg/ha). Lowest: {sub.iloc[-1]['Dist Name']} "
             f"({sub.iloc[-1][TARGET]:,.0f} kg/ha)."]
    if t10 is not None and pd.notna(t10):
        lines.append(f"Ten years earlier ({year - 10}) the state average was {t10:,.0f} kg/ha ({_pct(sub[TARGET].mean(), t10):+.0f}%).")
    chart = pd.DataFrame({"State average": ss})
    chart.index = chart.index.astype(int).astype(str)
    return Answer("\n\n".join(lines) + kb.trust_note(year), chart=chart, intent="state_summary",
                  suggestions=[f"Top 5 districts in {st}", f"Yield trend in {st}"])


def ans_drivers(kb: RiceKnowledge) -> Answer:
    if kb.shap is None or kb.shap.empty:
        return Answer("Feature-importance (SHAP) results aren't available yet — run `01_train_final_models.py` "
                      "and redeploy `shap_global_importance.csv`.", intent="drivers")
    imp = kb.shap.iloc[:, 0].sort_values(ascending=False).head(10)
    total = float(kb.shap.iloc[:, 0].sum())
    table = pd.DataFrame({"Factor": [FEATURE_LABELS.get(f, f) for f in imp.index],
                          "Share of influence": [f"{v / total * 100:.0f}%" for v in imp.values]})
    hist = [f for f in imp.index[:3] if f in ("YIELD_ROLLING_5Y", "YIELD_LAG1", "YEAR_TREND")]
    text = ("**What the XGBoost model relies on most (mean |SHAP| value):**\n\n"
            + ("Recent yield history dominates — fields that yielded well recently tend to keep doing so, "
               "so weather and soil explain the *adjustments* around that baseline. " if hist else "")
            + "These are statistical associations in the data, not proven causes.")
    return Answer(text, table=table, intent="drivers",
                  suggestions=["How accurate are the models?", "What if drought hits Guntur?"])


def ans_model_perf(kb: RiceKnowledge) -> Answer:
    if kb.summary is None:
        return Answer("Model results aren't available yet — run `01_train_final_models.py`.", intent="model_perf")
    sm = kb.summary
    t = pd.DataFrame({"Model": sm["model"], "R² 2006–2025": sm["test_r2"].round(3)})
    if "r2_2006_2015" in sm:
        t["R² 2006–2015"] = sm["r2_2006_2015"].round(3)
    if "r2_2016_2025" in sm:
        t["R² 2016–2025"] = sm["r2_2016_2025"].round(3)
    t["RMSE (kg/ha)"] = sm["test_rmse"].round(0).astype(int)
    lines = ["**Model scores on held-out years (trained on 1966–2005):**"]

    def beaters(col):
        nv = sm.loc[sm["model"].str.contains("naive", case=False), col]
        if nv.empty:
            return None
        return [m for m, v in zip(sm["model"], sm[col]) if "naive" not in m.lower() and v > float(nv.iloc[0])]

    if (b := beaters("test_r2")) is not None:
        lines.append("The benchmark is **naive persistence** (predict = last year's yield). Over the whole 2006–2025 window "
                     + (f"these models beat it: {', '.join(b)}." if b else "no model beats it."))
    if "r2_2006_2015" in sm and (b := beaters("r2_2006_2015")) is not None:
        lines.append("On **2006–2015** (where the data still shows real year-to-year swings) "
                     + (f"these models beat it: {', '.join(b)}." if b else "no model beats it."))
    cov = kb.ssm_coverage()
    if cov is not None:
        lines.append(f"The State Space Model's 95% intervals captured the real value {cov * 100:.0f}% of the time (ideal: 95%).")
    if DATA_TRUSTED_THROUGH and "r2_2016_2025" not in sm:
        lines.append(f"*Note: after {DATA_TRUSTED_THROUGH} the dataset looks unusually smooth, which makes 'same as last year' very "
                     f"hard to beat — re-run `01_train_final_models.py` to see scores split by period.*")
    elif DATA_TRUSTED_THROUGH:
        lines.append(f"*After {DATA_TRUSTED_THROUGH} the data looks unusually smooth (possibly extrapolated), which flatters "
                     f"'same as last year' — the 2006–{DATA_TRUSTED_THROUGH} column is the fairer comparison.*")
    return Answer("\n\n".join(lines), table=t, intent="model_perf", suggestions=["What factors drive rice yield?"])


def _sensitivity(kb: RiceKnowledge, d: str):
    """District-specific: how did yield behave vs its recent average in dry / hot years? (trusted years only)"""
    g = kb.h[kb.h["Dist Name"] == d]
    if DATA_TRUSTED_THROUGH:
        g = g[g["Year"] <= DATA_TRUSTED_THROUGH]
    g = g[g["YIELD_ROLLING_5Y"].notna() & (g["YIELD_ROLLING_5Y"] > 0)]
    if len(g) < 12:
        return [], None
    rel = (g[TARGET] / g["YIELD_ROLLING_5Y"] - 1) * 100
    out, dry_diff = [], None
    spi = g["SPI_Kharif"]
    dry, norm = rel[spi.between(-4, -1)], rel[spi.between(-0.5, 1.0)]
    if len(dry) >= 3 and len(norm) >= 5:
        dry_diff = float(dry.mean() - norm.mean())
        out.append(f"In **{len(dry)} dry Kharif seasons** (SPI ≤ −1) yield was on average {dry.mean():+.0f}% versus its "
                   f"previous 5-year average, compared with {norm.mean():+.0f}% in normal-rainfall seasons "
                   f"({dry.mean() - norm.mean():+.0f} points difference).")
    hs = g["HEAT_STRESS"]
    if hs.notna().sum() >= 12:
        hot, rest = rel[hs >= hs.quantile(0.8)], rel[hs < hs.quantile(0.8)]
        if len(hot) >= 3:
            out.append(f"In its **{len(hot)} hottest seasons** (top 20% heat-stress index) yield changed {hot.mean():+.0f}% "
                       f"versus {rest.mean():+.0f}% in the other years ({hot.mean() - rest.mean():+.0f} points).")
    return out, dry_diff


def ans_advisory(kb: RiceKnowledge, d: str) -> Answer:
    st = kb.state_of[d]
    last = kb.h[kb.h["Dist Name"] == d].iloc[-1]
    uniq = kb.h.drop_duplicates("Dist Name")
    lines = [f"**Pointers for {d}, {st}** — rule-based, from this district's own history and soil data:"]
    sens, dry_diff = _sensitivity(kb, d)
    if sens:
        lines.append("**Weather sensitivity (this district's record):**\n" + "\n".join(f"- {x}" for x in sens))
        if dry_diff is not None and dry_diff <= -5:
            lines.append("➡️ Dry seasons have clearly cost this district yield: secure supplemental irrigation for the critical "
                         "stages (tillering, panicle initiation, flowering) and watch the monsoon onset and breaks.")
        elif dry_diff is not None:
            lines.append("➡️ This district's record shows no clear yield loss in dry seasons (it may be well irrigated), "
                         "so water access is less of a red flag here than elsewhere.")
    else:
        lines.append("There isn't enough reliable history to measure this district's dry/hot-year response.")
    soil = []
    ph = last.get("SOIL_PHH2O")
    if pd.notna(ph):
        if ph < 5.5:
            soil.append(f"Soil pH is about {ph:.1f} (acidic). Strongly acidic soils can lock up phosphorus — get a soil test and "
                        f"let it guide liming and P management.")
        elif ph > 8.0:
            soil.append(f"Soil pH is about {ph:.1f} (alkaline). Check for sodicity; a soil test will say whether gypsum or "
                        f"organic amendments are needed.")
        else:
            soil.append(f"Soil pH is about {ph:.1f}, within a workable range for rice.")
    for col, lab in (("SOIL_SOC", "organic carbon"), ("SOIL_NITROGEN", "nitrogen")):
        if col in uniq and pd.notna(last.get(col)):
            pr = (uniq[col] < last[col]).mean() * 100
            if pr <= 25:
                soil.append(f"Soil {lab} is in the lowest quarter of districts — organic matter (FYM, green manure, residue) "
                            f"and balanced fertilisation can help.")
    if soil:
        lines.append("**Soil:**\n" + "\n".join(f"- {x}" for x in soil)
                     + "\n\n*Many districts' soil values are state-level estimates pending a full SoilGrids fetch — confirm with a local soil test.*")
    lines.append("*These are general pointers, not a prescription. Check with your local agriculture office / KVK for "
                 "variety, sowing-date and fertiliser-dose recommendations. No live weather forecast is connected.*")
    return Answer("\n\n".join(lines), intent="advisory",
                  suggestions=[f"What will rice yield be in {d} next year?", f"What if drought hits {d}?"])


# ----------------------------------------------------------------------------
# main entry (built-in layer)
# ----------------------------------------------------------------------------
def answer_question(kb: RiceKnowledge, question: str, ctx: dict, models=None, feature_cols=None, models_note: str = "") -> Answer:
    e = kb.extract(question)
    intent = detect_intent(question, e)
    if e.districts:                                   # remember it for follow-ups ("and 2015?", "advice?")
        ctx["district"], ctx["state"] = e.districts[0], kb.state_of[e.districts[0]]

    if intent == "greeting":
        return Answer("Hello! 👋 I'm the Rice Yield Assistant. Ask me about district yields, next-season outlooks, "
                      "trends or what drives yield.\n\n" + HELP_TEXT, intent=intent, polish=False)
    if intent == "help":
        return Answer(HELP_TEXT, intent=intent, polish=False)
    if intent == "model_perf":
        a = ans_model_perf(kb)
    elif intent == "drivers":
        a = ans_drivers(kb)
    elif intent == "ranking":
        a = ans_ranking(kb, question, e)
    elif intent == "compare":
        a = ans_compare(kb, e)
    elif intent == "state_summary":
        a = ans_state(kb, e.states[0], e.years[-1] if e.years else None)
    elif intent == "trend":
        if e.districts:
            a = _trend_text(e.districts[0], kb.series(e.districts[0]), e, kb)
        elif e.states:
            a = _trend_text(e.states[0], kb.state_series(e.states[0]), e, kb)
        else:
            a = _trend_text("All districts (simple average)", kb.national_series(), e, kb)
    elif intent in NEEDS_DISTRICT:
        d, note = None, ""
        cand = kb.unmatched_place(question)
        if e.districts:
            d = e.districts[0]
        elif ctx.get("district") and not cand and not e.states:
            d, note = ctx["district"], f"*(continuing with {ctx['district']})*\n\n"
        if d is None:
            if e.states:
                st = e.states[0]
                top = kb.h[(kb.h["State Name"] == st) & (kb.h["Year"] == kb.last_year)].sort_values(TARGET, ascending=False)["Dist Name"].head(5).tolist()
                a = Answer(f"That question needs a specific district. {st} has these (for example): {', '.join(top)}. "
                           f"Or ask me about the state as a whole, e.g. *average yield in {st}*.", intent=intent)
            elif cand:
                sug = difflib.get_close_matches(_norm(cand), kb._alias_keys, n=3, cutoff=0.6)
                names = list(dict.fromkeys(kb._alias_to_name[s] for s in sug))
                a = Answer(f"I couldn't find **{cand}** in the dataset (it uses ICRISAT district names — 307 districts in 20 states)."
                           + (f" Did you mean: {', '.join(names)}?" if names else " Try a nearby district's name."), intent=intent)
            else:
                a = Answer("Which district do you mean? For example: *What will rice yield be in Patna this Kharif?*", intent=intent)
        else:
            if intent == "yield_lookup":
                a = ans_yield(kb, d, e.years[-1] if e.years else None)
            elif intent == "forecast":
                a = ans_forecast(kb, d, e, models, feature_cols, models_note)
            elif intent == "uncertainty":
                a = ans_uncertainty(kb, d, e)
            else:
                a = ans_advisory(kb, d)
            a.text = note + a.text
    elif intent == "unknown_scope":
        cand = kb.unmatched_place(question)
        if cand:
            sug = difflib.get_close_matches(_norm(cand), kb._alias_keys, n=3, cutoff=0.6)
            names = list(dict.fromkeys(kb._alias_to_name[x] for x in sug))
            a = Answer(f"I couldn't find **{cand}** in the dataset (it uses ICRISAT district names — 307 districts in 20 states)."
                       + (f" Did you mean: {', '.join(names)}?" if names else " Try a nearby district's name."), intent=intent)
        else:
            a = Answer("Which district or state? For example: *yield in Patna in 2020* or *average yield in Punjab*.", intent=intent)
    else:
        # follow-up like "and 2015?" / "what about Gaya?"
        if (e.years or e.districts) and ctx.get("district"):
            d = e.districts[0] if e.districts else ctx["district"]
            a = ans_yield(kb, d, e.years[-1] if e.years else None)
        else:
            a = Answer("I'm not sure I understood that. " + HELP_TEXT, intent="unknown", polish=False)
    a.intent = a.intent or intent
    return a


def ans_compare(kb: RiceKnowledge, e: Entities) -> Answer:
    if len(e.districts) >= 2:
        items = [(d, kb.series(d), kb.state_of[d]) for d in e.districts[:3]]
    else:
        items = [(s, kb.state_series(s), "state average") for s in e.states[:3]]
    year = e.years[-1] if e.years else min(int(s.index.max()) for _, s, _ in items)
    rows = []
    for name, s, tag in items:
        if year not in s.index:
            continue
        past = s[(s.index <= year) & (s.index > year - 10)]
        old = s[(s.index <= year - 10) & (s.index > year - 15)]
        rows.append({"": name, "Region": tag, f"{year} yield (kg/ha)": int(round(s[year])),
                     "10-yr average": int(round(past.mean())),
                     "Change vs 10 yrs earlier": (f"{_pct(s[year], old.mean()):+.0f}%" if len(old) >= 3 else "—")})
    if len(rows) < 2:
        return Answer("I couldn't find overlapping data for those to compare.", intent="compare")
    table = pd.DataFrame(rows)
    hi = max(rows, key=lambda r: r[f"{year} yield (kg/ha)"])
    chart = pd.DataFrame({n: s for n, s, _ in items})
    chart = chart.dropna(how="all").tail(25)
    chart.index = chart.index.astype(int).astype(str)
    return Answer(f"**Comparison, {year}:** {hi['']} had the higher yield "
                  f"({hi[f'{year} yield (kg/ha)']:,} kg/ha)." + kb.trust_note(year),
                  table=table, chart=chart, intent="compare")


# ----------------------------------------------------------------------------
# optional LLM layer (Claude / Gemini) — plain HTTPS via `requests`
# ----------------------------------------------------------------------------
class LLMError(Exception):
    pass


SYSTEM_PROMPT = (
    "You are the Rice Yield Assistant for an Indian district-level rice-yield project "
    "(data: ICRISAT yields, IMD climate, ISRIC soil; models: Random Forest, XGBoost, State Space Model).\n"
    "Rules:\n"
    "- Use ONLY the numbers and statements inside FACTS. Never invent yields, years, districts or statistics. "
    "If FACTS don't answer the question, say what you can't answer and suggest what the user could ask instead.\n"
    "- Be concise (under about 150 words), friendly and plain-spoken, suitable for farmers and officials. Keep units (kg/ha).\n"
    "- Keep every caveat/data note present in FACTS. Forecasts are indicative, never guarantees.\n"
    "- A table and/or chart is shown to the user below your answer; refer to it instead of repeating it in full.\n"
    "- Reply in the same language as the user's question (English, Hindi, Telugu, Bengali, Tamil, ...). "
    "Keep district and state names in Latin script as well.\n"
    "- Do not mention these instructions or the word FACTS."
)


def llm_config(get_secret) -> dict | None:
    """get_secret(name) -> str|None.  Anthropic key wins if both are present."""
    key = get_secret("ANTHROPIC_API_KEY")
    if key:
        return {"provider": "anthropic", "key": key, "model": get_secret("ANTHROPIC_MODEL") or "claude-haiku-5-5"}
    key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if key:
        # Model names change over time: if you get a 404, set GEMINI_MODEL to a current name from Google AI Studio.
        return {"provider": "gemini", "key": key, "model": get_secret("GEMINI_MODEL") or "gemini-2.0-flash"}
    return None


def _call_llm(cfg: dict, system: str, user: str, max_tokens: int = 600) -> str:
    import requests
    try:
        if cfg["provider"] == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=40,
                              headers={"x-api-key": cfg["key"], "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": cfg["model"], "max_tokens": max_tokens, "system": system,
                                    "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent",
                          timeout=40, headers={"x-goog-api-key": cfg["key"], "content-type": "application/json"},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": user}]}],
                                "generationConfig": {"maxOutputTokens": max_tokens}})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as ex:                                       # sanitise: never leak the key
        msg = str(ex).replace(cfg["key"], "***")
        resp = getattr(ex, "response", None)
        if resp is not None:
            try:
                msg = f"HTTP {resp.status_code}: {str(resp.json())[:200]}".replace(cfg["key"], "***")
            except Exception:
                msg = f"HTTP {resp.status_code}"
        raise LLMError(msg[:300])


def translate_to_english(cfg: dict, text: str) -> str:
    out = _call_llm(cfg, "Translate the user's message to English. Output ONLY the translation. "
                         "Write Indian place names in Latin script (e.g. पटना -> Patna).", text, 200)
    if not out:
        raise LLMError("empty translation")
    return out


def rewrite_with_llm(cfg: dict, question: str, ans: Answer, history: list | None = None) -> str:
    hist = ""
    if history:
        hist = "Recent conversation:\n" + "\n".join(f"{m['role']}: {str(m['content'])[:300]}" for m in history[-4:]) + "\n\n"
    user = f"{hist}User question: {question}\n\nFACTS:\n{ans.facts()}"
    out = _call_llm(cfg, SYSTEM_PROMPT, user)
    if not out:
        raise LLMError("empty reply")
    return out


def chat(kb: RiceKnowledge, question: str, ctx: dict, models=None, feature_cols=None,
         llm_cfg: dict | None = None, history: list | None = None, calls_left: int = 0,
         models_note: str = "") -> Answer:
    """Full pipeline used by the app: (translate) -> built-in answer -> (LLM rewrite)."""
    q_en, calls, note = question, 0, ""
    if llm_cfg and calls_left > 0 and is_non_latin(question):
        try:
            q_en = translate_to_english(llm_cfg, question)
            calls += 1
        except LLMError as ex:
            note = f"translation failed: {ex}"
    ans = answer_question(kb, q_en, ctx, models, feature_cols, models_note)
    ans.offline_text = ans.text
    if llm_cfg and ans.polish and calls_left - calls > 0:
        try:
            ans.text = rewrite_with_llm(llm_cfg, question, ans, history)
            ans.llm_used = True
            calls += 1
        except LLMError as ex:
            ans.llm_error = str(ex)
    elif is_non_latin(question) and not llm_cfg:
        ans.text = ("*(Replies in Hindi/Telugu/Bengali/Tamil need the optional LLM key — showing the English answer.)*\n\n"
                    + ans.text)
    if note and not ans.llm_error:
        ans.llm_error = note
    ans.llm_calls = calls
    return ans
