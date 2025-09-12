from typing import List, Dict
import re
from rapidfuzz import fuzz
import math
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def parse_struct(text: str) -> Dict[str, str]:
    if text is None:
        return {"country_type": None, "state": None, "county": None}
    t = text.lower()

    if any(country in t for country in ["country: us", "country: usa", "country: united states", "country: u.s.", "united states"]):
        ctype = "US"
    elif "country: non-us" in t or "non us" in t or "non-us" in t:
        ctype = "Non-US"
    else:
        ctype = "UNKNOWN"
    m_state = re.search(r"state:\s*([a-z\-\.\s]+)", t)
    state = m_state.group(1).strip() if m_state else None
    m_county = re.search(r"county:\s*([a-z\-\.\s]+)", t)
    county = m_county.group(1).strip() if m_county else None
    return {"country_type": ctype, "state": state, "county": county}


def fuzzy_equal(a: str, b: str, threshold=85) -> int:
    if a is None and b is None:
        return 1
    if a is None or b is None:
        return 0
    a = a.lower().strip()
    b = b.lower().strip()
    return 1 if fuzz.ratio(a, b) >= threshold else 0



def compute_validation_metrics(pred_texts: List[str], gold_texts: List[str], threshold=85) -> Dict[str, float]:
    preds = [parse_struct(x) for x in pred_texts]
    golds = [parse_struct(x) for x in gold_texts]

    def safe_label(x, placeholder="UNKNOWN"):
        """
        Normalize labels for metric computation.
    
        Converts None, NaN, empty/whitespace, and common placeholders 
        like 'NA', 'N/A', 'null', 'none' into a consistent placeholder.
        """
        if x is None:
            return placeholder
    
        # Handle floats (e.g., np.nan, float('nan'))
        if isinstance(x, float):
            if math.isnan(x):
                return placeholder
            return str(x).strip()
    
        # Convert to string
        s = str(x).strip()
    
        # Empty string or whitespace
        if s == "":
            return placeholder
    
        # Common "null-like" tokens
        null_like = {"na", "n/a", "none", "null", "nan", "unk", "unknown"}
        if s.lower() in null_like:
            return placeholder

        return s

    # --- US vs non-US
    y_true_us = [1 if g["country_type"] == "US" else 0 for g in golds]
    y_pred_us = [1 if p["country_type"] == "US" else 0 for p in preds]

    us_f1 = f1_score(y_true_us, y_pred_us, average="weighted")
    us_acc = accuracy_score(y_true_us, y_pred_us)
    us_precision = precision_score(y_true_us, y_pred_us, average="weighted", zero_division=0)
    us_recall = recall_score(y_true_us, y_pred_us, average="weighted", zero_division=0)

    # --- State
    y_true_state = [safe_label(g["state"]) for g in golds]
    y_pred_state = [safe_label(p["state"]) for p in preds]

    state_acc = accuracy_score(y_true_state, y_pred_state)
    state_f1 = f1_score(y_true_state, y_pred_state, average="weighted", zero_division=0)
    state_precision = precision_score(y_true_state, y_pred_state, average="weighted", zero_division=0)
    state_recall = recall_score(y_true_state, y_pred_state, average="weighted", zero_division=0)

    # --- County
    y_true_county = [safe_label(g["county"]) for g in golds]
    y_pred_county = [safe_label(p["county"]) for p in preds]

    y_pred_county_clean = [
        g if fuzzy_equal(g, p, threshold) else p
        for g, p in zip(y_true_county, y_pred_county)
    ]

    county_acc = accuracy_score(y_true_county, y_pred_county_clean)
    county_f1 = f1_score(y_true_county, y_pred_county_clean, average="weighted", zero_division=0)
    county_precision = precision_score(y_true_county, y_pred_county_clean, average="weighted", zero_division=0)
    county_recall = recall_score(y_true_county, y_pred_county_clean, average="weighted", zero_division=0)

    return {
        "us_f1": us_f1, "us_acc": us_acc, "us_precision": us_precision, "us_recall": us_recall,
        "state_f1": state_f1, "state_acc": state_acc, "state_precision": state_precision, "state_recall": state_recall,
        "county_f1": county_f1, "county_acc": county_acc, "county_precision": county_precision, "county_recall": county_recall,
    }

