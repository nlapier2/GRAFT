
import re
import pandas as pd

def normalize_hgnc(x):
    """Minimal HGNC-like normalization: uppercase, strip whitespace.
    Replace with a proper HGNC mapper if available."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s == "":
        return s
    s = s.upper()
    s = s.replace("MIR-", "MIR")
    s = re.sub(r"\s+", "", s)
    return s
