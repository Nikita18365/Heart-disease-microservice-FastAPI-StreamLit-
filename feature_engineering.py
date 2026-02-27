from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from itertools import combinations

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Ensure all columns exist; create missing with NaN.
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _safe_float32(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]): # only float!
            df[c] = df[c].astype(np.float32)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = [
              "Age", "Cholesterol", "BP", "Number of vessels fluro", "Thallium",
              "Chest pain type", "Max HR", "ST depression", "Slope of ST",
              "Exercise angina"
             ]
    df = _ensure_columns(df, needed)

    df["age_cholesterol"]       = df["Age"] * df["Cholesterol"]
    df["bp_age"]                = df["BP"] * df["Age"]
    df["vessels_thallium"]      = df["Number of vessels fluro"] * df["Thallium"]
    df["chest_pain_vessels"]    = df["Chest pain type"] * df["Number of vessels fluro"]
    df["hr_age_ratio"]          = df["Max HR"] / (df["Age"] + 1.0)
    df["cholesterol_age_ratio"] = df["Cholesterol"] / (df["Age"] + 1.0)
    df["bp_ratio"]              = df["BP"] / (df["Age"] + 1.0)
    df["heart_risk_score"]      = (df["Thallium"] * 3.0 + df["Chest pain type"] * 2.0 + df["Number of vessels fluro"] * 2.0)
    df["thallium_sq"]           = df["Thallium"] ** 2.0
    df["chest_pain_sq"]         = df["Chest pain type"] ** 2.0
    df["st_slope_interaction"]  = df["ST depression"] * df["Slope of ST"]
    df["exercise_st"]           = df["Exercise angina"] * df["ST depression"]
    return df

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# FEArtifacts is a container where we store everything we “learned” during the preparation stage (fit), so that we can simply apply it later (transform)
@dataclass
class FEArtifacts:
    # Columns
    base_numeric_cols: List[str]                   # default df_train columns
    quantitative_cols: List[str]                   # the numerical attributes (which will go to pd.qcut)
    cross_pairs: List[Tuple[str, str]]             # cross feature [(quantive_col, from base_numeric_cols),(..., ...),...]
    cat_cols_to_label: List[str]                   # LabelEncoder() features

    # original stats (values)
    # Any because the values can be int/float (for example, ST depression — float)
    origin_mean_maps: Dict[str, Dict[Any, float]]  # average value of a feature by target in origin
    origin_count_maps: Dict[str, Dict[Any, float]] # number of feature values by target in origin
    origin_global_mean: float                      # df_origin[target].mean()

    # Qcut edges fixing
    qcut_edges: Dict[str, np.ndarray]              # {feature: array[1., 2., 3.,], ...} 

    # label/frequency encoding values
    label_maps: Dict[str, Dict[str, int]]          # {"quantitative_cols": {"(29.0, 41.0]": LE code_0, "(41.0, 49.0]": LE code_0, "__NA__": LE code_999}}
    freq_maps: Dict[str, Dict[int, float]]         # {"quantitative_cols": {0: freq, 1: 0.22, 999: 0.0001}}
    
    # Final features
    features: List[str]
    
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

def fit_artifacts(df_train: pd.DataFrame,
                  df_origin: pd.DataFrame,
                  target: str,
                  quantitative_cols: List[str],
                  n_bins: int = 20,
                  cross_pairs: Optional[List[Tuple[str, str]]] = None,
                  freq_only_for_quant: bool = True,
                 ) -> FEArtifacts:

    # TRAIN-TIME ONLY:
    #- origin stats: mean/count maps from df_origin grouped by numeric values
    #- qcut edges: learned on df_train
    #- label_maps: for Quant_* features
    #- freq_maps: by default only for Quant_*
    #- features: final list in order expected by the model

    df_train = df_train.copy()
    df_origin = df_origin.copy()

    base_numeric_cols = df_train.drop(columns = [target], errors = "ignore").select_dtypes(include=np.number).columns.tolist()

    # 1. Counting statistics variables by origin mean/count maps
    origin_global_mean = float(df_origin[target].mean())
    origin_mean_maps:     Dict[str, Dict[Any, float]] = {}
    origin_count_maps:    Dict[str, Dict[Any, float]] = {}
    for col in base_numeric_cols:
        if col not in df_origin.columns:
            continue
        agg = df_origin.groupby(col)[target].agg(["mean", "count"])
        origin_mean_maps[col] = agg["mean"].to_dict()
        origin_count_maps[col] = agg["count"].to_dict()

    # 2. Counting the bin array "qcut" edges on train
    qcut_edges: Dict[str, np.ndarray] = {}
    for col in quantitative_cols:
        if col not in df_train.columns:
            continue
        # We clean inf, delete NaN
        s = df_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s.nunique() < 2:
            continue
        # retbins=True - returns borders
        _, edges = pd.qcut(s, q = n_bins, duplicates = "drop", retbins = True)
        qcut_edges[col] = np.asarray(edges, dtype = np.float64)
    
    # 3. Adding to the List - Quant_* columns and cross cols
    quant_bin_cols = [f"Quant_{c}" for c in quantitative_cols]
    if cross_pairs is None:
        cross_pairs: List[str] = []
        cross_pairs += list(combinations(quant_bin_cols, 2))
        cross_pairs += [
                        ("Quant_Age", "Chest pain type"),
                        ("Quant_Age", "Number of vessels fluro"),
                        ("Quant_ST depression", "Slope of ST"),
                        ("Quant_Max HR", "Exercise angina"),
                        ("Quant_Max HR", "Chest pain type"),
                        ("Quant_ST depression", "Number of vessels fluro"),
                        ("Quant_Cholesterol", "Sex"),
                        ("Quant_BP", "Sex"),
                       ]
    cross_cols = [f"{a}__X__{b}" for a, b in cross_pairs]
    cat_cols_to_label = quant_bin_cols + cross_cols

    # 4. Creating a temporary df to extract categories for Quant_*
    tmp = df_train.drop(columns = [target], errors = "ignore").copy()
    tmp = add_interactions(tmp)
    for col in quantitative_cols:
        qname = f"Quant_{col}"
        if col in qcut_edges:
            tmp[qname] = pd.cut(tmp[col], bins = qcut_edges[col], include_lowest = True).astype(str).fillna("__NA__")
        else:
            tmp[qname] = "__NA__"

    # 5. Cross cols (strings)
    for a, b in cross_pairs:
        cname = f"{a}__X__{b}"
        tmp = _ensure_columns(tmp, [a, b])
        tmp[cname] = tmp[a].astype(str).fillna("__NA__") + "__X__" + tmp[b].astype(str).fillna("__NA__")

    # 6. Label and freq maps
    label_maps: Dict[str, Dict[str, int]]  = {}
    freq_maps: Dict[str, Dict[int, float]] = {}

    for col in cat_cols_to_label:
        vals = tmp[col].astype(str).fillna("__NA__")
        uniq = sorted(pd.unique(vals).tolist())
        label_maps[col] = {v: i for i, v in enumerate(uniq)}

    for col in quant_bin_cols:
        codes = tmp[col].map(label_maps[col]).fillna(-1).astype(int)
        freq = codes.value_counts(normalize=True).to_dict()
        freq_maps[col] = {int(k): float(v) for k, v in freq.items()}

    # 7. Build final features list
    features: List[str] = df_train.drop(columns = [target], errors = "ignore").columns.tolist()
    # Interactions (12 features)
    interaction_cols = [
                        "age_cholesterol", "bp_age", "vessels_thallium", "chest_pain_vessels",
                        "hr_age_ratio", "cholesterol_age_ratio", "bp_ratio", "heart_risk_score",
                        "thallium_sq", "chest_pain_sq", "st_slope_interaction", "exercise_st"
                       ]

    # Interaction features
    features.extend(interaction_cols)
    
    # Origin stats features
    for col in base_numeric_cols:
        if col in origin_mean_maps:
            features.append(f"origin_mean_{col}")
            features.append(f"origin_count_{col}")

    # Quant bins features
    for col in quantitative_cols:
        qname = f"Quant_{col}"
        features.append(qname)

    # Cross features
    features.extend(cross_cols)

    # Freq features
    for col in quant_bin_cols:
        features.append(f"Freq_{col}")

    return FEArtifacts(
                       base_numeric_cols = base_numeric_cols,
                       quantitative_cols = quantitative_cols,
                       cross_pairs = cross_pairs,
                       cat_cols_to_label = cat_cols_to_label,
                       
                       origin_mean_maps = origin_mean_maps,
                       origin_count_maps = origin_count_maps,
                       origin_global_mean = origin_global_mean,
                       
                       qcut_edges = qcut_edges,
        
                       label_maps = label_maps,
                       freq_maps = freq_maps,
                       
                       features = features
                      )

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

def build_features(df_raw: pd.DataFrame, artifacts: FEArtifacts) -> pd.DataFrame:
    # INFERENCE-TIME:
    # Apply exactly the same FE as training.
    df = df_raw.copy()

    required = list(set(artifacts.base_numeric_cols + artifacts.quantitative_cols))
    df = _ensure_columns(df, required)

    # 1. interactions
    df = add_interactions(df)

    # 2. origin stats
    for col in artifacts.base_numeric_cols:
        if col not in artifacts.origin_mean_maps:
            continue
        df[f"origin_mean_{col}"]  = df[col].map(artifacts.origin_mean_maps[col]).fillna(artifacts.origin_global_mean).astype(np.float32)
        df[f"origin_count_{col}"] = df[col].map(artifacts.origin_count_maps[col]).fillna(0).astype(np.float32)

    # 3. quant bins
    for col in artifacts.quantitative_cols:
        qname = f"Quant_{col}"
        if col in artifacts.qcut_edges:
            df[qname] = pd.cut(df[col].replace([np.inf, -np.inf], np.nan),
                               bins = artifacts.qcut_edges[col],
                               include_lowest = True
                              ).astype(str).fillna("__NA__")
        else:
            df[qname] = "__NA__"

    # 4. Cross features
    # Cross cols (strings) -> will be label-encoded later
    for a, b in artifacts.cross_pairs:
        cname = f"{a}__X__{b}"
        df = _ensure_columns(df, [a, b])
        df[cname] = df[a].astype(str).fillna("__NA__") + "__X__" + df[b].astype(str).fillna("__NA__")

    # 5. Label + freq
    for col in artifacts.cat_cols_to_label:
        df[col] = df[col].astype(str).fillna("__NA__")
        mapping = artifacts.label_maps.get(col, {})
        df[col] = df[col].map(mapping).fillna(-1).astype(np.int32)
        # freq only for Quant_
        if col.startswith("Quant_") and "__X__" not in col:
            df[f"Freq_{col}"] = df[col].map(artifacts.freq_maps.get(col, {})).fillna(0.0).astype(np.float32)

    df = _safe_float32(_ensure_columns(df, artifacts.features))
    return df[artifacts.features].copy()
