import re
from difflib import get_close_matches
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Text utilities
# -----------------------------
def normalize_title(s: str) -> str:
    """Lowercase, trim, drop trailing (YYYY)."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s*\(\d{4}\)$", "", s)
    return s

def move_trailing_article(title: str) -> str:
    """
    'NeverEnding Story, The' -> 'The NeverEnding Story'
    """
    if not isinstance(title, str):
        return ""
    t = re.sub(r"\s*\(\d{4}\)$", "", title)                           # remove (YYYY)
    t = re.sub(r"^(.*),\s*(The|A|An)$", r"\2 \1", t, flags=re.I)      # move , The/A/An to front
    t = re.sub(r"\s+", " ", t).strip()
    return t

def franchise_key(s: str) -> str:
    """
    Normalize a title to a franchise key (to detect sequels, parts, numerals).
    Used for excluding or penalizing same-franchise recommendations.
    """
    s = (s or "").lower()
    s = re.sub(r"\s*\(\d{4}\)$", "", s)                                # drop year
    s = re.sub(r",\s*(the|a|an)$", r" \1", s)                          # move article
    s = re.sub(r"\d+", " ", s)                                         # digits
    s = re.sub(r"\b(part|episode|chapter)\b", " ", s, flags=re.I)      # sequel words
    s = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", " ", s, flags=re.I)  # roman numerals
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Main recommender
# -----------------------------
class ContentRecommender:
    """
    Content-based recommender using TF-IDF over genres (dominant) and optional
    lightly-weighted title tokens. Includes franchise handling and an MMR option.
    """

    def __init__(self, title_weight: float = 0.15, title_min_df: int = 3, use_titles: bool = True):
        self.df: Optional[pd.DataFrame] = None
        self.use_titles = use_titles
        self.title_weight = float(title_weight)
        self.title_min_df = int(title_min_df)

        # Vectorizers / matrices
        self.v_gen: Optional[TfidfVectorizer] = None
        self.v_ttl: Optional[TfidfVectorizer] = None
        self.X_gen = None
        self.X_ttl = None
        self.X = None
        self.cos_sim = None

        # Lookups
        self.title_to_idx: Optional[dict] = None

    # ---------- fitting / preparation ----------
    def _prepare_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        # clean titles
        df["clean_title"] = df["title"].map(move_trailing_article)

        # clean genres
        df["genres"] = df["genres"].replace("(no genres listed)", "")
        df["genres_clean"] = df["genres"].fillna("").str.replace("|", " ", regex=False)

        # softer title tokens for modeling (strip sequel noise)
        df["title_for_model"] = (
            df["clean_title"]
              .str.replace(r"\d+", " ", regex=True)
              .str.replace(r"\b(part|episode|chapter)\b", " ", regex=True, case=False)
              .str.replace(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", " ", regex=True, case=False)
              .str.replace(r"[^A-Za-z ]", " ", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )

        # franchise key for exclusion/penalty
        df["_fr_key"] = df["clean_title"].map(franchise_key)

        # title lookup
        self.title_to_idx = {normalize_title(t): i for i, t in enumerate(df["clean_title"].fillna(""))}
        return df.reset_index(drop=True)

    def fit_from_movies_csv(self, path: str, use_titles: Optional[bool] = None):
        if use_titles is not None:
            self.use_titles = use_titles

        raw = pd.read_csv(path)
        self.df = self._prepare_df(raw)

        # vectorizing genres (dominant signal)
        self.v_gen = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
        self.X_gen = self.v_gen.fit_transform(self.df["genres_clean"])

        if self.use_titles:
            # Lightly-weighted title signal
            self.v_ttl = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                lowercase=True,
                stop_words="english",
                min_df=max(1, self.title_min_df),
            )
            self.X_ttl = self.v_ttl.fit_transform(self.df["title_for_model"])
            self.X = hstack([self.X_gen * 1.0, self.X_ttl * float(self.title_weight)])
        else:
            self.X = self.X_gen

        self.cos_sim = cosine_similarity(self.X)
        return self

    # ---------- core API ----------
    def _not_found(self, title: str) -> str:
        guess = get_close_matches(normalize_title(title), list(self.title_to_idx.keys()), n=1, cutoff=0.6)
        return f"Movie '{title}' not found." + (f" Did you mean '{guess[0]}'?" if guess else "")

    def _indices_sorted(self, scores: np.ndarray, exclude: Optional[List[int]] = None) -> np.ndarray:
        order = np.argsort(scores)[::-1]
        if exclude:
            mask = np.ones_like(order, dtype=bool)
            for ex in exclude:
                mask &= (order != ex)
            order = order[mask]
        return order

    def recommend(
        self,
        title: str,
        k: int = 10,
        include_scores: bool = True,
        exclude_franchise: bool = False,
        franchise_penalty: float = 0.0,
    ) -> Union[pd.DataFrame, str]:
        """Top-k by cosine similarity with optional franchise filtering/penalty."""
        if self.df is None or self.cos_sim is None or self.title_to_idx is None:
            raise RuntimeError("Model not fitted. Call fit_from_movies_csv(path) first.")

        key = normalize_title(title)
        if key not in self.title_to_idx:
            return self._not_found(title)

        i = self.title_to_idx[key]
        scores = self.cos_sim[i].copy()

        # handle same-franchise items
        cand_idx = np.arange(len(self.df))
        cand_idx = cand_idx[cand_idx != i]

        if exclude_franchise or franchise_penalty > 0:
            q_fr = self.df.loc[i, "_fr_key"]
            same = (self.df["_fr_key"] == q_fr).values
            same[i] = False
            if exclude_franchise:
                cand_idx = cand_idx[~same[cand_idx]]
            elif franchise_penalty > 0:
                scores[same] -= float(franchise_penalty)

        order = cand_idx[np.argsort(scores[cand_idx])[::-1]]
        top = order[:k]

        out = self.df.loc[top, ["clean_title", "genres_clean"]].copy()
        out.rename(columns={"clean_title": "title", "genres_clean": "genres"}, inplace=True)
        if include_scores:
            out.insert(1, "similarity", scores[top].round(3))
        return out.reset_index(drop=True)

    def recommend_mmr(
        self,
        title: str,
        k: int = 10,
        lambda_val: float = 0.8,
        pool_size: int = 200,
    ) -> Union[pd.DataFrame, str]:
        """
        Maximal Marginal Relevance: re-ranks to improve diversity.
        lambda_val ∈ (0,1): higher = more emphasis on similarity, lower = more diversity.
        """
        if self.df is None or self.cos_sim is None or self.title_to_idx is None:
            raise RuntimeError("Model not fitted. Call fit_from_movies_csv(path) first.")

        key = normalize_title(title)
        if key not in self.title_to_idx:
            return self._not_found(title)

        i = self.title_to_idx[key]
        sim_to_query = self.cos_sim[i].copy()

        # candidate pool by similarity
        base = np.argsort(sim_to_query)[::-1]
        base = [j for j in base if j != i][:pool_size]

        selected: List[int] = []
        if base:
            selected.append(base[0])

        while len(selected) < min(k, len(base)):
            mmr_scores = []
            for j in base:
                if j in selected:
                    continue
                diversity = max(self.cos_sim[j, selected]) if selected else 0.0
                mmr = lambda_val * sim_to_query[j] - (1 - lambda_val) * diversity
                mmr_scores.append((mmr, j))
            j_star = max(mmr_scores)[1]
            selected.append(j_star)

        out = self.df.loc[selected, ["clean_title", "genres_clean"]].copy()
        out.rename(columns={"clean_title": "title", "genres_clean": "genres"}, inplace=True)
        out.insert(1, "similarity", sim_to_query[selected].round(3))
        return out.reset_index(drop=True)

    # ---------- convenience ----------
    def titles(self) -> List[str]:
        if self.df is None:
            return []
        return self.df["clean_title"].dropna().tolist()