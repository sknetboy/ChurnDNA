from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def _recency_frequency_engagement(events_df: pd.DataFrame) -> pd.DataFrame:
    now = events_df["event_ts"].max()
    grouped = events_df.groupby("user_id")
    agg = grouped.agg(
        last_event=("event_ts", "max"),
        frequency=("event_type", "count"),
        avg_session_duration=("session_duration", "mean"),
        avg_pages_session=("pages_per_session", "mean"),
        adoption_score=("feature_adoption_rate", "mean"),
    )
    agg["recency_days"] = (now - agg["last_event"]).dt.days
    agg["engagement_score"] = agg["avg_session_duration"] * agg["avg_pages_session"]
    return agg.drop(columns=["last_event"]).reset_index()


def _digital_fatigue(events_df: pd.DataFrame) -> pd.DataFrame:
    now = events_df["event_ts"].max()
    out = []
    for uid, user_df in events_df.groupby("user_id"):
        windows = {}
        for days in [7, 14, 30]:
            cutoff = now - pd.Timedelta(days=days)
            recent = user_df[user_df["event_ts"] >= cutoff]
            windows[f"events_{days}d"] = len(recent)
            windows[f"engagement_{days}d"] = float((recent["session_duration"] * recent["pages_per_session"]).mean() if len(recent) else 0)
        fatigue = 0.0
        if windows["engagement_30d"] > 0:
            fatigue = 1 - (windows["engagement_7d"] / windows["engagement_30d"])
        out.append({"user_id": uid, "digital_fatigue_score": float(np.clip(fatigue, 0, 1)), **windows})
    return pd.DataFrame(out)


def _event_embeddings(events_df: pd.DataFrame, vector_size: int = 16) -> pd.DataFrame:
    sessions = events_df.sort_values("event_ts").groupby("user_id")["event_type"].apply(list)
    model = Word2Vec(sentences=sessions.tolist(), vector_size=vector_size, window=4, min_count=1, workers=1, epochs=30)

    rows = []
    for uid, seq in sessions.items():
        vecs = [model.wv[token] for token in seq]
        embedding = np.mean(vecs, axis=0)
        row = {"user_id": uid}
        for i, value in enumerate(embedding):
            row[f"w2v_{i}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table(events_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    rfe = _recency_frequency_engagement(events_df)
    fatigue = _digital_fatigue(events_df)
    embeddings = _event_embeddings(events_df)

    features = rfe.merge(fatigue, on="user_id", how="left").merge(embeddings, on="user_id", how="left")
    return features.merge(labels_df, on="user_id", how="left")
