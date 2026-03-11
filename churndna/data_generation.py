from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

EVENT_TYPES = ["login", "search", "click", "page_view", "purchase", "support_ticket"]


@dataclass
class GenerationConfig:
    n_users: int = 50_000
    months_history: int = 18
    seed: int = 42


def _base_user_table(config: GenerationConfig) -> pd.DataFrame:
    fake = Faker("es_ES")
    Faker.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    users = []
    for user_idx in range(config.n_users):
        users.append(
            {
                "user_id": f"U{user_idx:06d}",
                "country": fake.country_code(),
                "tenure_days": int(rng.integers(30, 900)),
                "plan": rng.choice(["free", "basic", "pro", "enterprise"], p=[0.35, 0.4, 0.2, 0.05]),
                "feature_adoption_rate": float(np.clip(rng.normal(0.5, 0.2), 0.01, 0.99)),
            }
        )
    return pd.DataFrame(users)


def generate_synthetic_events(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    """Generate event-level synthetic data for 50k users and 18 months by default."""
    rng = np.random.default_rng(config.seed)
    users = _base_user_table(config)
    horizon_days = int(config.months_history * 30)
    start_date = datetime.utcnow() - timedelta(days=horizon_days)

    records: list[dict] = []
    for user in users.itertuples(index=False):
        n_sessions = int(max(5, rng.poisson(40 + user.tenure_days / 50)))
        for _ in range(n_sessions):
            day_offset = int(rng.integers(0, horizon_days))
            session_start = start_date + timedelta(days=day_offset, hours=int(rng.integers(0, 24)))
            session_duration = float(max(1, rng.gamma(2.5, 5)))
            pages_per_session = int(max(1, rng.poisson(4 + user.feature_adoption_rate * 5)))
            event = rng.choice(EVENT_TYPES, p=[0.18, 0.25, 0.22, 0.25, 0.07, 0.03])
            records.append(
                {
                    "user_id": user.user_id,
                    "event_ts": session_start,
                    "event_type": event,
                    "session_duration": session_duration,
                    "pages_per_session": pages_per_session,
                    "days_since_last_visit": int(rng.integers(0, 21)),
                    "feature_adoption_rate": user.feature_adoption_rate,
                    "country": user.country,
                    "plan": user.plan,
                }
            )

    seed_df = pd.DataFrame(records)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(seed_df)
    synthetic_df = synthesizer.sample(num_rows=len(seed_df))
    synthetic_df["event_ts"] = pd.to_datetime(synthetic_df["event_ts"])
    return synthetic_df.sort_values("event_ts").reset_index(drop=True)


def build_churn_label(events_df: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """Label user churn in next horizon based on inactivity."""
    max_ts = events_df["event_ts"].max()
    last_user_event = events_df.groupby("user_id", as_index=False)["event_ts"].max()
    last_user_event["churn_30d"] = (max_ts - last_user_event["event_ts"]).dt.days > horizon_days
    return last_user_event[["user_id", "churn_30d"]]
