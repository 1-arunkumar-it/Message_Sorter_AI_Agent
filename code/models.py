from dataclasses import dataclass

import pandas as pd


@dataclass
class MessageContext:
    message: pd.Series
    user: pd.DataFrame
    group: pd.DataFrame | None
    business: pd.DataFrame | None
    history: pd.DataFrame
    events: pd.DataFrame
    business_history: pd.DataFrame
    notification_summary: pd.DataFrame
    media: str | None


@dataclass
class Prediction:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
