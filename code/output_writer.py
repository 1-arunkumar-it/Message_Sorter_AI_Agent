import pandas as pd


class OutputWriter:

    def __init__(self):
        self.predictions = []

    def add(self, prediction):
        self.predictions.append({
            "message_id": prediction.message_id,
            "action": prediction.action,
            "message_type": prediction.message_type,
            "reason": prediction.reason,
            "confidence": prediction.confidence,
            "evidence_message_ids": prediction.evidence_message_ids
        })

    def write(self):
        output = pd.DataFrame(self.predictions)

        output.to_csv(
            "../dataset/output.csv",
            index=False
        )
