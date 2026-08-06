import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from models import Prediction


class AIReasoner:

    def __init__(self):
        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = "gpt-5.5"

    def predict(self, context):

        prompt = self.build_prompt(context)

        while True:

            try:

                response = self.client.responses.create(
                    model=self.model,
                    input=prompt
                )

                text = response.output_text
                break

            except RateLimitError as e:

                retry = self.get_retry_time(e)

                print("\n===================================")
                print("OpenAI Rate Limit")
                print("===================================")
                print(f"Retrying after {retry} seconds...\n")

                time.sleep(retry)

            except Exception as e:
                print(e)
                raise

        if text.startswith("```json"):
            text = text.replace("```json", "", 1)

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        if not text.startswith("{"):

            match = re.search(r"\{.*\}", text, re.DOTALL)

            if match:
                text = match.group()

        result = json.loads(text)

        return Prediction(
            message_id=context.message["message_id"],
            action=result["action"],
            message_type=result["message_type"],
            reason=result["reason"],
            confidence=float(result["confidence"]),
            evidence_message_ids=result["evidence_message_ids"]
        )

    def get_retry_time(self, error):

        retry = 60

        try:

            message = str(error)

            match = re.search(
                r"retry.*?([0-9]+)\s*seconds",
                message,
                re.IGNORECASE
            )

            if match:
                retry = int(match.group(1))

        except Exception:
            pass

        return retry

    def build_prompt(self, context):

        return f"""
You are an AI-powered Message Notification Router for WhatsApp.

Your task is to classify ONE incoming message.

You must consider:

- User profile
- Previous message history
- Group information
- Business information
- Notification habits
- Attached media description

Decide the following fields.

action:
- notify
- digest
- mute

message_type:
- personal
- urgent
- event
- payment
- business_update
- promotion
- greeting
- forward
- spam
- scam
- unknown

Guidelines:

- Notify only if the message deserves immediate attention.
- Digest if useful but not urgent.
- Mute spam, scams, repetitive promotions, unwanted forwards, or unsafe content.
- Use the user's history to personalize the decision.
- Confidence must be between 0 and 1.
- evidence_message_ids should contain matching historical message ids separated with ';'.
- If there is no useful evidence, return "none".

Current Message

MESSAGE ID:
{context.message["message_id"]}

Conversation Type:
{context.message["conversation_type"]}

Message:
{context.message["message_text"]}

Forwarded Count:
{context.message["forwarded_count"]}

Media Description:
{context.media}

User:
{context.user.to_dict("records")}

Group:
{None if context.group is None else context.group.to_dict("records")}

Business:
{None if context.business is None else context.business.to_dict("records")}

Previous Messages:
{context.history.tail(5).to_dict("records")}

Previous Events:
{context.events.tail(5).to_dict("records")}

Business History:
{context.business_history.to_dict("records")}

Daily Notification Summary:
{context.notification_summary.to_dict("records")}

Return ONLY valid JSON.

{{
    "action":"notify",
    "message_type":"urgent",
    "reason":"Short human-readable explanation.",
    "confidence":0.95,
    "evidence_message_ids":"none"
}}

Do not return markdown.
Do not explain your reasoning.
Return only the JSON object.
"""