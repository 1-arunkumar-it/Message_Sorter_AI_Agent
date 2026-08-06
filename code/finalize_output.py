"""Create the deterministic final submission from the public challenge data.

Run from the repository root: ``python code/finalize_output.py``.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
FIELDS = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]
STOP = {
    "the", "and", "for", "with", "this", "that", "your", "you", "are",
    "from", "have", "will", "now", "today", "please", "here", "they",
    "into", "just", "can", "all", "before", "after", "has", "was", "not",
    "but", "our", "their", "message", "reply", "check", "details", "link",
}


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{3,}", text.lower()) if word not in STOP}


def truthy(value: str) -> bool:
    return value.strip() in {"1", "true", "True"}


def category(text: str, row: dict[str, str]) -> str:
    t = text.lower()
    if any(x in t for x in ("otp", "pin", "password", "verification code", "login code", "6 digit", "bank details", "account number")) and any(
        x in t for x in ("link", "share", "send", "confirm", "verify", "verification", "scan", "expire", "blocked", "restricted")
    ):
        return "scam"
    if any(x in t for x in ("scan this qr", "scan the qr", "scan and pay")) and any(x in t for x in ("pay", "clearance", "fee", "amount", "charge")):
        return "scam"
    if "loan approved" in t and "processing fee" in t:
        return "scam"
    if any(x in t for x in ("amazonpay-delivery", "account-login", "account-help", "bit.ly", "pay-check")):
        return "scam"
    if any(x in t for x in ("share with", "forward", "blessing", "good morning", "good luck", "health secret")) and not any(x in t for x in ("tanker", "fire alarm", "gate closes")):
        return "forward"
    if any(x in t for x in ("offer", "discount", "sale", "shop", "promotion", "benefit", "recliner", "kurta", "plots", "travel deal")) and "offer letter" not in t:
        return "promotion"
    if any(x in t for x in ("delivery", "pickup", "order", "ride update", "route status", "appointment", "prescription", "field trip", "internship", "potluck", "circular", "fire alarm")):
        return "event" if row["conversation_type"] == "group" else "business_update"
    if any(x in t for x in ("payment", "card statement", "amount due", "refund", "payout")):
        return "payment"
    if any(x in t for x in ("urgent", "today", "minutes", "call me", "moved to", "failing", "rollback", "gate", "tanker", "lift", "fire alarm", "maintenance")):
        return "urgent"
    if any(x in t for x in ("good morning", "bonjour", "reached home", "did you eat")):
        return "greeting" if "good morning" in t else "personal"
    return "personal" if row["conversation_type"] == "personal" else "unknown"


def find_evidence(row: dict[str, str], history: list[dict[str, str]], events: dict[str, dict[str, str]]) -> str:
    query = tokens(row["message_text"])
    candidates = []
    for old in history:
        score = len(query & tokens(old["message_text"]))
        if old["conversation_type"] == row["conversation_type"]:
            score += 2
        for key in ("group_id", "business_id", "sender_user_id"):
            if row[key] and row[key] == old[key]:
                score += 4
        event = events.get(old["message_id"], {})
        if truthy(event.get("message_opened", "")) or truthy(event.get("message_reported", "")):
            score += 1
        if score >= 7:
            candidates.append((score, old["message_id"]))
    candidates.sort(reverse=True)
    return ";".join(mid for _, mid in candidates[:2]) if candidates else "none"


def predict(row: dict[str, str], businesses: dict[str, dict[str, str]], relationships: dict[tuple[str, str], dict[str, str]], group_members: dict[tuple[str, str], dict[str, str]]) -> tuple[str, str, str, float]:
    text = row["message_text"] or ""
    low = text.lower()
    kind = category(text, row)
    conversation = row["conversation_type"]
    relationship = relationships.get((row["user_id"], row["business_id"]), {})
    group_member = group_members.get((row["group_id"], row["user_id"]), {})
    opted_out = bool(relationship) and (bool(relationship.get("promotions_opted_out_at")) or not truthy(relationship.get("allows_promotions", "0")))
    direct = f"@{row['user_id']}" in text or conversation == "personal"

    if kind == "scam":
        return "mute", "scam", "The message requests sensitive credentials or payment through an unsafe, high-pressure flow.", 0.96
    if kind == "forward":
        message_type = "greeting" if "good morning" in low or "blessing" in low else "forward"
        return "mute", message_type, "This is a repetitive forwarded chain or unsupported tip with no actionable value.", 0.90
    if kind == "promotion":
        if opted_out or int(row["forwarded_count"] or 0) >= 3:
            return "mute", "promotion", "This is promotional content that the user has opted out of or tends to dismiss.", 0.87
        return "digest", "promotion", "This is a legitimate low-priority offer that can be reviewed later.", 0.80

    # Untrusted payment/refund requests are phishing even without an explicit OTP.
    if kind == "payment" and any(x in low for x in ("release the amount", "reactivation fee", "clearance amount", "processing fee", "wallet details", "scan this qr", "send screenshot")):
        return "mute", "scam", "The payment request uses urgent pressure and an unsafe verification or payment instruction.", 0.94

    if conversation == "business":
        business = businesses.get(row["business_id"], {})
        verified = truthy(business.get("verified", ""))
        if not verified:
            if kind == "payment" or any(x in low for x in ("verify", "security", "payout", "refund", "access restricted")):
                return "mute", "scam", "An unverified business sender requests a financial or account-security action.", 0.91
            return "mute", "spam", "The business sender cannot be verified as a trusted source.", 0.84
        urgent_terms = ("today", "scheduled", "packed", "pickup", "route", "appointment", "payment update")
        if relationship and any(term in low for term in urgent_terms):
            return "notify", kind, "A verified business sent a time-sensitive update that matches the user's relationship.", 0.89
        return "digest", kind, "This verified business update is legitimate but does not require an immediate interruption.", 0.79

    urgent_terms = ("today", "tonight", "minutes", "by 6", "before 7", "before 5", "moved to", "call", "failing", "rollback", "tanker", "gate", "lift", "field trip", "internship", "appointment")
    if conversation == "personal":
        if any(term in low for term in ("nothing blocking", "nothing urgent", "no urgency", "no need to reply")):
            return "digest", "personal", "This direct message is safe but does not need to interrupt the user now.", 0.79
        if any(term in low for term in urgent_terms) and "no urgency" not in low and "nothing urgent" not in low:
            return "notify", "urgent", "A direct message needs a timely response or action from the user.", 0.86
        return "digest", kind, "This direct message is safe but does not need to interrupt the user now.", 0.79

    if any(term in low for term in urgent_terms) or direct:
        if group_member.get("group_muted_by_user") == "1" and not direct and "tanker" not in low and "gate" not in low:
            return "digest", kind, "Useful group information can wait because this group is muted by the user.", 0.76
        return "notify", kind if kind != "unknown" else "urgent", "A trusted group message contains a time-sensitive update or direct request.", 0.87
    return "digest", kind, "This is safe group information that can be reviewed in a later digest.", 0.79


def main() -> None:
    messages = read("messages.csv")
    history_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in read("message_history.csv"):
        history_by_user[item["user_id"]].append(item)
    events = {item["message_id"]: item for item in read("message_events.csv")}
    businesses = {item["business_id"]: item for item in read("business_accounts.csv")}
    relationships = {(item["user_id"], item["business_id"]): item for item in read("user_business_history.csv")}
    group_members = {(item["group_id"], item["user_id"]): item for item in read("group_members.csv")}

    predictions = []
    for row in messages:
        action, message_type, reason, confidence = predict(row, businesses, relationships, group_members)
        predictions.append({
            "message_id": row["message_id"], "action": action, "message_type": message_type,
            "reason": reason, "confidence": f"{confidence:.2f}",
            "evidence_message_ids": find_evidence(row, history_by_user[row["user_id"]], events),
        })
    for output in (ROOT / "output.csv", DATA / "output.csv"):
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to output.csv and dataset/output.csv")


if __name__ == "__main__":
    main()
