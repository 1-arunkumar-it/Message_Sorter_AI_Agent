
from models import MessageContext


class ContextBuilder:

    def __init__(self, loader, media_processor):
        self.loader = loader
        self.media_processor = media_processor

    def build_context(self, message):

        user = self.loader.users[
            self.loader.users["user_id"] == message["user_id"]
        ]

        group = None
        if message["conversation_type"] == "group":
            group = self.loader.groups[
                self.loader.groups["group_id"] == message["group_id"]
            ]

        business = None
        if message["conversation_type"] == "business":
            business = self.loader.business_accounts[
                self.loader.business_accounts["business_id"] == message["business_id"]
            ]

        history = self.loader.message_history[
            self.loader.message_history["user_id"] == message["user_id"]
        ]

        events = self.loader.message_events[
            self.loader.message_events["user_id"] == message["user_id"]
        ]

        business_history = self.loader.user_business_history[
            self.loader.user_business_history["user_id"] == message["user_id"]
        ]

        notification_summary = self.loader.daily_notification_summary[
            self.loader.daily_notification_summary["user_id"] == message["user_id"]
        ]

        media = self.media_processor.process(message)

        return MessageContext(
            message=message,
            user=user,
            group=group,
            business=business,
            history=history,
            events=events,
            business_history=business_history,
            notification_summary=notification_summary,
            media=media
        )
