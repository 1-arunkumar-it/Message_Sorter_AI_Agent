import pandas as pd


class DataLoader:

    def __init__(self):
        self.messages = None
        self.users = None
        self.groups = None
        self.group_members = None
        self.business_accounts = None
        self.user_business_history = None
        self.message_history = None
        self.message_events = None
        self.images = None
        self.voice_notes = None
        self.daily_notification_summary = None
        self.output_template = None

    def load(self):
        self.messages = pd.read_csv("../dataset/messages.csv")
        self.users = pd.read_csv("../dataset/users.csv")
        self.groups = pd.read_csv("../dataset/groups.csv")
        self.group_members = pd.read_csv("../dataset/group_members.csv")
        self.business_accounts = pd.read_csv("../dataset/business_accounts.csv")
        self.user_business_history = pd.read_csv("../dataset/user_business_history.csv")
        self.message_history = pd.read_csv("../dataset/message_history.csv")
        self.message_events = pd.read_csv("../dataset/message_events.csv")
        self.images = pd.read_csv("../dataset/images.csv")
        self.voice_notes = pd.read_csv("../dataset/voice_notes.csv")
        self.daily_notification_summary = pd.read_csv("../dataset/daily_notification_summary.csv")
        self.output_template = pd.read_csv("../dataset/output.csv")
