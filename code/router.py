from dataloader import DataLoader
from context_builder import ContextBuilder
from ai_reasoner import AIReasoner
from media_processor import MediaProcessor
from output_writer import OutputWriter


class NotificationRouter:

    def __init__(self):
        self.loader = DataLoader()

        self.media_processor = MediaProcessor(self.loader)

        self.context_builder = ContextBuilder(
            self.loader,
            self.media_processor
        )

        self.reasoner = AIReasoner()

        self.writer = OutputWriter()

    def run(self):

        self.loader.load()

        for _, message in self.loader.messages.iterrows():

            context = self.context_builder.build_context(message)

            prediction = self.reasoner.predict(context)

            self.writer.add(prediction)

        self.writer.write()