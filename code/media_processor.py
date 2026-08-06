import os
import time

from dotenv import load_dotenv
import base64
from openai import OpenAI
from openai import RateLimitError


class MediaProcessor:

    def __init__(self, loader):
        load_dotenv()

        self.loader = loader

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )       

        self.model = "gpt-5.5"

    def process(self, message):

        media_type = message["media_type"]

        if not media_type or str(media_type) == "nan":
            return "No media."

        if media_type == "image":
            return self.process_image(message["media_id"])

        if media_type == "voice":
            return self.process_voice(message["media_id"])

        return "Unknown media."

    def process_image(self, media_id):

        row = self.loader.images[
            self.loader.images["image_id"] == media_id
        ]

        if row.empty:
            return "Image not found."

        image_path = "../dataset/" + row.iloc[0]["file_path"]

        while True:

            try:

                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(
                        image_file.read()
                    ).decode("utf-8")

                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Describe this image briefly for notification routing."
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_data}"
                                }
                            ]
                        }
                    ]
                )

                return response.output_text

            except RateLimitError:

                print("Rate limit reached. Waiting 60 seconds...")

                time.sleep(60)


            except Exception:
                raise
            



    def process_voice(self, media_id):

        row = self.loader.voice_notes[
            self.loader.voice_notes["voice_id"] == media_id
        ]

        if row.empty:
            return "Voice note not found."

        audio_path = "../dataset/" + row.iloc[0]["file_path"]


        while True:

            try:

                with open(audio_path, "rb") as audio_file:

                    transcript = self.client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=audio_file
                    )

                response = self.client.responses.create(
                    model=self.model,
                    input=f"""
        Transcribe and summarize the following voice note.

        Transcript:

        {transcript.text}

        Return only a short summary suitable for notification routing.
        """
                )
            
                return response.output_text

                

            except RateLimitError:

                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)

            except Exception:
                raise