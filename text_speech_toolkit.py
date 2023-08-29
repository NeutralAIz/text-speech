from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseTool, BaseToolkit
from aws_diarization import AWSDiarizationTool
from aws_text_to_speech import AWSTextToSpeechTool


class LLMDirectToolkit(BaseToolkit, ABC):
    name: str = "Speech and Text Tools"
    description: str = "Work with text to speech and speech to text engines."

    def get_tools(self) -> List[BaseTool]:
        return [
            AWSDiarizationTool(), AWSTextToSpeechTool()
        ]

    def get_env_keys(self) -> List[str]:
        return []
