import time
import boto3
import traceback
import random
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger

class AWSTextToSpeechSchema(BaseModel):
    text: str = Field(
        ...,
        description="Text to be converted into speech",
    )
    gender: Optional[str] = Field(
        None,
        description="Gender of the voice (Male/Female)",
    )
    age: Optional[str] = Field(
        None,
        description="Age of the voice (Adult/Child)"
    )

class AWSTextToSpeechTool(BaseTool):
    name = "AWS Text To Speech Tool"
    description = "Text To Speech tool that converts given text into speech and store the audio file into an S3 bucket"
    args_schema: Type[AWSTextToSpeechSchema] = AWSTextToSpeechSchema

    s3_bucket_name = "neutralaiz-superagi-demo"
    job_name_prefix = "AWSTextToSpeechJob"

    voices = {
        "Male": {"Adult": ["Joey", "Matthew"], "Child": ["Justin", "Kevin"]},
        "Female": {"Adult": ["Joanna", "Kendra","Kimberly","Salli"], "Child": ["Ivy", "Ruth"]}
    }
    
    def _execute(self, text: str, gender: Optional[str] = None, age: Optional[str] = None):
        try:
            aws_access_key_id = get_config("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = get_config("AWS_SECRET_ACCESS_KEY")
            
            polly_client = boto3.Session(
                aws_access_key_id=aws_access_key_id,                  
                aws_secret_access_key=aws_secret_access_key,
                region_name='eu-west-2'
            ).client('polly')
            
            if gender and age:
                assert gender in self.voices and age in self.voices[gender]
                voiceId = random.choice(self.voices[gender][age])
            else:
                gender = random.choice(list(self.voices.keys()))
                age = random.choice(list(self.voices[gender].keys()))
                voiceId = random.choice(self.voices[gender][age])

            response = polly_client.start_speech_synthesis_task(
                VoiceId=voiceId,
                OutputS3BucketName=self.s3_bucket_name,
                OutputFormat='mp3', 
                Text=text,
                Engine='neural'
            )

            taskId = response['SynthesisTask']['TaskId']
            task_status = polly_client.get_speech_synthesis_task(TaskId = taskId)

            start_time = time.time()  # get the current time

            while task_status['SynthesisTask']['TaskStatus'] == 'IN_PROGRESS':
                if time.time() - start_time > 30:  # if more than 30 seconds have passed
                    print("Operation timed out.")
                    break
                print("Text to Speech conversion in progress...")
                time.sleep(5)
                task_status = polly_client.get_speech_synthesis_task(TaskId = taskId)

            if task_status['SynthesisTask']['TaskStatus'] == 'COMPLETED':
                print("Text to Speech conversion completed!")
            else:
                print(f"Task failed with reason: {task_status['SynthesisTask']['TaskStatusReason']}")

            return task_status
        except:
            logger.error(f"Error occured.\n\n{traceback.format_exc()}")
            return None