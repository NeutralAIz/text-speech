import time
import boto3
import traceback
import random
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.helper.resource_helper import ResourceHelper
from superagi.models.agent import Agent
from superagi.models.agent_execution import AgentExecution

class AWSTextToSpeechSchema(BaseModel):
    text: str = Field(
        ...,
        description="Text to be converted into speech.  Can optionally contain SSML.",
    )
    path: str = Field(
        ..., 
        description="Path of the S3 bucket to save the audio file (hardcoded start at resources/app/workspace/)"
    )
    fileprefix: str = Field(
        ..., 
        description="Prefix prepended to the audio file to be saved",
    )
    gender: Optional[str] = Field(
        None,
        description="Gender of the voice (Male/Female)",
    )
    age: Optional[str] = Field(
        None,
        description="Age of the voice (Adult/Child)"
    )
    voice: Optional[str] = Field(
        None,
        description='Voice to use.  Leave blank for random.'
    )
    ssml: Optional[bool] = Field(
        False,
        description="Does the text contain SSML codes?"
    )

class AWSTextToSpeechTool(BaseTool):
    name = "AWS Text To Speech Tool"
    description = "Text To Speech tool that converts given text into speech and store the audio file into an S3 bucket.  Available voices: Male: <Adult: [Joey, Matthew], Child: [Justin, Kevin]>, Female: <Adult: [Joanna, Kendra, Kimberly, Salli], Child: [Ivy, Ruth]>"
    args_schema: Type[AWSTextToSpeechSchema] = AWSTextToSpeechSchema

    agent_id: int = None
    agent_execution_id: int = None

    s3_bucket_name = "neutralaiz-superagi-demo"
    job_name_prefix = "AWSTextToSpeechJob"
    region_name = 'us-east-1'

    voices = {
        "Male": {"Adult": ["Joey", "Matthew"], "Child": ["Justin", "Kevin"]},
        "Female": {"Adult": ["Joanna", "Kendra","Kimberly","Salli"], "Child": ["Ivy", "Ruth"]}
    }
    
    def _execute(self, text: str, path: str, fileprefix: str, gender: Optional[str] = None, age: Optional[str] = None, voiceId: Optional[str] = None, ssml: Optional[bool] = False):
        try:
            aws_access_key_id = get_config("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = get_config("AWS_SECRET_ACCESS_KEY")
            
            polly_client = boto3.Session(
                aws_access_key_id=aws_access_key_id,                  
                aws_secret_access_key=aws_secret_access_key,
                region_name=self.region_name
            ).client('polly')
            
            if voiceId is None:
                if gender and age:
                    assert gender in self.voices and age in self.voices[gender]
                    voiceId = random.choice(self.voices[gender][age])
                else:
                    gender = random.choice(list(self.voices.keys()))
                    age = random.choice(list(self.voices[gender].keys()))
                    voiceId = random.choice(self.voices[gender][age])

            response = polly_client.start_speech_synthesis_task(
                #TaskId=filename,
                OutputS3KeyPrefix=path.strip('/') + "/" + fileprefix,
                VoiceId=voiceId,
                OutputS3BucketName=self.s3_bucket_name,
                OutputFormat='mp3', 
                Text=text,
                Engine='neural',
                TextType='ssml' if ssml else 'text'
            )

            taskId = response['SynthesisTask']['TaskId']
            task_status = polly_client.get_speech_synthesis_task(TaskId = taskId)

            start_time = time.time()  # get the current time

            while task_status['SynthesisTask']['TaskStatus'].upper() in ['INPROGRESS', 'SCHEDULED']:
                if time.time() - start_time > 120:  # if more than 30 seconds have passed
                    print("Operation timed out.")
                    break
                print("Text to Speech conversion in progress...")
                time.sleep(5)
                task_status = polly_client.get_speech_synthesis_task(TaskId = taskId)

            if task_status['SynthesisTask']['TaskStatus'].upper() == 'COMPLETED':
                print("Text to Speech conversion completed!")
                
                # Extract file name from the S3 URL
                audio_file_url = task_status['SynthesisTask']["OutputUri"]
                file_name = audio_file_url.split("/")[-1]
                
                # Pass this session to the Helper method
                self.add_audio_to_resources(file_name, self.toolkit_config.session)
                print("Text to Speech conversion completed!")

            else:
                raise Exception(f"Task failed with status: {task_status['SynthesisTask']['TaskStatus']}")

            return task_status
        except:
            logger.error(f"Error occured.\n\n{traceback.format_exc()}")
            return {traceback.format_exc()}
        

    def add_audio_to_resources(self, file_name, session):
        agent = Agent.get_agent_from_id(self.toolkit_config.session, self.agent_id)
        agent_execution = AgentExecution.get_agent_execution_from_id(session, self.agent_execution_id)
        ResourceHelper.make_written_file_resource(file_name, agent, agent_execution, session)