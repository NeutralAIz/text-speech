import time
import boto3
import traceback
from typing import Type
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger

class AWSDiarizationSchema(BaseModel):
    path: str = Field(
        ...,
        description="Path of the audio file",
    )
    file_name: str = Field(
        ...,
        description="Name of the audio file",
    )

class AWSDiarizationTool(BaseTool):
    name = "AWS Diarization Tool"
    description = (
        "Automated speech recognition (ASR) tool that converts speech from an audio "
        "file into raw text")
    args_schema: Type[AWSDiarizationSchema] = AWSDiarizationSchema

    s3_bucket_name = "neutralaiz-superagi-demo"
    region_name = 'us-east-1'
    job_name_prefix = "AWSDiarizationJob"
    
    def _execute(self, path: str, file_name: str):
        try:
            job_name = self.job_name_prefix + "_" + file_name
            job_uri = "s3://" + self.s3_bucket_name + "/" + path + "/" + file_name
            
            aws_access_key_id = get_config("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = get_config("AWS_SECRET_ACCESS_KEY")   
            
            transcribe = boto3.client('transcribe', region_name=self.region_name, 
                                      aws_access_key_id=aws_access_key_id,
                                      aws_secret_access_key=aws_secret_access_key)

            transcribe.start_transcription_job(
                TranscriptionJobName = job_name,
                Media = {'MediaFileUri': job_uri},
                OutputBucketName = self.s3_bucket_name,
                LanguageCode = 'en-US', 
                Settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 3}    
            )

            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName = job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)
            
            return status  
        except:
            logger.error(f"Error occured.\n\n{traceback.format_exc()}")
            return None