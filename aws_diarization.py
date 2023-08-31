import time
import boto3
import traceback
from typing import Type
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger
import requests
import random
import string

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
            unique_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))

            path = path.replace('s3://','')
            if self.s3_bucket_name in path:
                path = path.replace(self.s3_bucket_name, "").lstrip("/")  
            
            if path == '/':
                path = '' 
            
            job_name = self.job_name_prefix + "_" + file_name
            job_uri = "s3://" + self.s3_bucket_name + (path if path in (None, "") else "/" + path) + "/" + unique_string + "_" + file_name
            
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
            #return self.process_results(self.get_data(status))
        except:
            logger.error(f"Error occured. URI: {job_uri}, Path: {path}, file_name: {file_name}\n\n{traceback.format_exc()}")
            return f"Error occured. URI: {job_uri} Path: {path}, file_name: {file_name} \n\n{traceback.format_exc()}"
        
    # def get_data(self, data):
    #     transcript_url = data['TranscriptionJob']['Transcript']['TranscriptFileUri']
    #     response = requests.get(transcript_url)
    #     if response.status_code == 200:
    #         return response.text
    #     else:
    #         return f"Error occured.\n\n{traceback.format_exc()}
        
    # def process_results(self, data):
    #     segments = data['results']['speaker_labels']['segments']
    #     items = data['results']['items']

    #     master_transcript = {}
    #     confidences = []

    #     for seg in segments:
    #         speaker = seg['speaker_label']
    #         start_time = float(seg['start_time'])
    #         end_time = float(seg['end_time'])
            
    #         this_segment = {}
            
    #         for item in items:
    #             if 'start_time' in item.keys():
    #                 item_start = float(item['start_time'])
    #                 item_end = float(item['end_time'])
    #                 if item_start >= start_time and item_end <= end_time:
    #                     confidences.append(float(item['alternatives'][0]['confidence']))
    #                     if 'content' in item['alternatives'][0].keys():
    #                         word = item['alternatives'][0]['content']
    #                         if speaker in this_segment.keys():
    #                             this_segment[speaker].append(word)
    #                         else:
    #                             this_segment[speaker] = [word]
            
    #         master_transcript.update(this_segment)

    #     total_length = 0
    #     for seg in segments:
    #         total_length += (float(seg['end_time']) - float(seg['start_time'])) * 1000

    #     average_confidence = sum(confidences)/len(confidences)

    #     result_text = ""

    #     for speaker, words in master_transcript.items():
    #         result_text = result_text + f'{int(float(segments[0]["start_time"]) * 1000)}ms : Speaker {int(speaker[-1])+1} : {" ".join(words)}'

    #     result_text = result_text + f'\nTotal Length: {int(total_length)}ms, Average Confidence: {average_confidence : .2f}'

    #     return result_text