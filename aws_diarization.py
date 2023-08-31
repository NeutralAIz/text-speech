import time
import boto3
import traceback
from typing import Type
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger
import random
import string
from aws_helpers import add_file_to_resources, get_file_content

class AWSDiarizationSchema(BaseModel):
    path: str = Field(
        ...,
        description="Directory path inside the bucket (hardcoded start at resources/app/workspace/)",
    )
    file_name: str = Field(
        ...,
        description="Name of the target audio file",
    )

class AWSDiarizationTool(BaseTool):
    name = "AWS Diarization Tool"
    description = (
        "Automated speech recognition (ASR) tool that converts speech from an audio "
        "file into raw text")
    args_schema: Type[AWSDiarizationSchema] = AWSDiarizationSchema

    agent_id: int = None
    agent_execution_id: int = None

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
            
            job_name = self.job_name_prefix + "_" + unique_string + "_" + file_name
            job_uri = "s3://" + self.s3_bucket_name + (path if path in (None, "") else "/" + path) + "/" + file_name
            
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

            data = self.get_data(status)

            return self.process_results(data)
        except:
            logger.error(f"Error occured. URI: {job_uri}, Path: {path}, file_name: {file_name}\n\n{traceback.format_exc()}")
            return f"Error occured. URI: {job_uri} Path: {path}, file_name: {file_name} \n\n{traceback.format_exc()}"
        
    def get_data(self, data):
        try:
            transcript_url = data['TranscriptionJob']['Transcript']['TranscriptFileUri']
            file_name = self.get_filename_from_url(transcript_url)
            resource = add_file_to_resources(self.toolkit_config.session, file_name, self.agent_id, self.agent_execution_id)
            return get_file_content(self.toolkit_config.session, resource.path + resource.name, self.agent_id, self.agent_execution_id)
        except:
            logger.error(f"Error occured. file_name: {file_name}, transcript_url: {transcript_url}, \n\n{traceback.format_exc()}")
        
    def get_filename_from_url(self, url):
        try:
            """
            Get the file path from URL, after bucket name.
                
            :param url: The S3 URL
            """
            url_pieces = url.split('/')
            
            amazonaws_index = None
            for i, piece in enumerate(url_pieces):
                if "amazonaws.com" in piece:
                    amazonaws_index = i
                    break
            
            # the part after bucket is the file path
            file_path = '/'.join(url_pieces[amazonaws_index + 2:])
                    
            return file_path
        except:
            logger.error(f"Error occured. file_path: {file_path}, url: {url}, \n\n{traceback.format_exc()}")

    def process_results(self, data):
        try:
            if 'speaker_labels' in data['results']:
                segments = data['results']['speaker_labels']['segments']
            else:
                segments = [{'start_time': item['start_time'], 'end_time': item['end_time'], 
                            'speaker_label': 'spk_0'} for item in data['results']['items'] if 'start_time' in item]
                            
            speakers = data['results']['speaker_labels']['speakers'] if 'speaker_labels' in data['results'] else 1
            items = data['results']['items']

            master_transcript = {f'spk_{i}': [] for i in range(speakers)}
            confidences = []

            for seg in segments:
                speaker = seg['speaker_label']
                start_time = float(seg['start_time'])
                end_time = float(seg['end_time'])

                for item in items:
                    if 'start_time' in item.keys() and ('speaker_label' not in item or item['speaker_label'] == speaker):
                        item_start = float(item['start_time'])
                        item_end = float(item['end_time'])
                        if item_start >= start_time and item_end <= end_time:
                            if 'confidence' in item['alternatives'][0].keys():
                                confidences.append(float(item['alternatives'][0]['confidence']))
                            if 'content' in item['alternatives'][0].keys():
                                word = item['alternatives'][0]['content']
                                master_transcript[speaker].append(word)

            total_length = 0
            for seg in segments:
                total_length += (float(seg['end_time']) - float(seg['start_time'])) * 1000

            average_confidence = sum(confidences)/len(confidences) if confidences else 0.0

            result_text = ''

            for speaker, words in master_transcript.items():
                result_text += f'{int(float(segments[0]["start_time"]) * 1000)}ms : Speaker {int(speaker[-1])+1} : {" ".join(words)}\n'

            result_text += f'Total Length: {int(total_length)}ms, Average Confidence: {average_confidence : .2f}'
        
            return result_text

        except Exception as e:
            logger.error(f"Error occurred. data: {data}, \n\n{traceback.format_exc()}")