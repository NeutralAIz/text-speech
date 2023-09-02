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
import json
import json, datetime
import io
from aws_helpers import add_file_to_resources, get_file_content, handle_s3_path, transcribe_valid_characters

class AWSDiarizationSchema(BaseModel):
    path: str = Field(
        ...,
        description="Directory path inside the bucket",
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
            logger.info(f"_execute: file_name: {file_name}, path: {path}")
            unique_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))
            
            path = handle_s3_path(path)

            job_name = transcribe_valid_characters(self.job_name_prefix + "_" + unique_string + "_" + file_name)
            job_uri = "s3://" + self.s3_bucket_name + "/" + path + file_name
            
            logger.info(f"_execute: job_name: {job_name}, job_uri: {job_uri}")

            aws_access_key_id = get_config("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = get_config("AWS_SECRET_ACCESS_KEY")   
            
            transcribe = boto3.client('transcribe', region_name=self.region_name, 
                                      aws_access_key_id=aws_access_key_id,
                                      aws_secret_access_key=aws_secret_access_key)

            transcribe.start_transcription_job(
                TranscriptionJobName = job_name,
                Media = {'MediaFileUri': job_uri},
                OutputBucketName = self.s3_bucket_name,
                OutputKey = path,
                LanguageCode = 'en-US', 
                Settings = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 3}    
            )

            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName = job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)

            data = self.get_data(status)
            result = self.process_to_text(data)
            return result
        except:
            logger.error(f"Error occured. URI: {job_uri}, Path: {path}, file_name: {file_name}\n\n{traceback.format_exc()}")
            return f"Error occured. URI: {job_uri} Path: {path}, file_name: {file_name} \n\n{traceback.format_exc()}"
        
    def get_data(self, data):
        try:
            transcript_url = data['TranscriptionJob']['Transcript']['TranscriptFileUri']
            file_path = handle_s3_path(transcript_url)
            resource = add_file_to_resources(self.toolkit_config.session, file_path, self.agent_id, self.agent_execution_id)
            return get_file_content(self.toolkit_config.session, resource.name, self.agent_id, self.agent_execution_id)
        except:
            logger.error(f"Error occured. file_path: {file_path}, transcript_url: {transcript_url}, \n\n{traceback.format_exc()}")


    def convert_time_stamp(timestamp: str) -> str:
        """ Function to help convert timestamps from s to H:M:S """
        delta = datetime.timedelta(seconds=float(timestamp))
        seconds = delta - datetime.timedelta(microseconds=delta.microseconds)
        return str(seconds)

    def process_to_text(self, data, threshold_for_grey=0.98):
        data = json.loads(data)
        
        with io.StringIO() as file:

            # Document title and intro
            title = f"Transcription of {data['jobName']}"
            file.write(f"{title}\n\n")

            # Document intro
            file.write("Transcription using AWS Transcribe automatic speech recognition and"
                    " the 'tscribe' python package.\n")
            file.write(datetime.datetime.now().strftime("Document produced on %A %d %B %Y at %X.\n\n"))

            low_confidence_open = False

            # Transcript
            # If speaker identification
            if "speaker_labels" in data["results"].keys():

                # A segment is a blob of pronunciation and punctuation by an individual speaker
                for segment in data["results"]["speaker_labels"]["segments"]:

                    # If there is content in the segment, write the time and speaker
                    if len(segment["items"]) > 0:
                        file.write(f"{self.convert_time_stamp(segment['start_time'])} "
                                f"{segment['speaker_label']}:")

                        # For each word in the segment...
                        for word in segment["items"]:

                            # Get the word with the highest confidence
                            pronunciations = list(
                                filter(
                                    lambda x: x["type"] == "pronunciation",
                                    data["results"]["items"],
                                )
                            )
                            word_result = list(
                                filter(
                                    lambda x: x["start_time"] == word["start_time"]
                                    and x["end_time"] == word["end_time"],
                                    pronunciations,
                                )
                            )
                            result = sorted(
                                word_result[-1]["alternatives"], key=lambda x: x["confidence"]
                            )[-1]

                            # If the word is low confidence and there is no open bracket, open one
                            if float(result["confidence"]) < threshold_for_grey and not low_confidence_open:
                                file.write(" [")
                                low_confidence_open = True
                            elif float(result["confidence"]) >= threshold_for_grey and low_confidence_open:
                                file.write("] ")
                                low_confidence_open = False 

                            # Write the word                        
                            file.write(f"{result['content']}")

                            # If the next item is punctuation, write it
                            try:
                                word_result_index = data["results"]["items"].index(
                                    word_result[0]
                                )
                                next_item = data["results"]["items"][word_result_index + 1]
                                if next_item["type"] == "punctuation":
                                    file.write(next_item["alternatives"][0]["content"])
                            except IndexError:
                                pass

                        # Close the bracket if we ended the segment on a low confidence word
                        if low_confidence_open:
                            file.write("]")
                            low_confidence_open = False 

                        # Add a line break after each segment
                        file.write("\n")
                        
            # Get written data as string
            return file.getvalue()