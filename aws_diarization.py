import time
import boto3
import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.resource_manager.file_manager import FileManager
import random
import string
import json
import json, datetime
import io
import os
from aws_helpers import add_file_to_resources, get_file_content, handle_s3_path, transcribe_valid_characters, ensure_path

class AWSDiarizationSchema(BaseModel):
    target_file: str = Field(
        ...,
        description="Name of the target audio file.",
    )

class AWSDiarizationTool(BaseTool):
    name = "AWS Diarization Tool"
    description = (
        "Tool that transcribes an audio file into raw text.  Handles multiple speakers.")
    args_schema: Type[AWSDiarizationSchema] = AWSDiarizationSchema

    agent_id: int = None
    agent_execution_id: int = None

    s3_bucket_name = "neutralaiz-superagi-demo"
    region_name = 'us-east-1'
    job_name_prefix = "AWSDiarizationJob"
    resource_manager: Optional[FileManager] = None
    
    def _execute(self, target_file: str):
        try:
            file_name = os.path.basename(target_file)
            path = os.path.dirname(target_file)

            unique_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))
            
            path = ensure_path(path, True)
            
            logger.info(f"_execute: file_name: {file_name}, path: {path}")

            job_name = transcribe_valid_characters(self.job_name_prefix + "_" + unique_string + "_" + file_name)
            job_uri = "s3://" + self.s3_bucket_name + "/" + path + ("/" if not file_name.startswith("/") and not path.endswith("/") else "") + file_name
            
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

            raw_data = self.get_data(status)
            processed_data = self.process_to_text(raw_data)

            processed_data_filename = transcribe_valid_characters(self.job_name_prefix + "_" + unique_string + "_" + "transcript" + "_" + file_name)
            processed_data_filename = self.resource_manager.write_file(path + ("/" if not processed_data_filename.startswith("/") and not path.endswith("/") else "") + processed_data_filename, processed_data)

            return processed_data_filename
        except:
            logger.error(f"Error occured. URI: {job_uri}, Path: {path}, file_name: {file_name}\n\n{traceback.format_exc()}")
            return f"Error occured. URI: {job_uri} Path: {path}, file_name: {file_name} \n\n{traceback.format_exc()}"
        
    def get_data(self, data):
        try:
            transcript_url = data['TranscriptionJob']['Transcript']['TranscriptFileUri']
            file_path = handle_s3_path(transcript_url)
            logger.error(f"get_data - transcript_url: {transcript_url}, file_path: {file_path}")
            resource = add_file_to_resources(self.toolkit_config.session, file_path, self.agent_id, self.agent_execution_id)
            file_content = get_file_content(self.toolkit_config.session, resource.name, self.agent_id, self.agent_execution_id)
        except:
            logger.error(f"Error occured. file_path: {file_path}, transcript_url: {transcript_url}")


    def convert_time_stamp(self, timestamp: str) -> str:
        """
        Function to help convert timestamps from s to H:M:S
        """
        delta = datetime.timedelta(seconds=float(timestamp))
        seconds = delta - datetime.timedelta(microseconds=delta.microseconds)
        return str(seconds)

    def process_to_text(self, data: str, threshold_for_grey: float = 0.96) -> str:
        """
        This function takes a JSON string of transcribe data, extracts the key information, 
        and formats it into a string. 
        It applies formatting to highlight low-confidence areas.
        It also ensures punctuations and words are kept together without unwanted space.

        :param data: JSON data as a string 
        :param threshold_for_grey: Confidence level below which transcriptions are uncertain.
        :return: Transcription formatted as a text string
        """
        
        # Parse json from string format
        data = json.loads(data)

        # Open a stringIO object to write the transcription into
        with io.StringIO() as file:
            
            # Begin writing with job details and time of transcription
            title = f"Transcription of {data['jobName']}"
            file.write(f"{title}\n\n")
            file.write("Transcription using AWS Transcribe automatic speech recognition and"
                    " the 'tscribe' python package.\n")
            file.write(datetime.datetime.now().strftime("Document produced on %A %d %B %Y at %X.\n\n"))

            # Initialize boolean to track low confidence word sequences
            low_confidence_open = False

            # Check if there are speaker labels in the data
            if "speaker_labels" in data["results"].keys():
                
                # For each speaker segment in the transcription labels
                for segment in data["results"]["speaker_labels"]["segments"]:
                    
                    # If there are segment items (words)
                    if len(segment["items"]) > 0:
                        
                        # Write the speaker name and start time
                        file.write(f"{self.convert_time_stamp(segment['start_time'])} "
                            f"{segment['speaker_label']}:")

                        # For each word spoken by speaker
                        for word in segment["items"]:
                            
                            # Filter for pronunciation words
                            pronunciations = list(
                                filter(
                                    lambda x: x["type"] == "pronunciation",
                                    data["results"]["items"],
                                )
                            )
                            
                            # Find the exact pronunciation item for current word
                            word_result = list(
                                filter(
                                    lambda x: x["start_time"] == word["start_time"]
                                    and x["end_time"] == word["end_time"],
                                    pronunciations,
                                )
                            )
                            
                            # Get the alternative with highest confidence from word
                            result = sorted(
                                word_result[-1]["alternatives"], key=lambda x: x["confidence"]
                            )[-1]

                            # If the confidence level is lower than threshold, start a low-confidence sequence
                            if float(result["confidence"]) < threshold_for_grey and not low_confidence_open:
                                file.write(" [")
                                low_confidence_open = True

                            # Get the next item, check if it's punctuation to avoid space before it
                            word_to_write = result['content'] + " "
                            next_item = {}  
                            try:
                                word_result_index = data["results"]["items"].index(word_result[0])
                                next_item = data["results"]["items"][word_result_index + 1]
                                if next_item["type"] == "punctuation":
                                    if next_item["alternatives"][0]["content"] in [".", "?", "!"]:
                                        word_to_write = word_to_write.rstrip() + next_item["alternatives"][0]["content"] + "  " 
                                    elif next_item["alternatives"][0]["content"] in [",", ";"]:
                                        word_to_write = word_to_write.rstrip() + next_item["alternatives"][0]["content"] + " " 
                                    else:
                                        word_to_write += next_item["alternatives"][0]["content"]
                            except IndexError:
                                pass

                            # If the word confidence is above threshold, finish a low-confidence sequence
                            if float(result["confidence"]) >= threshold_for_grey and low_confidence_open:
                                file.write(word_to_write.rstrip())  # Remove trailing space
                                file.write("] ")  
                                low_confidence_open = False
                            else:
                                file.write(word_to_write)

                        # If there is an unclosed low-confidence sequence at the end of speaker segment, close it
                        if low_confidence_open:
                            file.write("] ")  
                            low_confidence_open = False 

                        # Add a new line after each speaker segment
                        file.write("\n")  

            # Get the resulting transcription string from the StringIO object
            return file.getvalue()