import time
import boto3
import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.lib.logger import logger
import random
import string
import json
import json, datetime
import io
import os
from aws_helpers import add_file_to_resources, get_file_content, handle_s3_path, transcribe_valid_characters

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
    
    def _execute(self, target_file: str):
        try:
            file_name = os.path.basename(target_file)
            path = os.path.dirname(target_file)

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
        and writes it to a text string. Formatting is applied to highlight low-confidence areas.
        It also ensures punctuations and words are kept together without unwanted space.

        :param data: The string of JSON data
        :param threshold_for_grey: The confidence level below which transcriptions are considered low-confidence.
        :return: Written text as a string
        """
        data = json.loads(data)
        
        with io.StringIO() as file:
            # Begin by formatting and writing the document title and introduction
            title = f"Transcription of {data['jobName']}"
            file.write(f"{title}\n\n")

            file.write("Transcription using AWS Transcribe automatic speech recognition and"
                    " the 'tscribe' python package.\n")
            file.write(datetime.datetime.now().strftime("Document produced on %A %d %B %Y at %X.\n\n"))

            low_confidence_open = False

            # If speaker identification is included in the results
            if "speaker_labels" in data["results"].keys():
                # A segment is a continuous block of speech from the same speaker
                for segment in data["results"]["speaker_labels"]["segments"]:
                    # If the segment has any items, write the start time and the speaker
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
                            
                            # Mark low confidence words with brackets
                            if float(result["confidence"]) < threshold_for_grey and not low_confidence_open:
                                file.write(" [")
                                low_confidence_open = True
                            elif float(result["confidence"]) >= threshold_for_grey and low_confidence_open:
                                file.write("] ")
                                low_confidence_open = False

                            # Prepare the word with a trailing space
                            word_to_write = result['content'] + " "
                            # Initialise next_item as empty to handle cases when it might not be updated in the try-except
                            next_item = {}  
                            try:
                                # Get next item to check if it is punctuation
                                word_result_index = data["results"]["items"].index(word_result[0])
                                next_item = data["results"]["items"][word_result_index + 1]
                                # If it's a punctuation mark, replace the trailing space with the punctuation
                                # and follow formatting rules for sentence ending punctuation vs pause punctuation
                                if next_item["type"] == "punctuation":
                                    if next_item["alternatives"][0]["content"] in [".", "?", "!"]:
                                        word_to_write = word_to_write.rstrip() + next_item["alternatives"][0]["content"] + "  " 
                                    elif next_item["alternatives"][0]["content"] in [",", ";"]:
                                        word_to_write = word_to_write.rstrip() + next_item["alternatives"][0]["content"] + " " 
                                    else:
                                        word_to_write += next_item["alternatives"][0]["content"]
                            except IndexError:
                                pass

                            if next_item.get("type") != "punctuation" and not low_confidence_open and word_to_write.endswith(" "):
                                word_to_write = word_to_write

                            # Write the prepared word
                            file.write(word_to_write)

                        # Close bracket if we ended the segment on a low confidence word
                        if low_confidence_open:
                            file.write("]")
                            low_confidence_open = False 

                        # Start a new line for the next segment
                        file.write("\n")

            # Return what we've written as a continuous string
            return file.getvalue()
            # Begin by formatting and writing the document title and introduction
            title = f"Transcription of {data['jobName']}"
            file.write(f"{title}\n\n")

            file.write("Transcription using AWS Transcribe automatic speech recognition and"
                    " the 'tscribe' python package.\n")
            file.write(datetime.datetime.now().strftime("Document produced on %A %d %B %Y at %X.\n\n"))

            low_confidence_open = False

            # If speaker identification is included in the results
            if "speaker_labels" in data["results"].keys():

                # A segment is a continuous block of speech from the same speaker
                for segment in data["results"]["speaker_labels"]["segments"]:

                    # If the segment has any items, write the start time and the speaker
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

                            # Open bracket before low-confidence words
                            if float(result["confidence"]) < threshold_for_grey and not low_confidence_open:
                                file.write(" [")
                                low_confidence_open = True
                            # Close bracket after low-confidence words
                            elif float(result["confidence"]) >= threshold_for_grey and low_confidence_open:
                                file.write("] ")
                                low_confidence_open = False

                            # Prepare the word_to_write with no trailing space
                            word_to_write = result['content']

                            # Initialise next_item as empty to handle cases when it is not updated in the try-except
                            next_item = {}  
                            try:
                                # Get next item to check if it is punctuation
                                word_result_index = data["results"]["items"].index(word_result[0])
                                next_item = data["results"]["items"][word_result_index + 1]
                                # If it's a punctuation mark, append it directly without a space# If next_item isn't a punctuation, add a space so that words do not stick together  
                                if next_item["type"] == "punctuation":
                                    word_to_write += next_item["alternatives"][0]["content"]
                            except IndexError:
                                pass

                            # If next_item isn't a punctuation, add a space so that words do not stick together  
                                if next_item.get("type") != "punctuation" and not low_confidence_open:
                                    word_to_write += " "

                            # Write the formatted word + punctuation / space as appropriate
                            file.write(word_to_write)

                        # Close bracket if we ended the segment on a low confidence word
                        if low_confidence_open:
                            file.write("]")
                            low_confidence_open = False 

                        # Start a new line for the next segment
                        file.write("\n")

            # Return what we've written as a continuous string
            return file.getvalue()