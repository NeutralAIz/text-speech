
from superagi.lib.logger import logger
from superagi.helper.resource_helper import ResourceHelper
from superagi.models.agent import Agent
from superagi.models.agent_execution import AgentExecution
from typing import Optional
import os
from unstructured.partition.auto import partition
from superagi.helper.s3_helper import S3Helper
import traceback
import re

def handle_s3_path(filepath):
    # Convert filepath to lower case for case insensitive comparison
    filepath_lower = filepath.lower()
    split_path = filepath.split("/")
    # For http/https, the required path starts from 6th element (index 5)
    if filepath_lower.startswith("http"):
        filename = "/".join(split_path[5:])
    # For s3, the required path starts from 4th element (index 3)
    elif filepath_lower.startswith("s3"):
        filename = "/".join(split_path[3:])
    else:
        filename = filepath  # If it's not http/https or s3, return the original string
    
    result = ensure_path(filename)
      
    return result

def ensure_path(filepath, account_for_s3: bool = False):    # pattern to match any s3 url format
    logger.info(f"ensure_path: filepath:{filepath}")
    new_filepath = ""
    root_path = ""

    if filepath == "/":
        filepath = ""

    try:
        root_path = ResourceHelper().get_root_output_dir() if not account_for_s3 else "resources" + ResourceHelper().get_root_output_dir()

        absolute_root = os.path.abspath(root_path)
        absolute_file = os.path.abspath(filepath)
        if absolute_file.startswith(absolute_root):
            return filepath
        else:
            parts_of_file_path = filepath.split(os.sep)
            parts_of_root_path = root_path.split(os.sep)
            for path in parts_of_root_path:
                if path in parts_of_file_path:
                    parts_of_file_path.remove(path)
            missing_path = os.sep.join(parts_of_file_path)
            new_filepath = os.path.join(root_path, missing_path)
            logger.info(f"ensure_path: new_filepath:{new_filepath}")
            return new_filepath
    except:
        logger.error(f"Error occured. filepath: {filepath}, root_path: {root_path}, new_filepath: {new_filepath}\n\n{traceback.format_exc()}")


def transcribe_valid_characters(targetValue: str):
    logger.info(f"transcribe_valid_characters: targetValue:{targetValue}")
    cleaned = re.sub(r'[^0-9a-zA-Z._-]+', '_', targetValue)
    logger.info(f"transcribe_valid_characters: cleaned:{cleaned}")
    return cleaned

def get_file_content(session, file_name: str, agent_id: int, agent_execution_id: int):
    """
    Read the content of a file.

    Args:
        file_name : The name of the file to read.
        agent_id : The id of the agent.
        agent_execution_id : The id of the agent execution.

    Returns:
        The content of the file as a string.
    """
    
    logger.info(f"get_file_content: file_name:{file_name}")
    try:
        final_path = ResourceHelper.get_agent_read_resource_path(file_name, agent=Agent.get_agent_from_id(
            session=session, agent_id=agent_id), agent_execution=AgentExecution
                                                                .get_agent_execution_from_id(session=session,
                                                                                            agent_execution_id=agent_execution_id))

        temporary_file_path = None
        final_name = final_path.split('/')[-1]
        
        save_directory = "/"
        
        if final_path.split('/')[-1].lower().endswith('.txt'):
            return S3Helper().read_from_s3(final_path)
        else:
            temporary_file_path = save_directory + file_name
            with open(temporary_file_path, "wb") as f:
                contents = S3Helper().read_binary_from_s3(final_path)
                f.write(contents)

        if final_path is None or not os.path.exists(final_path) and temporary_file_path is None:
            raise FileNotFoundError(f"File '{file_name}' not found.")
        directory = os.path.dirname(final_path)
        os.makedirs(directory, exist_ok=True)

        if temporary_file_path is not None:
            final_path = temporary_file_path

        logger.info(f"get_file_content: final_path:{final_path}")

        elements = partition(final_path)
        content = "\n\n".join([str(el) for el in elements])

        if temporary_file_path is not None:
            os.remove(temporary_file_path)
    
        return content              
    except:
        logger.error(f"Error occured.\n\n{traceback.format_exc()}")
        return {traceback.format_exc()}

def add_file_to_resources(session, file_name, agent_id: int, agent_execution_id: int):
    logger.info(f"add_file_to_resource: file_name:{file_name}")
    agent = Agent.get_agent_from_id(session, agent_id)
    agent_execution = AgentExecution.get_agent_execution_from_id(session, agent_execution_id)
    return ResourceHelper.make_written_file_resource(file_name, agent, agent_execution, session)