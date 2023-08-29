import traceback
from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool
import os
from superagi.llms.openai import OpenAi
import tiktoken
from superagi.config.config import get_config

class OpenAIDirectSchema(BaseModel):
    system: str = Field(
        "You are a helpful assistant",
        description="How the AI should act - examples: You are a data scientist and you..., You are a software architect creating a diagram, You are a writer building a story...",
    )
    message: str = Field(
        ...,
        description="The message you would like the model to respond to",
    )
    data: Optional[str] = Field(
        None,
        description="Structured data you would like to add in a seperate message, in the same thread",
    )
    model: Optional[str] = Field(
        "gpt-3.5-turbo",
        description="Which OpenAI Model to use:\nname: gpt-3.5-turbo description: 4k length, cheapest, default\nname: gpt-4 description: 8k length, smartest\nname: gpt-3.5-turbo-16k description: 16k length, longest",
    )
    
class OpenAIDirectTool(BaseTool):
    name = "OpenAI Direct Call Tool"
    description = (
        "Make a call directly to the OpenAI GPT series of models."
    )
    args_schema: Type[OpenAIDirectSchema] = OpenAIDirectSchema

    def _execute(self, system: str = "", message: str = "", data: str = "", model: str = "gpt-3.5-turbo"):
        # Retrieve the API key from the environment variable or, if not set, the application's config
        api_key = os.environ.get("OPENAI_API_KEY", None) or get_config("OPENAI_API_KEY", "")

        if not api_key:
            raise Exception("OpenAI API Key not found in environment variables or application configuration.")

        # Initialize the OpenAi class with the API key and chosen model
        openai_api = OpenAi(api_key=api_key, model=model)

        # Package the system and message inputs into the messages list format as needed by the Chat API
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]

        # If there is additional structured data, add it to the messages list as a user message
        if data:
            messages.append({"role": "user", "content": data})

        # Configure a safety cushion. This is a certain percentile, let's say 20%, that you will hold on top of the tokens calculated from messages.
        safety_cushion = 0.05

        # Get the number of tokens used by your messages
        tokens_used = self.num_tokens_from_messages(messages, model)

        # Calculate your safe max length as (1 + safety_cushion) * tokens_used
        max_length = int((1 + safety_cushion) * tokens_used)

        # Cap your max_length to the maximum model token limit if it exceeds it
        max_model_token_limit = get_config("MAX_MODEL_TOKEN_LIMIT")
        max_length = min(max_length, max_model_token_limit)

        # Perform the API call and return the results
        result = openai_api.chat_completion(messages, max_length)
        
        if "error" in result:
            # There was an error with the API call, raise an exception
            raise Exception(f"Error running OpenAI API: {result['error']}")
        
        return result['content']
    
    
    def num_tokens_from_messages(self, messages, model: str ="gpt-3.5-turbo"):
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens