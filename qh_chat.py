import dataclasses
from enum import Enum

import json

from pathlib import Path
import requests

QUALIFIED_HEALTH_DEV_BASE_URL = "https://dev-api.qualifiedhealthai.com"

class ParsedChatSSEStream:
    """Parses the Server-Sent Events stream from the Qualified Health system.
    
    For now it works synchronously. Streaming is not supported, so the entire
    stream must have been received before parsing.
    """

    def __init__(self, response):
        """Parses the Server-Sent Events stream from the Qualified Health system.
        
        response: requests.Response received from a Qualified Health query."""
        
        # Decode the response into a list of strings.
        lines = [_.decode("utf-8") for _ in response.iter_lines()]
        assert all([type(l) is str for l in lines])
    
        nlines = len(lines)
        i = 0

        # Initialize the variables that will be set by the stream.
        self.chat_id = None
        self.message_id = None
        self.filtered_message = None
        self.filter_info = None
        self.llm_response = []

        while i < nlines:
            line = lines[i]

            # Detect an event.
            if line.startswith("event"):
                event_type = lines[i].split(": ")[-1]

                # Process an event.
                self._handle_event(lines, i, event_type)
                i += 1

            i += 1
        
        self.llm_response = "".join(self.llm_response)

        return
    
    def _handle_event(self, lines, i, event_type):
        """Handles an event in the Server-Sent Events stream."""

        # Detect unexpected end of stream.
        if i == len(lines) - 1:
            raise Exception(f"Expected 'data: ' on the next line, but reached end of stream instead.")

        # Read data, which is always on the next line.
        data_line = lines[i+1]
        data = json.loads(data_line[len("data: "):].strip())

        if event_type == "user_message":
            # Extract the chat ID and message ID. 
            self.chat_id = data["chatId"]
            self.message_id = data["id"]
        if event_type == "filter_result":
            # Extract the filtered message and filter information.
            self.filtered_message = data["userMessage"]["output"]
            self.filter_info = data["userMessage"]["outMap"]
        if event_type == "assistant_message_chunk":
            # Extract the response from the assistant and append to the list
            # of chunks. These will be combined to form a single string response
            # at the end.
            self.llm_response.append(data["chunk"])

        return
    
class UploadFileType(Enum):
    """Enum for the types of files that can be uploaded to the Qualified Health platform."""
    TEXT = "text/plain"
    PDF = "application/pdf"
    TABLE = "table"
    IMAGE = "image"

@dataclasses.dataclass 
class UploadFile:
    """Dataclass for a file to be uploaded to the Qualified Health platform."""
    filepath: Path
    filetype: UploadFileType

class QualifiedHealthChat:
    """Class representing a chat session with the Qualified Health system."""
    def _upload_file(self, upload_file):
        """Uploads a file to the Qualified Health platform."""

        # API endpoint for file upload.
        url = f"{QUALIFIED_HEALTH_DEV_BASE_URL}/files/upload"
        
        # Request headers.
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.user_token}"
        }
        
        # File to be uploaded.
        self.file_info = upload_file
        files = {
            "file": (
                upload_file.filepath.name, 
                open(upload_file.filepath, "rb"), 
                # Only plaintext files are supported for the moment.
                upload_file.filetype.value)
        }

        response = requests.post(url, headers=headers, files=files)

        # Close the file.
        files['file'][1].close()
        
        if response.status_code != 201:
            raise Exception(f"File upload unsuccessful: status code {response.status_code} returned. " + 
                             "Reason given:\n {response.text}")
            
        return response.json()
            
    def _start_conversation(self, initial_message):
        """Initiates the chat session. This and other functions
        are currently *blocking*. That means that they will only return
        when the LLM response has fully been received.
        
        Args:
            initial_message: The first message to be sent to the LLM. 
                             This is required by the platform.
        """
        data = {
            "action": "establish_chat",
            "message": initial_message,
            "type": self.conversation_type
        }
        
        if self.using_file:
            # Set the file to be used in the conversation.
            data["files"] = [self.file_dict]

            # Post the request as JSON.
            response = requests.post(
                url=f"{QUALIFIED_HEALTH_DEV_BASE_URL}/conversation",
                stream=False,
                json=data, 
                headers={
                    "Authorization": f"Bearer {self.user_token}"})
        else:
            # Post the request using the data field. 
            response = requests.post(
                url=f"{QUALIFIED_HEALTH_DEV_BASE_URL}/conversation",
                stream=False,
                data=data, 
                headers={"Authorization": f"Bearer {self.user_token}"})
        
        if response.status_code != 201:
            raise Exception(f"Conversation start unsuccessful: status code {response.status_code} returned. " + 
                             f"Reason given:\n {response.reason}")
            
        # Parse the SSE stream.
        parsed_response = ParsedChatSSEStream(response)
        # Set the chat ID. This is only done here.
        self.chat_id = parsed_response.chat_id

        if parsed_response.filtered_message is not None:
            # If the initial message was filtered for PII/PHI, the platform
            # expects a resubmission. Currently the resubmission is done in a way
            # that accepts the proposed PHI/PII filtering.

            # Save the filtered message in the query history.
            self.query_history.append(parsed_response.filtered_message)

            # Save the filtering information.
            self.filter_mappings.append(parsed_response.filter_info)

            # Resubmit the message. This method will also store the response.
            self.send_message(message=None,
                              resubmit=True,
                              message_id=parsed_response.message_id,
                              already_filtered=True)
        else:
            # Save the initial message in the query history.
            self.query_history.append(initial_message)

            # Save an empty dictionary to reflect that no filtering was done.
            self.filter_mappings.append({})

            # Save the response in the response history.
            self.response_history.append(parsed_response.llm_response)
            
        return
            
    def send_message(self, message, resubmit=False, message_id=None, already_filtered=False):
        """Sends a message to the LLM. This method is also responsible for storing
        the query, filtering information, and response, as well as resubmitting in
        the event of filtering. This method is currently *blocking* - that is, no
        streaming. It will only return when the LLM response has been fully received.
        
        Args:
            message: The message to be sent. If resubmit is True, this should be None.
            resubmit: Whether the message is a resubmission of a previous message.
            message_id: The ID of the message to be resubmitted. Required if resubmit is True.
            already_filtered: Whether the message has already been filtered for PII/PHI.
        """
        if resubmit:
            if message_id is None:
                raise Exception("Resubmitting a message requires specifying a message ID.")
            
            data = {
                "action": "resubmit_message",
                "chatId": self.chat_id,
                "messageId": message_id,
                "type": self.conversation_type
            }
            
            if already_filtered: 
                # Specifying 'rulesOverrides' tells the system whether to accept
                # the redaction or not. For now, we always accept it.
                data["rulesOverrides"] = []
        else:
            data = {
                "action": "new_message",
                "message": message,
                "chatId": self.chat_id,
                "type": self.conversation_type
            }
        
        if self.using_file:
            # Set the file to be used in the conversation.
            data["files"] = [self.file_dict]

            # Post the request as JSON. JSON posting is required here, hence the
            # control flow.
            print(f"Posting request with data: {data}")
            response = requests.post(
                url=f"{QUALIFIED_HEALTH_DEV_BASE_URL}/conversation",
                stream=False,
                json=data, 
                headers={"Authorization": f"Bearer {self.user_token}"})
        else:
            # Otherwise send an ordinary request.
            response = requests.post(
                url=f"{QUALIFIED_HEALTH_DEV_BASE_URL}/conversation",
                stream=False,
                data=data, 
                headers={"Authorization": f"Bearer {self.user_token}"})
        
        if response.status_code != 201:
            raise Exception(f"Message send unsuccessful: status code {response.status_code} returned.")
        
        # Parse the SSE stream.
        parsed_response = ParsedChatSSEStream(response)
        print(parsed_response.__dict__)
        if not already_filtered and parsed_response.filtered_message is not None:
            # Save the filtered message in the query history.
            self.query_history.append(parsed_response.filtered_message)

            # Save the redaction information.
            self.filter_mappings.append(parsed_response.filter_info)

            # Resubmit the message. This call will store the response.
            self.send_message(message=None,
                              resubmit=True,
                              message_id=parsed_response.message_id,
                              already_filtered=True)

            # By this point everything is done, so return.
            return
        if not resubmit:
            # Save the message in the query history.
            self.query_history.append(message)

            # Save an empty dictionary to reflect that no filtering was done.
            self.filter_mappings.append({})

        # Save the response in the response history.
        self.response_history.append(parsed_response.llm_response)
        return
        
    def __init__(self, user_token, initial_message, upload_file=None, qh_url=QUALIFIED_HEALTH_DEV_BASE_URL):
        """Initializes a chat session with the Qualified Health system.
        
        Args:
            user_token: The user token for the platform API.
            initial_message: The first message to be sent to the LLM.
            upload_file: An optional file to be uploaded to the platform. Defaults to None.
            qh_url: URL to use to connect to the Qualified Health system. Defaults to the URL of the dev
                    deployment.
        """

        # User authentication token. Currently must be manually retrieved 
        # from the platform. The easiest way to do so is to log into the platform
        # through the front end, hit 'Inspect' in the browser and navigate to the 
        # requests (in Chrome, this is under the 'Network' tab). The token will be
        # in the headers of the request to the API.
        self.user_token = user_token 
    
        # Initialize query and response history, as well as filtering history.
        self.query_history = []
        self.response_history = []
        self.filter_mappings = []
        if upload_file:
            # Upload the file to the system.
            print("Uploading file...")
            self.file_dict = self._upload_file(upload_file)
            print("File uploaded.")
            self.using_file = True

            # When using a file this is the correct conversation type.
            self.conversation_type = "dataSource"
        else:
            self.file_dict = None
            self.using_file = False

            # Set the conversation type to 'general' since a file isn't being used.
            self.conversation_type = "general"
        
        print("I made it here!")
        
        # Start the conversation.
        print("Starting conversation...")
        self._start_conversation(initial_message)
        print("Conversation started.")