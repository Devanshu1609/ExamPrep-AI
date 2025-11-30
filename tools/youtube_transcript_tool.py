# tools/youtube_transcript_tool.py
import os
from dotenv import load_dotenv
from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

class YouTubeTranscriptTool:
    def __init__(self):
        self.name = "youtube_transcript_tool"
        self.description = "Fetches transcript text for a given YouTube video URL."
        load_dotenv()
        self.api_key = os.getenv("YOUTUBE_API_KEY")

    @tool("get_youtube_transcript", return_direct=True)
    def get_youtube_transcript(self, video_url: str) -> str:
        """
        Fetches transcript text from the given YouTube video URL.
        Returns the full transcript text.
        """
        try:
            video_id = parse_qs(urlparse(video_url).query).get("v", [None])[0]
            if not video_id:
                raise ValueError("Invalid YouTube URL. Could not extract video ID.")
            if not self.api_key:
                raise ValueError("YouTube API key not set. Please add YOUTUBE_API_KEY to your .env file.")
            # Example: Use the API key in a request (pseudo-code, replace with actual API call if needed)
            # import requests
            # response = requests.get(f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={self.api_key}&part=snippet")
            # if response.status_code != 200:
            #     raise Exception(f"YouTube API error: {response.text}")
            # transcript_data = # parse transcript from response
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry["text"] for entry in transcript_data])
            return transcript_text
        except Exception as e:
            return f"Error fetching transcript: {str(e)}"

    def get_tools(self):
        return [self.get_youtube_transcript]
