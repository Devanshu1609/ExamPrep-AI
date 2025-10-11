# tools/youtube_transcript_tool.py
from langchain.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

class YouTubeTranscriptTool:
    def __init__(self):
        self.name = "youtube_transcript_tool"
        self.description = "Fetches transcript text for a given YouTube video URL."

    @tool("get_youtube_transcript", return_direct=True)
    def get_youtube_transcript(self, video_url: str) -> str:
        """
        Fetches transcript text from the given YouTube video URL.
        Returns the full transcript text.
        """
        try:
            # Extract video ID
            video_id = parse_qs(urlparse(video_url).query).get("v", [None])[0]
            if not video_id:
                raise ValueError("Invalid YouTube URL. Could not extract video ID.")
            
            # Fetch transcript
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry["text"] for entry in transcript_data])
            return transcript_text
        except Exception as e:
            return f"Error fetching transcript: {str(e)}"

    def get_tools(self):
        return [self.get_youtube_transcript]
