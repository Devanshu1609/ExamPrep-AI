# agents/youtube_summarizer_agent.py
from langchain.agents import create_agent
from tools.youtube_transcript_tool import YouTubeTranscriptTool
from tools.analysis_storage_tool import AnalysisStorageTool

class YouTubeSummarizerAgent:
    def __init__(self, model: str = "gemini-1.5-flash", persist_directory: str = "vector_store"):
        self.model = model
        self.youtube_tool = YouTubeTranscriptTool()
        self.storage_tool = AnalysisStorageTool(persist_directory)
        self.tools = self.youtube_tool.get_tools() + self.storage_tool.get_tools()
        self.name = "youtube_summarizer_agent"
        self.description = (
            "Fetches and analyzes YouTube videos for educational purposes, "
            "producing topic-wise explanations, timestamps, and key takeaways."
        )

    def create_agent(self):
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=(
                "You are an AI academic assistant that analyzes YouTube lecture or tutorial videos.\n\n"
                "YOUR GOAL:\n"
                "Use the transcript of a YouTube video to create a detailed, structured learning summary "
                "that helps students revise the video content effectively.\n\n"

                "INSTRUCTIONS:\n"
                "1. Extract and review the transcript text carefully.\n"
                "2. Identify different topics or sections covered in the video.\n"
                "3. For each topic, write a detailed, student-friendly explanation.\n"
                "4. Include approximate timestamps if available.\n"
                "5. Emphasize important concepts, definitions, examples, and problem-solving steps.\n"
                "6. Do NOT just create a short summary — your goal is a rich, educational explanation.\n\n"

                "OUTPUT FORMAT (JSON ONLY):\n"
                "{\n"
                "  \"video_title\": string,\n"
                "  \"summary_overview\": string,                // overall description of the video\n"
                "  \"topics\": [                                // list of sections/topics covered\n"
                "     {\n"
                "         \"topic\": string,\n"
                "         \"timestamp_range\": string,          // e.g. '00:00 - 05:30'\n"
                "         \"detailed_explanation\": string,\n"
                "         \"key_points\": [string, ...],\n"
                "         \"examples\": [string, ...]\n"
                "     }\n"
                "  ],\n"
                "  \"key_takeaways\": [string, ...],\n"
                "  \"recommended_actions\": [string, ...]       // e.g. 'Revise Topic X', 'Practice Question Type Y'\n"
                "}\n\n"

                "AFTER generating the JSON:\n"
                "- ALWAYS call the tool `store_analysis_result` with:\n"
                "    agent_name='youtube_summarizer_agent',\n"
                "    result_type='video_summary',\n"
                "    result=<your JSON>\n"
                "  (Include video_id if available.)\n\n"

                "DO NOT output text or markdown outside of JSON.\n"
                "Your explanations should be clear, educational, and optimized for students preparing for exams.\n"
                "- RETURN the same JSON as the final output, without any extra commentary."
            ),
            name=self.name
        )
