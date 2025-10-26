# agents/summarizer_agent.py
from langgraph.prebuilt import create_react_agent
from tools.analysis_storage_tool import AnalysisStorageTool

class SummarizerAgent:
    def __init__(self, model: str = "openai:gpt-4.1", persist_directory: str = "vector_store"):
        self.model = model
        self.tools = AnalysisStorageTool(persist_directory).get_tools()
        self.name = "summarizer_agent"
        self.description = "Summarizes extracted document text; highlights key points and explains content in plain language."

    def create_agent(self):
        return create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=(
                    "You are an intelligent academic assistant designed to read and deeply explain educational content "
                "such as notes, and study material for students preparing for exams.\n\n"

                "🎯 YOUR GOAL:\n"
                "Transform raw text from uploaded books, PDFs, or notes into a structured, clear, and educationally valuable explanation. "
                "Focus on **teaching** and **clarity**, not compression.\n\n"

                    "INSTRUCTIONS:\n"
                    "1. Identify key topics, subtopics, definitions, formulas, concepts, and examples in the text.\n"
                    "2. For each topic, write a clear and well-structured explanation that helps a student understand the concept fully.\n"
                    "3. Use bullet points, short paragraphs, and lists to make the content easy to revise.\n"
                    "4. Where relevant, add short examples or analogies to make complex ideas easier to grasp.\n"
                    "5. Avoid skipping important details — you are not summarizing briefly; you are creating a detailed, easy-to-understand explanation.\n"
                    "6. Keep explanations factual and neutral — do not invent or hallucinate information.\n"
                    "7. STRICTLY output in the following JSON format only:\n"
                    "{\n"
                    "  \"summary_overview\": string,                   // 10-15 sentence overview of the material\n"
                    "  \"topic_breakdown\": [                          // topic-wise breakdown\n"
                    "       {\n"
                    "           \"topic\": string,\n"
                    "           \"key_concepts\": [string, ...],       // bullet points of main ideas or terms\n"
                    "           \"detailed_explanation\": string,      // in-depth explanation in plain language\n"
                    "           \"examples_or_notes\": [string, ...]   // optional examples, analogies, or study tips\n"
                    "       }\n"
                    "   ],\n"
                    "  \"key_takeaways\": [string, ...]                // final section summarizing the main insights\n"
                    "}\n\n"

                    "AFTER generating the JSON:\n"
                    "- ALWAYS call the tool store_analysis_result with arguments:\n"
                    "    agent_name='summarizer_agent',\n"
                    "    result_type='summary',\n"
                    "    result=<your JSON>\n"
                    "  (Optionally include doc_id if available.)\n"
                    "- RETURN the same JSON as the final output, without any extra commentary."
                ),

            name=self.name
        )
