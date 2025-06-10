import google.generativeai as genai
from langchain.tools import BaseTool
import re

# Gemini API Key
GOOGLE_API_KEY = "AIzaSyCex4H9ReM_dfVbR_WHOARgUocwKi5jySI"
genai.configure(api_key=GOOGLE_API_KEY)



def format_gemini_response(response):
   response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
   response = re.sub(r"##\s*(.*?)##", r"<h2>\1</h2>", response)
   response = response.replace("* ", "- ")
   response = re.sub(r"(\d\.)\s", r"<br>\1 ", response)
   response = response.replace("\n", "<br>")
   return response
# Custom Tool
class SoilImageAnalyzerTool(BaseTool):
    name: str = "SoilImageAnalyzer"
    description: str = "Analyzes soil images for texture, color, moisture, fertility, etc."

    def _run(self, image_bytes: bytes):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }

            response = model.generate_content([
                "This is a photo of soil taken for agricultural analysis. "
                "You are a professional soil scientist. Analyze the image based on common soil analysis parameters.",
                image_part,
                (
                    "Answer with:\n"
                    "- Soil texture (sand, clay, silt, loam)\n"
                    "- Color analysis and fertility prediction\n"
                    "- Moisture level guess\n"
                    "- Possible organic content\n"
                    "- Crop suitability suggestions"
                )
            ])
            return response.text
        except Exception as e:
            return f"Error analyzing the soil image: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async method not implemented.")