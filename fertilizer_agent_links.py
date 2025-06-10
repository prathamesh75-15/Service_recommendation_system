import os
import google.generativeai as genai
from langchain.agents import initialize_agent, AgentType,AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import GooglePalm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from serpapi import GoogleSearch

SERPAPI_KEY = "fab7340aeb37ad7229cb7bf02666945ef88593c3c7f96fd909e5b50b75025555"  


load_dotenv()

# Configure Google Gemini
genai.configure(api_key="AIzaSyDXMkkyiS0d3cu01RBUVOH9dQuZfruWkxc")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyDXMkkyiS0d3cu01RBUVOH9dQuZfruWkxc",
    temperature=0.7
)

# Final Fertilizer Advisor Function
def get_fertilizer_analysis(crop, budget, fertilizer_type="any", soil=None, land=None):
    prompt = f"""
    You are an expert agricultural assistant.

    A farmer is cultivating **{crop}** with a budget of ₹{budget}.
    {"Soil type: " + soil + "." if soil else ""}
    {"Land area: " + land + "." if land else ""}
    They are looking for **{fertilizer_type}** fertilizer options.

    Tasks:
    1. Suggest 2-3 suitable {fertilizer_type if fertilizer_type != "any" else ""} fertilizers for this crop and soil.
    2. Make sure they fit within the budget of ₹{budget}.
    3. For each fertilizer, explain:
        - Whether it's organic or chemical
        - NPK ratio or organic composition
        - Why it's suitable for this crop and soil
        - How much can be bought within budget
    4. Recommend the best option with a brief comparison.
    The Result should be in max 500 words
    """

    response = llm.invoke(prompt)
    structured_response = format_gemini_response(response.content if hasattr(response, "content") else str(response))

    # fertilizer_names = re.findall(r"\b([A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*){0,5}(?:\s*-\s*[A-Z][a-zA-Z0-9\-]*)?)\s*\(", structured_response)
    # print(fertilizer_names)
    # fert_links = get_fertilizer_links(fertilizer_names)
    # print("Links found: ", fert_links)

    return structured_response



def get_fertilizer_links(structured_response):
    fertilizer_names = re.findall(r"\b([A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*){0,5}(?:\s*-\s*[A-Z][a-zA-Z0-9\-]*)?)\s*\(", structured_response)
    links = {}
    for fert in fertilizer_names:
        try:
            params = {
                "engine": "google",
                "q": f"buy {fert} fertilizer online India",
                "api_key": SERPAPI_KEY,
                "num": 3
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract links from first few organic results
            if "organic_results" in results:
                urls = [item["link"] for item in results["organic_results"][:3]]
                links[fert] = urls
            else:
                links[fert] = ["No link found"]

        except Exception as e:
            links[fert] = [f"Error: {str(e)}"]

    return links


def format_gemini_response(response):
   response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
   response = re.sub(r"##\s*(.*?)##", r"<h2>\1</h2>", response)
   response = response.replace("* ", "- ")
   response = re.sub(r"(\d\.)\s", r"<br>\1 ", response)
   response = response.replace("\n", "<br>")
   return response