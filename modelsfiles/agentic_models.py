import re
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent
import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import json

genai.configure(api_key="AIzaSyCRw5CDXp7ad6U9Uwjac-sc_Xcd7gLNaso")
llm = GoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key="AIzaSyCRw5CDXp7ad6U9Uwjac-sc_Xcd7gLNaso")





def format_gemini_response(response):
   response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
   response = re.sub(r"##\s*(.*?)##", r"<h2>\1</h2>", response)
   response = response.replace("* ", "- ")
   response = re.sub(r"(\d\.)\s", r"<br>\1 ", response)
   response = response.replace("\n", "<br>")
   return response




# Recommendation function
def organic_farming_advisor(query, language):
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        # prompt = f"Answer in {language}: {query}"
        prompt = f"""Answer in {language}: {query}
        answer should be max 300 words"""
        response = model.generate_content(prompt)
        # print(response)
        structured_response = format_gemini_response(response.text)
        return structured_response
    except Exception as e:
        return f"Error: {str(e)}"
    

    
# Follow-up question generator using Gemini
def generate_followup_question(user_query):
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    prompt = f"""Act as an agricultural expert. Given this user query: "{user_query}", 
    ask one concise follow-up question to better understand their problem. 
    Keep it natural and focused on farming-related aspects."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip('"').strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Could you please provide more details about your agricultural issue?"





# AI Agent to provide structured answers when no recommendation is found
# search_tool = DuckDuckGoSearchRun()
search_tool = TavilySearchResults(tavily_api_key="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")
def ai_agent_answer(user_query):
    """
    Fetches structured information using a combination of Gemini and DuckDuckGo search.
    """
    
    search_tool = TavilySearchResults(tavily_api_key="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")

    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
    )

    try:
        # 1. Initial Gemini Response
        # initial_response = llm.invoke(f"""
        # Please provide a **detailed and well-structured** response to the following question:

        # {user_query}    

        # Your response should include explanations, examples, and relevant details where necessary.
        #     """)
        initial_response=llm.invoke(f"Think you are agriculture experst.Please provide a concise answer to the following question: {user_query} in 100 to 200 words by which farmere can get initial idea.")
        print("initial response :",initial_response)

        # 2. DuckDuckGo Search for Further Information
        search_results = agent.run(user_query)
        print("search result :",search_results)
        # 3. Combine Gemini and Search Results
        # Ask Gemini to refine its answer based on the search results
        final_prompt = f"""

        You initially provided this response: {initial_response}.

        Here are some search results related to the query: {user_query}:
        {search_results}
        Please analyze both initial response and search resuult and provide a best result which for user query : {user_query} by thinking that you are an agriculture experst 
        Response should be accurate and comprehensive answer and it should be in max 300 wrods
       
        """
    

        final_response = llm.invoke(final_prompt)

        print("suggested_action :",final_response)
        print
        return {
            "structured_response": {
                "query": format_gemini_response(user_query),
                "gemini_initial_response": format_gemini_response(initial_response),
                "search_results": format_gemini_response(search_results),
                "suggested_action": format_gemini_response(final_response)
            }
        }
    except Exception as e:
        print(f"AI Agent Error: {e}")
        return {"structured_response": "No relevant information found."}

    



def scheme_display_agent(user_query):
    try:
        print("Scheme agent activated")  # Debugging Log
        
        # Step 1: Use DuckDuckGo search to find schemes
        search_query = f"Latest government schemes for farmers in India related to {user_query}"
        search_results = search_tool.run(search_query)  # Assuming search_tool is defined elsewhere
        print(f" Searching for: {search_query}")
        
        # Step 2: Use Gemini LLM to refine and structure the results
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        prompt = (
            f"You are an AI assistant for farmers. Based on the user's query: '{user_query}', "
            "analyze the following search results and extract **only the top 3 most relevant schemes**.\n\n"
            "Return the response in **valid JSON format** with the following structure:\n\n"
            "{\n"
            '    "schemes": [\n'
            '        {\n'
            '            "name": "Scheme Name",\n'
            '            "description": "Short Description",\n'
            '            "eligibility": "Eligibility Criteria",\n'
            '            "financial_support": "Financial Support Details"\n'
            '        },\n'
            '        ...\n'
            '    ]\n'
            "}\n\n"
            f"Here is the raw data:\n{search_results}"
        )

        refined_schemes = model.generate_content(prompt)
        print(refined_schemes.text.strip())
        json_match = re.search(r'\{.*\}', refined_schemes.text.strip(), re.DOTALL)
        # Step 3: Convert response to structured JSON
        if json_match:
            json_text = json_match.group()  # Extract JSON portion
            try:
                structured_data = json.loads(json_text)  # Convert JSON string to Python dictionary
                return structured_data  # ✅ Return structured JSON data
            except json.JSONDecodeError:
                print("❌ Error: Extracted text is not valid JSON.")
                return {"error": "Invalid response format from AI agent."}
        else:
            print("❌ Error: No JSON found in response.")
            return {"error": "Invalid response format from AI agent."}
    except Exception as e:
        print(f"Pricing Agent Error: {e}")
        return {"error": "Could not retrieve relevant pricing details."}
