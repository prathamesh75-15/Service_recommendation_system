from flask import Flask, render_template, request, jsonify, redirect, session
import os
from googletrans import Translator
# from deep_translator import GoogleTranslator
from modelsfiles.fertilizer_agent_links import get_fertilizer_analysis, get_fertilizer_links
from modelsfiles.recommendation import recommendation, services_ranking, predict_query_type, services
from modelsfiles.agentic_models import organic_farming_advisor, generate_followup_question, ai_agent_answer, scheme_display_agent, format_gemini_response
from modelsfiles.soil_tool import SoilImageAnalyzerTool , format_gemini_response


app = Flask(__name__)
app.secret_key = os.urandom(24) 


service_images = {
    'Loan': '/static/loan.jpeg',
    'Training': '/static/traning.jpg',
    'Subsidy': '/static/subsidy.jpg',
    'Market Access': '/static/market access.jpg',
    'Soil Testing': '/static/soil testing.jpg',
    'Crop Selection Advisory': '/static/crop selection adivisory.jpg',
    'Weather Alerts': '/static/weather alert.jpg',
    'Irrigation Plans': '/static/irrigations.jpg',
    'Organic Farming Support': '/static/organic farming support.jpg',
    'Precision Farming': '/static/precision farming.jpg',
    'Crop-Specific Training': '/static/organic farming vegitable.jpg',
    'Wheat Monitoring': '/static/wheat monitoring.jpg',
    'Corn Disease Detection': '/static/corn diseases detection.jpg',
    'Rice Water Management': '/static/rice water management.jpg',
    'Vegetable Organic Farming': '/static/vegitable organic farming.jpg',
    'Water Analysis Facility': '/static/Water Analysis Facility.jpg',
    'Tractor Booking Facility':'/static/Tractor Booking Facility.jpg',
    'Seed Selection Advisory':'/static/Seed Selection Advisory.jpg',
    'Fertilizer Recommendation': '/static/Fertilizer Recommendation.jpg',
    'Pest and Disease Control': '/static/Pest and Disease Control.jpg',
    'Weather Forecasting': '/static/Weather Forecasting.jpg',
    'Government Scheme Assistance':'/static/Government Scheme.webp',
    'Rental Equipment Facility': '/static/Rental Equipment Facility.jpg',
    'Insurance Service':'/static/Insurance Service.jpg'
      
}



@app.route('/')
def index():
    return render_template('Home.html')




@app.route('/tryservice')
def tryservice():
    return render_template('index_new.html')




@app.route('/recommendation', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        farmer_issues = request.form.get('farmer_issues', '').strip()
        followup_answer = request.form.get('followup_answer', '').strip()
        previous_query = request.form.get('previous_query', '')

        # Use follow-up answer if provided
        if followup_answer:
            farmer_issues = followup_answer

        if not farmer_issues:
            return render_template('index_new.html', error="Please enter farmer issues.")

        # Fetch session counter (to track follow-up attempts)
        followup_count = session.get('followup_count', 0)

        # Predict query type
        # query_type = predict_query_type(farmer_issues)
        recommendations = recommendation(farmer_issues)

        # If recommendations found, display them
        if recommendations:
            session.pop('followup_count', None)  # Reset follow-up count
            return render_template('try_error_result.html', recommendations=recommendations, services=services, service_images=service_images)

        # Limit follow-up attempts to 1 (or any desired threshold)
        if followup_count < 1:
            followup_question = generate_followup_question(farmer_issues)
            if followup_question:
                session['followup_count'] = followup_count + 1  # Increment follow-up count
                return render_template('index_new.html', followup_question=followup_question, previous_query=farmer_issues)

        # If no relevant results even after follow-up, use AI Agents
        structured_response = ai_agent_answer(farmer_issues)
        scheme_display = scheme_display_agent(farmer_issues)
        print(scheme_display )
        print(structured_response)
   
        # Reset follow-up count after AI agents step in
        session.pop('followup_count', None)
        
        return render_template('try_error_result.html', structured_response=structured_response, pricing_data=scheme_display)

    return render_template('index_new.html')




@app.route('/api/services')
def get_services():
    # return jsonify([{"name": name, "description": desc} for name, desc in services.items()])
    return jsonify([
        {
            "name": name,
            "description": desc,
            "image": service_images.get(name, "/static/images/default.jpg")  # Default image if not found
        } 
        for name, desc in services.items()
    ])



@app.route('/services')
def services_page():
    return render_template('services.html')




@app.route('/services/<service_name>')
def service_page(service_name):
    formatted_service = service_name.replace("_", " ")  # Convert URL format to match dictionary keys

    if formatted_service == "Rental Equipment Facility":
        return redirect("http://127.0.0.1:5001/")
    # if formatted_service == "Insurance Service":
    #     return render_template(f"services/insurance_services/insurance.html")
    if formatted_service in services:
        return render_template(f"services/{service_name}.html")
    else:
        return "<h2>Service not found</h2>", 404




@app.route('/<form_name>_form')
def render_form(form_name):
    form_template = f"services/{form_name}_form.html"
    try:
        return render_template(form_template)
    except:
        return "Form not found", 404




@app.route("/chat")
def home():
    return render_template("organic_new.html")




@app.route("/chat-message", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    language = data.get("language", "en")  # Default is English
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Language Mapping
    lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
    lang_name = lang_map.get(language, "English")

    response = organic_farming_advisor(query, lang_name)
    return jsonify({"response": response})




translator = Translator()

@app.route('/translate', methods=['POST'])
def translate_response():
    data = request.get_json()
    selected_language = data.get('language')

    translated_data = {
        'query': translator.translate(data.get('query', ''), dest=selected_language).text,
        'ai_response': translator.translate(data.get('ai_response', ''), dest=selected_language).text,
        'search_results': translator.translate(data.get('search_results', ''), dest=selected_language).text,
        'suggested_action': translator.translate(data.get('suggested_action', ''), dest=selected_language).text
    }

    return jsonify({'success': True, 'translated': translated_data})




@app.route('/translate_pricing', methods=['POST'])
def translate_pricing_schemes():
    try:
        abc= Translator()
        data = request.get_json()
        selected_language = data.get('language')
        schemes = data.get('schemes', [])

        if not selected_language or not isinstance(schemes, list):
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400

        translated_schemes = []

        for scheme in schemes:
            translated_scheme = {
                'name': abc.translate(scheme.get('name', ''), dest=selected_language).text,
                'description': abc.translate(scheme.get('description', ''), dest=selected_language).text,
                'eligibility': abc.translate(scheme.get('eligibility', ''), dest=selected_language).text,
                'financial_support': abc.translate(scheme.get('financial_support', ''), dest=selected_language).text
            }
            translated_schemes.append(translated_scheme)
        print(translated_schemes)
        return jsonify({'success': True, 'translated': {'schemes': translated_schemes}})
    
    except Exception as e:
        # Log the error in console for debugging
        print("Translation Error:", str(e))
        # Send error as JSON so frontend can handle it
        return jsonify({'success': False, 'error': str(e)}), 500




@app.route('/fertilizer_rec')
def fertilizer_rec():
    return render_template('fertilizer.html')



@app.route('/fertilizer-calculator', methods=['POST'])
def calculate():
    crop = request.form['crop']
    budget = request.form['amount']
    soil = request.form.get('soil', '')
    land = request.form.get('land_area', '')

    result = get_fertilizer_analysis(crop, budget, soil, land)
    fertilizer_links = get_fertilizer_links(result)

    return render_template('fertilizer_result.html', result=result, crop=crop, amount=budget, fertilizer_links=fertilizer_links)






@app.route('/translate_fertilizer', methods=['POST'])
def translate_fertilizer():
    try:
        
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "en")
        translated_text = translator.translate(text, dest=lang).text
        # print(translated_text)
        return jsonify({"translated_text": translated_text})

    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({"error": "Translation failed", "details": str(e)}), 500
    


# soil analysis service route
@app.route("/soil-analysis", methods=["GET", "POST"])
def soil_analysis():
    tool = SoilImageAnalyzerTool()
    result = None
    if request.method == "POST":
        file = request.files.get("soil_image")
        if file and file.filename:
            image_bytes = file.read()
            result = tool._run(image_bytes=image_bytes)
            result = format_gemini_response(result)
        else:
            result = "No image uploaded."

    return render_template("soil_analysis.html", result=result)




if __name__ == '__main__':
    app.run(debug=True)
