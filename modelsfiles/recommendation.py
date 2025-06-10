import torch
from sentence_transformers import util
import pickle
from keras.models import load_model
import numpy as np
from sentence_transformers import SentenceTransformer



# Load necessary data
embeddings = pickle.load(open('service_embeddings_new.pkl', 'rb'))
if isinstance(embeddings, dict):
    embeddings = torch.tensor(list(embeddings.values()))
services = pickle.load(open('services_new.pkl', 'rb'))
rec_model = pickle.load(open('rec_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# loading of model neural network 
# query_type_model = tf.keras.models.load_model('C:\\Users\\DELL\\small recommendation system\\neural_network_model_for_QueryType.h5')  # Load your query type model
query_type_model = load_model("optimized_model.h5") 
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
def services_ranking(scores_list):
    # Get all services with their similarity scores
    all_services_with_scores = [
        (list(services.keys())[idx], scores_list[idx]*100) for idx in range(len(scores_list))
    ]
    all_services_with_scores.sort(key=lambda x: x[1], reverse=True)
    # print(all_services_with_scores)



def recommendation(farmer_issues):
    
    cosine_scores = util.cos_sim(rec_model.encode(farmer_issues, convert_to_tensor=True), embeddings)
    scores_list = cosine_scores[0].tolist()
    # services_ranking(scores_list)
    print(scores_list)
    top_results = torch.topk(cosine_scores, k=6) # get top 3 recommendation

    indices = top_results.indices[0].tolist() # extract the indices of top result
    top_scores = top_results.values[0].tolist()
    recommended_services = [
        list(services.keys())[idx] for idx in indices
        ]
    print("recommended service",recommended_services)
    # print(query_type)

    #services releated to querytype{
    # releated_service_to_queryType =[]
    # for service, description in services.items():
    #     description_embedding = rec_model.encode(description, convert_to_tensor =True)
    #     query_embbeding = rec_model.encode(query_type, convert_to_tensor =True)
    #     similarity_score = util.cos_sim(query_embbeding ,description_embedding )
    #     if similarity_score > 0.3:  # Threshold for similarity
    #         releated_service_to_queryType.append(service)
    #}

    # the main one ->
    # similarity_score= util.cos_sim(rec_model.encode(query_type, convert_to_tensor=True), embeddings)
    # similarity_score_list = similarity_score[0].tolist()
    # services_ranking(similarity_score_list)
    # # print(similarity_score)
    # print(similarity_score_list)
    # # print(similarity_score[0].tolist())
    # top_results = torch.topk(similarity_score, k=6)
    # indices1 = top_results.indices[0].tolist()
    # releated_service_to_queryType=[
    #     list(services.keys())[idx1] for idx1 in indices1
    # ]

    # print(releated_service_to_queryType)

    # all_recommendations = list(set(recommended_services + releated_service_to_queryType))
    # print(all_recommendations)

    filtered_recommendations = []

# Filter recommendations from farmer_issues
    for idx, score in zip(indices, top_scores):
        # normalized_score = normalize_score(score)
        if score >= 0.40:
            filtered_recommendations.append(list(services.keys())[idx])

    # Filter recommendations from query_type
    # for idx1, score in zip(indices1, similarity_score[0].tolist()):
    #     # normalized_score = normalize_score(score)
    #     if score >= 0.35:
    #         filtered_recommendations.append(list(services.keys())[idx1])

    # Remove duplicates
    # filtered_recommendations = list(filtered_recommendations)
    print("filter recommendation",filtered_recommendations)
    return filtered_recommendations


#function for query_type prediction 
def predict_query_type(user_input):

    user_input_cleaned = user_input.lower().replace('[^\w\s]', '')
    user_input_vectorized = vectorizer.transform([user_input_cleaned]).toarray()
    # user_input_vectorized = tf.sparse.reorder(user_input_vectorized)
    query_type = query_type_model.predict(user_input_vectorized)
    predicted_index = np.argmax(query_type, axis=1)

    # Decode the predicted index to get the query type
    predicted_query_type = label_encoder.inverse_transform(predicted_index)

    print(predicted_query_type)
    return predicted_query_type[0].strip()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # query_cleaned = user_input.lower().replace('[^\w\s]', '')  # Normalize text

    # query_vectorized = vectorizer.transform([query_cleaned]).toarray()
    # query_vectorized = query_vectorized.astype(np.float32)
    # interpreter.set_tensor(input_details[0]['index'], query_vectorized)
    # interpreter.invoke()
    # prediction = interpreter.get_tensor(output_details[0]['index'])
    # predicted_index = np.argmax(prediction, axis=1)
    # print(predicted_index)
    # predicted_query_type = label_encoder.inverse_transform(predicted_index)

    # return predicted_query_type[0]
    
