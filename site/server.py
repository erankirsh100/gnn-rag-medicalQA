from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
# from pipeline import run_pipeline
import google.generativeai as genai
import csv
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add this line to set the secret key
import threading
# import pipeline workaround
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from pipeline import run_pipeline

SYMPTOMS_CSV = os.path.join(PROJECT_ROOT, "site", "data", "unique_symptoms.csv")
with open(SYMPTOMS_CSV) as f:
    symptoms_list = [row[0] for row in csv.reader(f)]

# Load your Gemini API Key
GOOGLE_API_KEY = "AIzaSyB0-HsZQqEZZkLxA1bfqDRo69Zmze_37tI"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Define the Gemini Model
model = genai.GenerativeModel('gemini-2.5-pro')
first_load_flag = True
jobdone = None

def generate_answer(user_data):

    prompt = f"""
    Hi, I'm a {user_data['age']} {user_data['gender']}, in the last few {user_data['duration']} i have been feeling the following symptoms: {user_data['symptom']}.
    The severity of my symptoms is {user_data['severity']} in a scale of 1-5. 
    Here is a description of my symptoms: {user_data['description']}.
    
    Additional info about me:
    I am a {user_data['smoker']}, and I excercise {user_data['excercise']}.

    I would like to know what may be the disease causing these symptoms.

    keep your answer concise and to the point, do not expand on the input information.
    """
    global jobdone

    jobdone = False
    response = run_pipeline(prompt)["rag_gnn_output"]
    jobdone = True
    # init_searcher() should run at opening, so pipeline can run fast on first call
    return response

@app.route('/')
def index():
    global first_load_flag
    print("first_load_flag:", first_load_flag)
    if first_load_flag:
        print("First load - initializing searcher")
        dummy_user_data = dict()
        dummy_user_data['age']="25"
        dummy_user_data['gender'] = "male"
        dummy_user_data['symptom'] = "headache"
        dummy_user_data['description'] = "a persistent headache"
        dummy_user_data['duration'] = "2 days"
        dummy_user_data['severity'] = "4"
        dummy_user_data['smoker'] = "no"
        dummy_user_data['excercise'] = "regularly"
        # generate_answer(dummy_user_data)  # init_searcher() workaround
        first_load_flag = False
        threading.Thread(target=generate_answer, args=(dummy_user_data,)).start()
    return render_template('landingPage.html')

@app.route('/process')
def process():
    # innitialize the searcher to avoid delay on first request
    return {'done': jobdone}

@app.route('/form')
def form():
    return render_template('Form.html')

# @app.route('/')
# def form():
#     # If first load, show a lightweight loading page that will trigger server init.
#     # The /init endpoint performs the heavy initialization (once) and then redirects
#     # the browser back to '/' to render the real form.
#     global first_load_flag
#     if first_load_flag:
#         return render_template('landingPage.html')

#     return render_template('Form.html')

# @app.route('/init', methods=['GET'])
# def init_app():
#     """Endpoint called from loading.html to run heavy initialization once."""
#     global first_load_flag
#     if not first_load_flag:
#         return jsonify({"status": "ready"})

#     try:
#         # same dummy payload used previously to warm up pipeline/searcher
#         dummy_user_data = {
#             'age': "25",
#             'gender': "male",
#             'symptom': "headache",
#             'description': "a persistent headache",
#             'duration': "2 days",
#             'severity': "4",
#             'smoker': "no",
#             'excercise': "regularly"
#         }
#         # run heavy init (blocking) once
#         generate_answer(dummy_user_data)
#         first_load_flag = False
#         return jsonify({"status": "ready"})
#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower()
    matches = [s for s in symptoms_list if s.lower().startswith(query)]
    return jsonify(matches[:20])  # limit to 10 results

@app.route('/submit', methods=['POST'])
def submit():

    user_found = False

    user_data = {}
    user_data['age']=request.form.get('age')
    user_data['gender'] = request.form.get('gender')
    user_data['symptom'] = request.form.get('symptom')[:-2]
    user_data['description'] = request.form.get('description')
    user_data['duration'] = request.form.get('duration')
    user_data['severity'] = request.form.get('severity')
    user_data['smoker'] = request.form.get('smoker')
    user_data['excercise'] = request.form.get('exercise')


    prompt = f"""
    Hi, I'm a {user_data['age']} {user_data['gender']}, in the last few {user_data['duration']} i have been feeling the following symptoms: {user_data['symptom']}.
    The severity of my symptoms is {user_data['severity']} in a scale of 1-5. 
    Here is a description of my symptoms: {user_data['description']}.
    
    Additional info about me:
    I am a {user_data['smoker']}, and I excercise {user_data['excercise']}.

    I would like to know what may be the disease causing these symptoms.
    """


    # if not user_found:
    #     flash('{}'.format(prompt))
    #     return redirect(url_for('form'))

    answer = generate_answer(user_data)
    # linkedin_about = linkedin_about.split('*')[-1]
    # linkedin_about = linkedin_about.split(':')[-1]
    return render_template('result.html', answer=answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)