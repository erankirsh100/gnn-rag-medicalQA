from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
import google.generativeai as genai
import csv
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add this line to set the secret key

with open('data/unique_symptoms.csv') as f:
    symptoms_list = [row[0] for row in csv.reader(f)]

# Load your Gemini API Key
GOOGLE_API_KEY = "AIzaSyB0-HsZQqEZZkLxA1bfqDRo69Zmze_37tI"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Define the Gemini Model
model = genai.GenerativeModel('gemini-2.5-pro')

def generate_answer(user_data):

    prompt = f"""
    Hi, I'm a {user_data['age']} {user_data['gender']}, in the last few {user_data['duration']} i have been feeling the following symptoms: {user_data['symptom']}.
    The severity of my symptoms is {user_data['severity']} in a scale of 1-5. 
    Here is a description of my symptoms: {user_data['description']}.
    
    Additional info about me:
    I am a {user_data['smoker']}, and I excercise {user_data['excercise']}.

    I would like to know what may be the disease causing these symptoms.
    """
    print(prompt)
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def form():
    return render_template('Form.html')

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
    return render_template('result.html', answer=markdown.markdown(answer))

if __name__ == '__main__':

    app.run(debug=True)
