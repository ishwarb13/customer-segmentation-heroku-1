from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert form data
    gender = 1 if request.form['gender'] == 'Male' else 0
    ever_married = 1 if request.form['ever_married'] == 'Yes' else 0
    age = int(request.form['age'])
    graduated = 1 if request.form['graduated'] == 'Yes' else 0
    work_experience = int(request.form['work_experience'])
    spending_score_map = {'Low': 0, 'Average': 1, 'High': 2}
    spending_score = spending_score_map[request.form['spending_score']]
    family_size = int(request.form['family_size'])
    var_1_map = {'Cat_1': 1, 'Cat_2': 2, 'Cat_3': 3, 'Cat_4': 4, 'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7}
    var_1 = var_1_map[request.form['var_1']]

    # Create input array in the specified order
    input_features = [gender, ever_married, age, graduated, work_experience, spending_score, family_size, var_1]
    final_features = [np.array(input_features)]

    print("Input features:", input_features) 
    print("Input features:", final_features) 

    # Predict using the model
    prediction = model.predict(final_features)

    # Map prediction to segment
    label = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D'}
    output = label[prediction[0]]

    return render_template('result.html', prediction_text=f'Customer Segment: {output}')

if __name__ == "__main__":
    app.run(debug=True)







