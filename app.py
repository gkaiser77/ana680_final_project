#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('car_evaluation_model.pkl')

# Define the feature names
feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Define the mapping from encoded values to labels
label_mapping = {
    0: 'unacc',
    1: 'acc',
    2: 'good',
    3: 'vgood'
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extract data from form
        data = {
            'buying': int(request.form['buying']),
            'maint': int(request.form['maint']),
            'doors': int(request.form['doors']),
            'persons': int(request.form['persons']),
            'lug_boot': int(request.form['lug_boot']),
            'safety': int(request.form['safety'])
        }

        # Convert the data to a DataFrame with column names
        input_data = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert the prediction to a label
        prediction_label = label_mapping.get(prediction[0], 'Unknown')

        # Return the prediction in the rendered HTML
        return render_template('index.html', prediction=prediction_label)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)



