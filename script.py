from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model using tf.saved_model.load
model = tf.saved_model.load('saved_model')

# Function to preprocess and predict sentiment
def predict_sentiment(review_text):
    # Preprocess the input text
    processed_text = [review_text]
    
    # Make predictions
    # Note: Use the proper index for the output layer of your model
    model_signature = list(model.signatures.values())[0]
    output = model(tf.constant(processed_text, dtype=tf.string)) # Use the correct index based on your model summary
    output_tensor = output['dense_1']
    
    # Return the sentiment prediction as a string
    predictions = tf.nn.sigmoid(output_tensor).numpy()
    sentiment = "Positive" if predictions[0][0] > 0.5 else "Negative"
    return sentiment

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        # Predict sentiment
        sentiment = predict_sentiment(review_text)
        return render_template('predict.html', review=review_text, sentiment=sentiment)

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
