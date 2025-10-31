from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    source = request.form['source']
    message = request.form['message']
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    result = "🚫 Spam Message" if prediction == 1 else "✅ Not Spam"

    return render_template('index.html', prediction=result, message=message, source=source)

if __name__ == '__main__':
    app.run(debug=True)
