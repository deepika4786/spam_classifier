from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        data = vectorizer.transform([email]).toarray()
        prediction = model.predict(data)
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
