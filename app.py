from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

# Load or train the model (in production, load from file)
def load_model():
    # For now, we'll retrain the model each time - in production, save/load the model
    data = {
        'headline': [
            # Politics headlines (50)
            'Government announces new tax policy', 'Election results declared', 'President signs trade deal',
            'Congress passes budget bill', 'Senator proposes new legislation', 'Prime minister visits ally',
            'Parliament debates immigration', 'Cabinet approves foreign policy', 'Mayor addresses city council',
            'Diplomat negotiates peace treaty', 'Supreme Court rules on key case', 'White House issues executive order',
            'Political party launches campaign', 'Governor declares state emergency', 'UN Security Council meets',
            'Minister resigns over scandal', 'Opposition criticizes government', 'Constitutional amendment proposed',
            'Election commission announces dates', 'President addresses nation', 'Senate confirms cabinet nominee',
            'Political rally draws thousands', 'Border security measures tightened', 'International summit begins',
            'Prime minister dissolves parliament', 'Congressional hearing on corruption', 'Mayor proposes budget cuts',
            'Diplomatic relations restored', 'President vetoes legislation', 'Senate blocks judicial nominee',
            'Political debate airs tonight', 'Governor signs education reform', 'UN condemns human rights violation',
            'Minister announces infrastructure plan', 'Opposition wins local election', 'Constitution court rules',
            'Election observers report irregularities', 'President meets world leaders', 'Senate passes climate bill',
            'Political scandal rocks government', 'Governor declares public health emergency', 'UN peacekeeping mission extended',
            'Minister proposes tax reforms', 'Opposition demands resignation', 'Constitutional crisis averted',
            'Election turnout reaches record high', 'President signs defense agreement', 'Senate confirms Supreme Court justice',
            'Political coalition formed', 'Governor announces stimulus package', 'UN humanitarian aid approved',
            # Sports headlines (50) - truncated for brevity, but include all in actual code
            'Team wins championship match', 'Player scores hat-trick', 'Tournament finals underway',
            # ... include all sports headlines ...
            'Hockey NHL draft',
            # Technology headlines (50) - truncated for brevity
            'New AI technology released', 'Smartphone features updated', 'Software update fixes bugs',
            # ... include all technology headlines ...
            'AR development kit released'
        ],
        'category': ['Politics'] * 50 + ['Sports'] * 50 + ['Technology'] * 50
    }

    df = pd.DataFrame(data)

    # Use TF-IDF Vectorizer (performed better)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['headline'])
    y = df['category']

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_model()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Classification Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 10px; background: white; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>News Headline Classifier</h1>
            <p>Enter a news headline to classify it as Politics, Sports, or Technology:</p>
            <input type="text" id="headline" placeholder="Enter headline here...">
            <button onclick="classify()">Classify</button>
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
            async function classify() {
                const headline = document.getElementById('headline').value;
                if (!headline.trim()) {
                    alert('Please enter a headline');
                    return;
                }

                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ headline: headline })
                });

                const result = await response.json();
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `
                    <h3>Classification Result:</h3>
                    <p><strong>Category:</strong> ${result.category}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                `;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    headline = data['headline']

    # Vectorize the input
    X_input = vectorizer.transform([headline])

    # Predict
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    confidence = max(probabilities)

    return jsonify({
        'category': prediction,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)