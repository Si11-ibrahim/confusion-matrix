from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_CATEGORIES = ['Politics', 'Sports', 'Technology']
RANDOM_STATE = 42

def get_training_data():
    """
    Returns the training dataset for the news classifier.
    Contains 50 headlines each for Politics, Sports, and Technology categories.
    """
    
    # Politics headlines (50)
    politics_headlines = [
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
        'Political coalition formed', 'Governor announces stimulus package', 'UN humanitarian aid approved'
    ]
    
    # Sports headlines (50)
    sports_headlines = [
        'Athlete breaks world record', 'Coach announces lineup change', 'Stadium hosts major event',
        'Soccer team advances to playoffs', 'Basketball star signs contract', 'Olympic games begin',
        'Football league standings update', 'Tennis champion wins grand slam', 'Golf tournament concludes',
        'Swimmer sets new Olympic record', 'Baseball team trades player', 'Hockey playoffs start',
        'Cricket world cup final', 'Rugby team wins international match', 'Boxing champion defends title',
        'Formula 1 race winner announced', 'Cycling tour champion crowned', 'Volleyball team qualifies',
        'Basketball finals MVP awarded', 'Soccer transfer record broken', 'Tennis player retires',
        'Golf major tournament begins', 'Swimming world championships', 'Baseball hall of fame inductee',
        'Hockey Stanley Cup winner', 'Cricket team tour announced', 'Rugby world cup host selected',
        'Boxing heavyweight champion', 'Formula 1 constructor championship', 'Cycling Olympic gold medalist',
        'Volleyball world league', 'Basketball draft lottery winner', 'Soccer league table leader',
        'Tennis Australian Open', 'Golf Masters tournament', 'Swimming national records broken',
        'Baseball World Series begins', 'Hockey trade deadline passes', 'Cricket IPL auction',
        'Rugby Six Nations championship', 'Boxing title fight announced', 'Formula 1 testing begins',
        'Cycling Tour de France route', 'Volleyball Olympic qualification', 'Basketball NBA finals',
        'Soccer Champions League draw', 'Tennis Wimbledon champion', 'Golf PGA championship',
        'Swimming Paralympic games', 'Baseball home run record', 'Hockey NHL draft'
    ]
    
    # Technology headlines (50)
    technology_headlines = [
        'Tech company launches product', 'Scientists develop quantum computer', 'App store adds new features',
        'Cybersecurity firm detects threat', 'Social media platform updates algorithm', 'Virtual reality headset announced',
        'Blockchain startup raises funds', 'Cloud computing service expands', 'Mobile app reaches million downloads',
        'Tech giant acquires startup', 'Artificial intelligence breakthrough', 'Smart home device launched',
        'Software company releases update', 'Quantum computing milestone', 'App developers conference begins',
        'Cyber attack thwarted', 'Social network changes privacy policy', 'VR gaming platform announced',
        'Cryptocurrency exchange hacked', 'Cloud storage service upgraded', 'Mobile payment system expanded',
        'Tech IPO valued at billions', 'AI research paper published', 'Smartwatch adds health features',
        'Software vulnerability discovered', 'Social media algorithm update', 'AR glasses development announced',
        'Blockchain technology adopted', 'Cloud gaming service launched', 'Mobile operating system updated',
        'Tech company opens new campus', 'Machine learning model trained', 'Smart device security improved',
        'Software development tools released', 'Quantum encryption breakthrough', 'App marketplace grows',
        'Cybersecurity conference held', 'Social platform adds video features', 'VR content creation tools',
        'Cryptocurrency regulation proposed', 'Cloud infrastructure upgraded', 'Mobile network 5G expanded',
        'Tech startup valued at unicorn', 'AI ethics guidelines released', 'Smart home automation advanced',
        'Software open source project', 'Quantum computing applications', 'App privacy features enhanced',
        'Cyber threat intelligence shared', 'Social media fact checking', 'AR development kit released'
    ]
    
    # Combine all headlines with their categories
    all_headlines = politics_headlines + sports_headlines + technology_headlines
    all_categories = (['Politics'] * len(politics_headlines) + 
                     ['Sports'] * len(sports_headlines) + 
                     ['Technology'] * len(technology_headlines))
    
    return {
        'headline': all_headlines,
        'category': all_categories
    }

def load_model():
    """
    Loads and trains the classification model.
    In production, this should load a pre-trained model from disk.
    """
    print("Loading and training the classification model...")
    
    # Get training data
    data = get_training_data()
    print(f"Training data: {len(data['headline'])} headlines, {len(data['category'])} categories")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Transform headlines to feature vectors
    X = vectorizer.fit_transform(df['headline'])
    y = df['category']
    
    # Train Logistic Regression model
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X, y)
    
    print("Model training completed successfully!")
    return model, vectorizer

# Load the model at startup
print("Initializing News Headline Classifier...")
model, vectorizer = load_model()

# ================================
# ROUTES
# ================================

@app.route('/')
def home():
    """
    Main page route - serves the classification interface.
    """
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to basic HTML if template file is missing
        return '''
        <h1>AI News Classifier</h1>
        <p>Template file not found. Please ensure templates/index.html exists.</p>
        <form>
            <input type="text" id="headline" placeholder="Enter headline...">
            <button type="button" onclick="classify()">Classify</button>
        </form>
        <div id="result"></div>
        <script>
            async function classify() {
                const headline = document.getElementById('headline').value;
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({headline: headline})
                });
                const result = await response.json();
                document.getElementById('result').innerHTML = 
                    `<p>Category: ${result.category}</p><p>Confidence: ${(result.confidence*100).toFixed(1)}%</p>`;
            }
        </script>
        '''

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify():
    """
    Classification API endpoint.
    Accepts POST requests with headline text and returns category prediction.
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return _create_cors_response({})
    
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'headline' not in data:
            return _create_cors_response({'error': 'No headline provided'}, 400)
            
        headline = data['headline'].strip()
        if not headline:
            return _create_cors_response({'error': 'Empty headline provided'}, 400)
            
        print(f"📰 Classifying: '{headline[:50]}{'...' if len(headline) > 50 else ''}'")

        # Vectorize the input headline
        X_input = vectorizer.transform([headline])

        # Make prediction
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = float(max(probabilities))

        # Prepare result
        result = {
            'category': prediction,
            'confidence': confidence,
            'headline': headline[:100]  # Return truncated headline for confirmation
        }
        
        print(f"✅ Result: {prediction} ({confidence:.1%} confidence)")
        return _create_cors_response(result)
        
    except Exception as e:
        error_msg = f"Classification error: {str(e)}"
        print(f"❌ {error_msg}")
        return _create_cors_response({'error': error_msg}, 500)

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return _create_cors_response({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'categories': MODEL_CATEGORIES
    })

@app.route('/stats')
def get_stats():
    """
    Returns model statistics and information.
    """
    training_data = get_training_data()
    return _create_cors_response({
        'model_type': 'Logistic Regression',
        'vectorizer_type': 'TF-IDF',
        'categories': MODEL_CATEGORIES,
        'training_samples': len(training_data['headline']),
        'samples_per_category': len(training_data['headline']) // len(MODEL_CATEGORIES)
    })

# ================================
# UTILITY FUNCTIONS
# ================================

def _create_cors_response(data, status_code=200):
    """
    Creates a JSON response with CORS headers.
    """
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response, status_code

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting AI News Classifier Server")
    print("="*50)
    print(f"📊 Model: Logistic Regression")
    print(f"🔤 Vectorizer: TF-IDF")
    print(f"📂 Categories: {', '.join(MODEL_CATEGORIES)}")
    print(f"🌐 Access: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=True
    )