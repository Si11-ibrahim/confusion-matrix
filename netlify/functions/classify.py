import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def handler(event, context):
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Method not allowed'})
        }

    try:
        # Parse the request body
        data = json.loads(event['body'])
        headline = data.get('headline', '')

        if not headline:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'No headline provided'})
            }

        # Load or train the model (in production, this should be cached or loaded from file)
        model, vectorizer = load_model()

        # Vectorize the input
        X_input = vectorizer.transform([headline])

        # Predict
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = float(max(probabilities))

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'category': prediction,
                'confidence': confidence
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }

def load_model():
    # Training data (same as in the main app)
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
            # Sports headlines (50)
            'Team wins championship match', 'Player scores hat-trick', 'Tournament finals underway',
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
            'Swimming Paralympic games', 'Baseball home run record', 'Hockey NHL draft',
            # Technology headlines (50)
            'New AI technology released', 'Smartphone features updated', 'Software update fixes bugs',
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
        ],
        'category': ['Politics'] * 50 + ['Sports'] * 50 + ['Technology'] * 50
    }

    df = pd.DataFrame(data)

    # Use TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['headline'])
    y = df['category']

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    return model, vectorizer