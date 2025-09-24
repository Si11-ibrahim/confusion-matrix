import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

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
        
        # Sports headlines (50)
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
print(len(data['headline']))
print(len(data['category']))

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)

# Function to train and evaluate
def evaluate_vectorizer(vectorizer, name):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy ({name}): {accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Politics', 'Sports', 'Technology'], yticklabels=['Politics', 'Sports', 'Technology'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()  # Close the figure to free memory

# Evaluate Count Vectorizer
count_vec = CountVectorizer(stop_words='english')
evaluate_vectorizer(count_vec, 'Count Vectorizer')

# Evaluate TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(stop_words='english')
evaluate_vectorizer(tfidf_vec, 'TF-IDF Vectorizer')