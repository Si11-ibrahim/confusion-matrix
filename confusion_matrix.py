"""
News Headline Classification Pipeline
=====================================
This script compares Count Vectorizer vs TF-IDF Vectorizer performance
for classifying news headlines into Politics, Sports, and Technology categories.
"""

"""
Text Classification Pipeline: TF-IDF vs Count Vectorizer Comparison
Compares performance of Count Vectorizer and TF-IDF Vectorizer on news headline classification.
Categories: Politics, Sports, Technology
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
import numpy as np

# Configure matplotlib for better display
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("🗞️  NEWS HEADLINE CLASSIFICATION PIPELINE")
print("=" * 60)
print("📊 Comparing Count Vectorizer vs TF-IDF Vectorizer")
print("🎯 Categories: Politics, Sports, Technology")
print("=" * 60)

# Training data
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
print(f"📈 Total headlines: {len(data['headline'])}")
print(f"📈 Total categories: {len(set(data['category']))}")
print(f"📈 Headlines per category: {len(data['headline']) // len(set(data['category']))}")

df = pd.DataFrame(data)
print(f"📊 Dataset shape: {df.shape}")
print("\n📋 Category distribution:")
print(df['category'].value_counts().to_string())

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['category'], test_size=0.2, random_state=42)

# Function to train and evaluate
def evaluate_vectorizer(vectorizer, name):
    """
    Train and evaluate a text vectorizer with logistic regression.
    Shows confusion matrix visualization and returns model performance.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating: {name}")
    print('='*60)
    
    # Transform text data
    print("🔄 Transforming text data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Training features shape: {X_train_vec.shape}")
    print(f"   Testing features shape: {X_test_vec.shape}")
    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Train model
    print("🤖 Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 {name} Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📈 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Create confusion matrix with proper labels
    cm = confusion_matrix(y_test, y_pred, labels=['Politics', 'Sports', 'Technology'])
    
    # Create the plot with better formatting
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with enhanced styling
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Politics', 'Sports', 'Technology'], 
                yticklabels=['Politics', 'Sports', 'Technology'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14})
    
    # Enhanced title and labels
    plt.title(f'Confusion Matrix - {name}\nAccuracy: {accuracy:.2%}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            percentage = cm[i][j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Save the figure
    filename = f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved as: {filename}")
    
    # Force the plot to display
    plt.show(block=False)  # Non-blocking show
    plt.pause(0.1)  # Small pause to ensure rendering
    
    return model

# # Evaluate Count Vectorizer
# print("="*60)
# print("EVALUATING COUNT VECTORIZER")
# print("="*60)
# count_vec = CountVectorizer(stop_words='english')
# count_model = evaluate_vectorizer(count_vec, 'Count Vectorizer')

# # Evaluate TF-IDF Vectorizer
# print("\n" + "="*60)
# print("EVALUATING TF-IDF VECTORIZER")
# print("="*60)
# tfidf_vec = TfidfVectorizer(stop_words='english')
# tfidf_model = evaluate_vectorizer(tfidf_vec, 'TF-IDF Vectorizer')

# Create a side-by-side comparison
print("\n" + "="*60)
print("CREATING COMPARISON VISUALIZATION")
print("="*60)

# Get predictions for both models
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)
y_pred_count = count_model.predict(X_test_count)

X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

# Create side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count Vectorizer confusion matrix
cm_count = confusion_matrix(y_test, y_pred_count)
sns.heatmap(cm_count, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Politics', 'Sports', 'Technology'], 
            yticklabels=['Politics', 'Sports', 'Technology'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Count Vectorizer\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_count)), 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

# TF-IDF Vectorizer confusion matrix  
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Politics', 'Sports', 'Technology'], 
            yticklabels=['Politics', 'Sports', 'Technology'],
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('TF-IDF Vectorizer\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_tfidf)), 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)

plt.suptitle('Confusion Matrix Comparison: Count vs TF-IDF Vectorizers', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save comparison
comparison_filename = 'confusion_matrix_comparison.png'
plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
print(f"Comparison saved as: {comparison_filename}")

# Show comparison
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Count Vectorizer Accuracy: {accuracy_score(y_test, y_pred_count):.2%}")
print(f"TF-IDF Vectorizer Accuracy: {accuracy_score(y_test, y_pred_tfidf):.2%}")

if accuracy_score(y_test, y_pred_tfidf) > accuracy_score(y_test, y_pred_count):
    print("\n✅ TF-IDF Vectorizer performs better!")
elif accuracy_score(y_test, y_pred_count) > accuracy_score(y_test, y_pred_tfidf):
    print("\n✅ Count Vectorizer performs better!")
else:
    print("\n🤝 Both vectorizers perform equally well!")