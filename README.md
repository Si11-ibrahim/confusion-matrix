# News Headline Classifier

A machine learning web application that classifies news headlines into three categories: Politics, Sports, and Technology.

## Features

- **High Accuracy**: ~90% classification accuracy
- **Real-time Classification**: Instant results using TF-IDF vectorization and Logistic Regression
- **Modern UI**: Clean, responsive web interface
- **Serverless**: Deployed on Netlify Functions for scalability

## Demo

Try the live demo: [Your Netlify URL]

## Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open http://localhost:5000 in your browser

## Deployment to Netlify

1. **Connect to Netlify**:
   - Sign up for a free account at [netlify.com](https://netlify.com)
   - Connect your GitHub repository or drag & drop the project files

2. **Automatic Deployment**:
   - Netlify will automatically detect the `netlify.toml` configuration
   - The site will build and deploy automatically

3. **Manual Deployment** (if needed):
   - Install Netlify CLI: `npm install -g netlify-cli`
   - Login: `netlify login`
   - Deploy: `netlify deploy --prod`

## Project Structure

```
├── index.html              # Frontend interface
├── app.py                  # Flask development server
├── requirements.txt        # Python dependencies
├── netlify.toml           # Netlify configuration
├── netlify/
│   └── functions/
│       └── classify.py    # Serverless classification function
└── txt_classification_pipeline.py  # Original ML pipeline
```

## API Usage

The classification API accepts POST requests to `/.netlify/functions/classify`:

```json
{
  "headline": "Your news headline here"
}
```

Response:
```json
{
  "category": "Politics",
  "confidence": 0.95
}
```

## Model Details

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Training Data**: 150 labeled headlines (50 per category)
- **Accuracy**: ~90% on test set

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask (development), Netlify Functions (production)
- **ML**: scikit-learn, pandas
- **Deployment**: Netlify

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License