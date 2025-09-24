const { spawn } = require('child_process');
const path = require('path');

// Training data - same as Python version
const trainingData = {
  headlines: [
    // Politics headlines (50)
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
    // Sports headlines (50)
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
    // Technology headlines (50)
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
  categories: ['Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
               'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
               'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
               'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
               'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
               'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
               'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
               'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
               'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
               'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
               'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology',
               'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology',
               'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology',
               'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology',
               'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology']
};

// Simple TF-IDF implementation
function calculateTFIDF(text, documents) {
  const words = text.toLowerCase().split(/\W+/).filter(word => word.length > 0);
  const tfidf = {};

  words.forEach(word => {
    if (!tfidf[word]) {
      // Calculate TF (term frequency in this document)
      const tf = words.filter(w => w === word).length / words.length;

      // Calculate IDF (inverse document frequency)
      const docsWithWord = documents.filter(doc =>
        doc.toLowerCase().includes(word)
      ).length;
      const idf = Math.log(documents.length / (1 + docsWithWord));

      tfidf[word] = tf * idf;
    }
  });

  return tfidf;
}

// Simple classification based on keyword matching
function classifyHeadline(headline) {
  const lowerHeadline = headline.toLowerCase();

  // Politics keywords
  const politicsKeywords = ['government', 'president', 'election', 'congress', 'senator', 'minister', 'parliament', 'political', 'policy', 'law', 'court', 'supreme', 'diplomat', 'treaty', 'summit', 'cabinet', 'opposition', 'constitution', 'vote', 'campaign'];

  // Sports keywords
  const sportsKeywords = ['team', 'player', 'match', 'championship', 'tournament', 'athlete', 'coach', 'stadium', 'soccer', 'football', 'basketball', 'tennis', 'golf', 'swimming', 'olympic', 'cricket', 'rugby', 'boxing', 'formula', 'cycling', 'volleyball', 'hockey', 'mvp', 'league', 'finals'];

  // Technology keywords
  const techKeywords = ['ai', 'artificial', 'intelligence', 'software', 'app', 'tech', 'computer', 'quantum', 'cybersecurity', 'social media', 'virtual reality', 'vr', 'blockchain', 'cryptocurrency', 'cloud', 'mobile', 'smartphone', 'smart', 'device', 'update', 'algorithm', 'machine learning', 'data', 'digital', 'internet'];

  let politicsScore = 0;
  let sportsScore = 0;
  let techScore = 0;

  politicsKeywords.forEach(keyword => {
    if (lowerHeadline.includes(keyword)) politicsScore++;
  });

  sportsKeywords.forEach(keyword => {
    if (lowerHeadline.includes(keyword)) sportsScore++;
  });

  techKeywords.forEach(keyword => {
    if (lowerHeadline.includes(keyword)) techScore++;
  });

  const maxScore = Math.max(politicsScore, sportsScore, techScore);
  const totalScore = politicsScore + sportsScore + techScore;

  let category = 'Technology'; // default
  let confidence = 0.33; // default confidence

  if (maxScore > 0) {
    if (politicsScore === maxScore) {
      category = 'Politics';
      confidence = politicsScore / totalScore;
    } else if (sportsScore === maxScore) {
      category = 'Sports';
      confidence = sportsScore / totalScore;
    } else {
      category = 'Technology';
      confidence = techScore / totalScore;
    }
  }

  return { category, confidence: Math.max(confidence, 0.33) };
}

exports.handler = async (event, context) => {
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
  };

  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    // Parse the request body
    let body;
    try {
      body = JSON.parse(event.body || '{}');
    } catch (e) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Invalid JSON in request body' })
      };
    }

    const headline = body.headline?.trim();

    if (!headline) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'No headline provided' })
      };
    }

    // Classify the headline
    const result = classifyHeadline(headline);

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        category: result.category,
        confidence: result.confidence,
        headline: headline
      })
    };

  } catch (error) {
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: `Server error: ${error.message}`
      })
    };
  }
};