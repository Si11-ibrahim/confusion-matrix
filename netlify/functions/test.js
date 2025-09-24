exports.handler = async (event, context) => {
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
  };

  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify({
      message: 'Simple test function working!',
      method: event.httpMethod || 'UNKNOWN',
      path: event.path || 'UNKNOWN',
      body: (event.body || 'NO BODY').substring(0, 100)  // First 100 chars
    })
  };
};