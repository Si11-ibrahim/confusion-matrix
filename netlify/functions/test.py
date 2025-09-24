import json

def main(event, context):
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
    }
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            'message': 'Simple test function working!',
            'method': event.get('httpMethod', 'UNKNOWN'),
            'path': event.get('path', 'UNKNOWN'),
            'body': event.get('body', 'NO BODY')[:100]  # First 100 chars
        })
    }

# Netlify needs this as the main entry point
handler = main