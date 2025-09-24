import json

def handler(event, context):
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS'
    }
    
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            'status': 'OK',
            'message': 'News Classifier API is running',
            'timestamp': str(context.aws_request_id) if hasattr(context, 'aws_request_id') else 'unknown'
        })
    }