import sys
import os
import importlib.util
import json

# Load the classify module from the netlify functions directory
spec = importlib.util.spec_from_file_location("classify_func", os.path.join(os.path.dirname(__file__), 'netlify', 'functions', 'classify.py'))
classify_func = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classify_func)
handler = classify_func.handler

# Test event
test_event = {
    'httpMethod': 'POST',
    'body': json.dumps({'headline': 'Government announces new tax policy'})
}

# Test context (mock)
class MockContext:
    pass

try:
    result = handler(test_event, MockContext())
    print('Function test result:')
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f'Error testing function: {e}')
    import traceback
    traceback.print_exc()