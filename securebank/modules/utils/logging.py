import os
import time
import json


def log_request(log_dir, endpoint, request, status, output=None, error=None):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_entry = {
        'timestamp': timestamp,
        'endpoint': endpoint,
        'request_method': request.method,
        'request_data': request.get_json() if request.is_json else {},
        'status': status,
        'output': output,
        'error': error
    }
    log_filename = f"log_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w') as log_file:
        json.dump(log_entry, log_file, indent=2)