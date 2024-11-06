# utils/api_utils.py
# backend/utils/api_utils.py

import time

def rate_limit(request_count, max_requests, period):
    """
    Simple rate-limiting function.
    - request_count: current count of requests made
    - max_requests: maximum requests allowed in the period
    - period: period in seconds
    """
    if request_count >= max_requests:
        print(f"Rate limit reached. Waiting for {period} seconds.")
        time.sleep(period)
        request_count = 0  # Reset count after waiting
    return request_count
