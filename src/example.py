# Wrong - returns None implicitly
async def wrong_function():
    value = 42

# Correct - returns an awaitable
async def correct_function():
    value = 42
    return value
