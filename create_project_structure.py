import os

# Define the folder and file structure
structure = {
    "backend": [
        "app.py",
        "config.py",
        "requirements.txt",
        "Dockerfile",
        "api/__init__.py",
        "api/data_fetch.py",
        "api/signal_generation.py",
        "api/trade_execution.py",
        "models/__init__.py",
        "models/training.py",
        "models/prediction.py",
        "models/ensemble.py",
        "models/backtesting.py",
        "indicators/__init__.py",
        "indicators/sma.py",
        "indicators/ema.py",
        "indicators/macd.py",
        "indicators/rsi.py",
        "indicators/others.py",
        "utils/__init__.py",
        "utils/data_cleaning.py",
        "utils/api_utils.py",
        "utils/logger.py",
        "database/__init__.py",
        "database/models.py",
        "database/db_manager.py",
        "telegram/__init__.py",
        "telegram/notifications.py",
        "cache/__init__.py",
        "cache/redis_cache.py",
        "tests/__init__.py",
        "tests/test_data_fetch.py",
        "tests/test_models.py",
        "tests/test_indicators.py",
        "logs/app.log",
    ],
    "frontend": [
        "public/favicon.ico",
        "pages/_app.js",
        "pages/index.js",
        "pages/settings.js",
        "pages/analytics.js",
        "components/Navbar.js",
        "components/Sidebar.js",
        "components/SignalChart.js",
        "components/IndicatorChart.js",
        "components/AlertPopup.js",
        "styles/globals.css",
        "styles/Dashboard.module.css",
        "utils/api.js",
        "Dockerfile",
        "package.json",
        "next.config.js",
    ],
    ".": ["docker-compose.yml", "README.md", ".env"],
}

# Create directories and files
for folder, files in structure.items():
    for file in files:
        # Create the full path
        path = os.path.join(folder, file)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create the file if it doesn't already exist
        with open(path, "w") as f:
            if path.endswith(".py"):
                f.write("# " + file)  # Add a comment in Python files
            elif path.endswith(".md"):
                f.write("# Project Documentation")  # Placeholder for README
            elif path.endswith(".env"):
                f.write("# Environment Variables")  # Placeholder for .env file
            elif path.endswith(".js"):
                f.write("// " + file)  # Add a comment in JavaScript files
            elif path.endswith(".yml"):
                f.write("# Docker Compose Configuration")  # Placeholder for YAML
            elif path.endswith(".log"):
                f.write("")  # Leave log files empty
