from .app import TradingApp
import signal
import logging

def create_app():
    return TradingApp()

if __name__ == '__main__':
    try:
        app = create_app()
        signal.signal(signal.SIGINT, lambda s, f: app.cleanup())
        signal.signal(signal.SIGTERM, lambda s, f: app.cleanup())
        app.run()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
