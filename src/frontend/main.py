import os
import logging
from google.adk.agents import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Google Agent Frontend...")

    # Example initialization of a Google Agent
    # agent = Agent(api_key=os.getenv("GOOGLE_API_KEY"))
    # logger.info(f"Agent initialized: {agent}")

    print("Hello from the Google Agent Frontend!")

if __name__ == "__main__":
    main()
