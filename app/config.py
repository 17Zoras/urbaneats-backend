import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "UrbanEats Backend")
ENV = os.getenv("ENV", "local")
