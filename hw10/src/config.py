from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv())


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/local/sandbox/csci631/datasets"))
