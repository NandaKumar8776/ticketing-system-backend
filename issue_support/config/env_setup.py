import os
import logging
import warnings

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT_NAME"] = os.getenv("LANGCHAIN_PROJECT_NAME")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Disable OpenTelemetry default exporter that tries to connect to localhost:4318
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL")
os.environ["LANGFUSE_PROJECT_NAME"] = os.getenv("LANGFUSE_PROJECT_NAME")

# Suppress OpenTelemetry connection warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)