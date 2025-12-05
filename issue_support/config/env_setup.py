import os
import logging
import warnings

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT_NAME"] = os.getenv("LANGCHAIN_PROJECT_NAME")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Configure Langfuse for tracing
# Note: We're using Langfuse CallbackHandler for LangChain tracing (see workflow.py)
# OpenTelemetry OTLP export is disabled to avoid conflicts and 404 errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"  # Disable OpenTelemetry SDK to prevent OTLP export errors

os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL")

# Disable all OpenTelemetry exporters to prevent 404 errors
# Langfuse CallbackHandler handles tracing automatically for LangChain operations
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# Suppress OpenTelemetry connection warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)