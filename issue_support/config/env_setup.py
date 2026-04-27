import os
import logging
import warnings

from dotenv import load_dotenv
load_dotenv()

def _set_if_present(key: str) -> None:
    val = os.getenv(key)
    if val is not None:
        os.environ[key] = val

_set_if_present("GROQ_API_KEY")
_set_if_present("LANGCHAIN_PROJECT_NAME")
_set_if_present("LANGCHAIN_API_KEY")
_set_if_present("LANGFUSE_SECRET_KEY")
_set_if_present("LANGFUSE_PUBLIC_KEY")
_set_if_present("LANGFUSE_BASE_URL")

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Disable all OpenTelemetry exporters to prevent 404 errors
# Langfuse CallbackHandler handles tracing automatically for LangChain operations
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# Suppress OpenTelemetry connection warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)