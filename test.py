from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel  # For model reference
from vertexai.rag import (
    import_files,
    LlmParserConfig,
    TransformationConfig,
    ChunkingConfig
)
from vertexai import rag
from google.protobuf import timestamp_pb2


import logging
import time
import psutil
import os
from datetime import datetime
import json
from pathlib import Path


# === COMPREHENSIVE LOGGER SETUP ===
import logging
import time
import psutil
import os
from datetime import datetime
import sys
from pathlib import Path


class ResourceLogger:
  def __init__(self, log_file="rag_import_metrics.log"):
    self.log_file = log_file
    self.start_time = time.time()
    self.process = psutil.Process(os.getpid())

    # ✅ FIXED: UTF-8 encoding for Windows
    self.setup_logging()

  def setup_logging(self):
    """Configure logging with UTF-8 encoding for Windows"""
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.stream.reconfigure(encoding='utf-8')

    # Formatter (plain ASCII emojis only)
    formatter = logging.Formatter(
      '%(asctime)s [%(levelname)s] %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    self.logger = root_logger

  def log_start(self, paths, corpus_name):
    self.logger.info("80" + "=")
    self.logger.info("VERTEX AI RAG IMPORT STARTED")
    self.logger.info(f"Files: {len(paths)} files")
    self.logger.info(f"Corpus: {corpus_name}")
    self.logger.info(f"Initial Memory: {self.get_memory_mb():.1f} MB")
    self.logger.info(f"CPU Usage: {self.get_cpu_percent():.1f}%")
    self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    self.logger.info(f"GCS Paths: {paths}")
    self.logger.info("80" + "=")

  def log_config(self, llm_parser_config, transformation_config):
    self.logger.info("CONFIGURATION:")
    self.logger.info(f"   Model: {llm_parser_config.model_name}")
    self.logger.info(f"   Max Requests/min: {llm_parser_config.max_parsing_requests_per_min}")
    self.logger.info(f"   Chunk Size: {transformation_config.chunking_config.chunk_size}")
    self.logger.info(f"   Chunk Overlap: {transformation_config.chunking_config.chunk_overlap}")

  def log_step(self, step_name, duration=None):
    if duration:
      self.logger.info(f"Step '{step_name}' completed - {duration:.2f}s")
    else:
      self.logger.info(f"Starting: {step_name}")

  def get_memory_mb(self):
    return self.process.memory_info().rss / 1024 / 1024

  def get_cpu_percent(self):
    return self.process.cpu_percent(interval=0.1)

  def log_resources(self, label=""):
    mem = self.get_memory_mb()
    cpu = self.get_cpu_percent()
    elapsed = time.time() - self.start_time
    self.logger.info(f"Resources [{label}]: CPU {cpu:.1f}% | RAM {mem:.1f}MB | Elapsed {elapsed:.1f}s")

  def log_completion(self, response):
    elapsed = time.time() - self.start_time
    self.logger.info("80" + "=")
    self.logger.info("RAG IMPORT COMPLETED")
    self.logger.info("RESULTS:")
    self.logger.info(f"   Files Imported: {getattr(response, 'imported_rag_files_count', 'N/A')}")
    self.logger.info(f"   Operation Name: {getattr(response, 'operation', {}).get('name', 'N/A')}")
    self.logger.info(f"TOTAL TIME: {elapsed:.2f}s ({elapsed / 60:.1f}min)")
    final_mem = self.get_memory_mb()
    self.logger.info(f"Final Memory: {final_mem:.1f}MB")
    self.logger.info("80" + "=")

  def log_error(self, error):
    self.logger.error(f"ERROR: {str(error)}")


# === YOUR ORIGINAL CODE WITH LOGGER ===
logger = ResourceLogger("rag_import_detailed.log")

try:
  # AUTH (Step 1)
  logger.log_step("Authentication setup")
  credentials_dict = {
  "type": "service_account",
  "project_id": "vertexrag-487011",
  "private_key_id": "59e0951056e8099f3186aceac5a8bcfc5bf0d29c",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCjiw7K8BXF2jHe\nI0jy8Puh4gtxPI6Lf7d5OMTo29MOU0DswDGCwebc5BSDYZRUD/Oa/eOJPJe0ko/z\n+J6DehmNDbdFxY8uaSI3qq3Ry3i0T/gRsW481iD1HV1JZ0L05vsk11jxvw/qXVS+\nM731FfFPnS1/pfUyMmSlP4tYpgk3u0JmsB9Wjr+ksMqTatEuqYjcXZCdc7fi0zVW\nWW8PoD6dK+7Bo8pcyopCBjy6JHzNa82ysIEwnM1wq/aaK0E+Vdyx6+QbaRzzl9qM\niEEh3t3//WIxyjFNv98cRzjYJTYkfrK/dug/UHKSwMfxq3p1TF0NW6F+UBWsuTpt\nnpw7JtSzAgMBAAECggEAB6orQdCZvKGax7Xwo84uhOpCrgZwCdKtByXleKJg1GGL\nf+0MBPxQdRbNbVDj8kKjIKb4hISN+Z0K2RzUVQYib21uovr82Gh9/Yzmw5fdKto6\nnp2ptHk2pzY+moOX7EFtEM908DOLq+i+4YEsTHHAUwJjUQG2qakZ9XKyda+ma/Gs\nGKbcOKM8GIWtGoJHNpxKoU0Uj7RK3bDuHGl4DdHS+6MMISf0dvtwrqNHjAUUEZBZ\nlNKz/18038yX5wvQBHLNylWeYR0FBOqyVAvzRadKkI7pvQPwQvAU388rSWCCHbNA\niUhGVtXRgseFOrW8Qzlj/70WLzQnjtxSkljh1IUBVQKBgQDXw6vrPgVZ5GZm4jMo\nvJAysJuGojiI84SCw6Q7GyWDA1hKpBbC9KNOFiwuU/eyzVAG+bNaLNCVa76a1a35\n2ISOYGPUW9QnjBVPFqp+eKHXWwNoVMwWJaCGM97hTnCTab6G6xcdtwhgRSwA4jaK\nBDMG0JYRuWSOVQbe7D9sJjsnVwKBgQDCCmlwWxfGtC6/28ae/r8X8dsjS4xQ0RHJ\nlwkC1vFvbNEeU6LQZzPj29eKWkmepe8XGKW8iG1kJqTJxhjB2q9SeWuqAQ8tAIRQ\n+OfxnlCNeHiYz88d9YtBacIj5pI2xyOAQdMMvgby34Tbi2WFuC5dBPa+IKge8wYL\nGmaMmt9wBQKBgQDXqYtTjSCII1V/jKUaGLABGqm5vrfHm7Bdi/PB6HZsJ5G+uZjO\nsPvx9xOeEuvI2pMdBcURYy3xzEouNVq6GoMUVKA0CL1b8hbygHNWCnmp6hzT0b5U\nfLOgsIQcq+y2S8HW1XC7kNFceIdtMq7U8TGXpDH78VTjN3WwqG8USHpj7QKBgDvB\n5xziAN2B6g/OCEo42/Ls2fbxskHFUTwLFoxYU7xj/7bePPr/fXyD3MpP6fJA2fP7\n9DausTmxqPg22LMCvRGiMSUG9HyAdz9UmGHRxq761fEBtqBcWUmI33Ac9xSFmYpL\nO0rmTs+HRKw0LszSnvyopbCB8CSv5UGQHNraa0tdAoGBAIXK3A5b8Zy6iACqjHGG\nlTh32M6lFhxkPG5g2lIxvr+WLP5oqZoNN0wyqhI09WAF/uQP/jdZVrwiq9nwmsBf\n0ZChc4Tz4H5btSe54KPGYDty0/xVwwHano1kOMZck5B/YW+7lSmm/yT8qKYhZvQa\n4WKJaGg4TLWKgPzGyfGXXkhE\n-----END PRIVATE KEY-----\n",
  "client_email": "rag-517@vertexrag-487011.iam.gserviceaccount.com",
  "client_id": "114132874509045009689",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/rag-517%40vertexrag-487011.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

  credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )

  logger.log_step("Vertex AI initialization")
  vertexai.init(
    project="vertexrag-487011",
    location="europe-west4",
    credentials=credentials
  )
  logger.log_resources("After init")

  # CONFIG (Step 2)
  paths = ["gs://rag_taniant"]
  corpus_name = "projects/vertexrag-487011/locations/europe-west4/ragCorpora/2305843009213693952"

  logger.log_start(paths, corpus_name)

  MODEL_NAME = "projects/vertexrag-487011/locations/europe-west4/publishers/google/models/gemini-2.5-pro"

  start_config = time.time()
  llm_parser_config = LlmParserConfig(
    model_name=MODEL_NAME,
    max_parsing_requests_per_min=10,
    custom_parsing_prompt="""
You are a precise document content extractor for RAG ingestion.

From the provided document:
- Extract all readable text verbatim from text documents or selectable text fields.
- For images, provide a detailed, factual description of visible content, including any text within the image via OCR, captions, labels, diagrams, or key visuals.
- For PDFs or scans appearing as images without selectable text fields, perform OCR to extract all text accurately, preserving layout, headings, tables, and structure.

Output only the extracted text and image descriptions concatenated cleanly, without summaries, interpretations, opinions, or added metadata. Preserve original formatting like paragraphs, lists, and tables.
"""
  )

  transformation_config = TransformationConfig(
    chunking_config=ChunkingConfig(
      chunk_size=1024,
      chunk_overlap=128,
    ),
  )

  logger.log_config(llm_parser_config, transformation_config)
  logger.log_step("Configuration complete", time.time() - start_config)

  # EXECUTE IMPORT (Step 3)
  logger.log_step("Starting rag.import_files")
  logger.log_resources("Before import")

  import_start = time.time()
  response = rag.import_files(
    corpus_name=corpus_name,
    paths=paths,
    llm_parser=llm_parser_config,
    transformation_config=transformation_config
  )

  import_duration = time.time() - import_start

  logger.log_step("rag.import_files completed", import_duration)
  logger.log_resources("After import")

  # RESULTS (Step 4)
  logger.log_completion(response)

  # POLL OPERATION STATUS (Bonus monitoring)
  logger.log_step("Monitoring operation progress")
  operation_name = getattr(response, 'operation', {}).get('name', '')
  if operation_name:
    logger.info(f"📋 Operation: {operation_name}")
    # Note: Add polling logic here if needed
    # while not response.done():
    #     time.sleep(30)
    #     logger.log_resources("Operation poll")

except Exception as e:
  logger.log_error(e)
  raise

print("✅ Full logging saved to: rag_import_detailed.log")
