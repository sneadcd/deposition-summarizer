# app.py
# Final Version: Multi-Provider, Two-Model Selection, Permissive Google Safety, Default Temperature, No Overwrite
# Revision: Include custom instructions in Step 2 (Global Context) and update UI guidance for name spellings.

import streamlit as st
import os
import io  # For handling file uploads and downloads in memory
from dotenv import load_dotenv
import string # For fallback text cleaning in DOCX generation
import traceback # For detailed error logging
import requests # For OpenRouter model fetching

# --- Set page config FIRST ---
st.set_page_config(layout="wide", page_title="Deposition Summarizer")

# --- !!! DEFINE OUTPUT DIRECTORY HERE !!! ---
# Use a raw string (r"...") or escaped backslashes for Windows paths
OUTPUT_DIRECTORY = r"D:\OneDrive\OneDrive - Barnes Maloney PLLC\Documents\Depo Summaries"
# OUTPUT_DIRECTORY = "D:\\OneDrive\\OneDrive - Barnes Maloney PLLC\\Documents\\Depo Summaries" # Alternative

# --- Application Metadata for OpenRouter ---
# Optional: Replace with your actual site URL and app name if you deploy publicly
YOUR_SITE_URL = "http://localhost:8501" # Default Streamlit URL
YOUR_APP_NAME = "Deposition Summarizer"
# ---

# --- Load API Keys from .env ---
load_dotenv()
api_keys = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "Google": os.getenv("GOOGLE_API_KEY"),
    "OpenRouter": os.getenv("OPENROUTER_API_KEY"),
}

# --- Import Libraries ---
# LLM Libraries
from openai import OpenAI, OpenAIError
from anthropic import Anthropic, APIError as AnthropicAPIError
import google.generativeai as genai
from google.api_core import exceptions as GoogleAPICoreExceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# File Parsing Libraries
import docx # python-docx library
import fitz # PyMuPDF library, often imported as fitz

# Text Processing (Chunking)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Initialize LLM Clients (if keys exist) ---
# This dictionary will hold initialized clients for available providers
clients = {}
client_load_status = {} # To show status in sidebar

# Initialize OpenAI Client
if api_keys["OpenAI"]:
    try:
        clients["OpenAI"] = OpenAI(api_key=api_keys["OpenAI"])
        client_load_status["OpenAI"] = "✅ Initialized"
    except OpenAIError as e:
        client_load_status["OpenAI"] = f"❌ OpenAI Error: {e}"
    except Exception as e:
        client_load_status["OpenAI"] = f"❌ Unexpected Error: {e}"
else:
    client_load_status["OpenAI"] = "⚠️ Key Missing"

# Initialize Anthropic (Claude) Client
if api_keys["Anthropic"]:
    try:
        clients["Anthropic"] = Anthropic(api_key=api_keys["Anthropic"])
        client_load_status["Anthropic"] = "✅ Initialized"
    except AnthropicAPIError as e:
        client_load_status["Anthropic"] = f"❌ Anthropic Error: {e}"
    except Exception as e:
        client_load_status["Anthropic"] = f"❌ Unexpected Error: {e}"
else:
    client_load_status["Anthropic"] = "⚠️ Key Missing"

# Initialize Google (Gemini) Client
if api_keys["Google"]:
    try:
        # Configure the genai library with the API key
        genai.configure(api_key=api_keys["Google"])
        # Store the configured genai module itself as the "client"
        clients["Google"] = genai
        client_load_status["Google"] = "✅ Initialized"
        # Optional: Test configuration by trying to list models
        # try:
        #     models_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        #     if not models_list: client_load_status["Google"] = "⚠️ Initialized, but no usable models listed."
        # except Exception as list_e:
        #      client_load_status["Google"] = f"⚠️ Initialized, but failed to list models: {list_e}"
    except GoogleAPICoreExceptions.GoogleAPIError as e:
         client_load_status["Google"] = f"❌ Google API Config Error: {e}"
    except Exception as e: # Catch potential auth errors during configure
        client_load_status["Google"] = f"❌ Google Init Error: {e}"
else:
    client_load_status["Google"] = "⚠️ Key Missing"

# Initialize OpenRouter Client (using OpenAI SDK structure)
if api_keys["OpenRouter"]:
    try:
        clients["OpenRouter"] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_keys["OpenRouter"],
            # Add headers recommended by OpenRouter
            default_headers={
                 "HTTP-Referer": YOUR_SITE_URL,
                 "X-Title": YOUR_APP_NAME,
            },
        )
        client_load_status["OpenRouter"] = "✅ Initialized"
    except OpenAIError as e:
         client_load_status["OpenRouter"] = f"❌ OpenRouter Config Error: {e}"
    except Exception as e:
         client_load_status["OpenRouter"] = f"❌ Unexpected OpenRouter Init Error: {e}"
else:
    client_load_status["OpenRouter"] = "⚠️ Key Missing"


# --- Model Mapping (Define known models for selection) ---
# This provides default options; OpenRouter list is fetched dynamically.
# Updated Anthropic models based on user provided info (using specific versions)
MODEL_MAPPING = {
    "OpenAI": ["o3-mini", "gpt-4o", "o1-mini"], # Updated OpenAI Models
    "Anthropic": [
        "claude-3-7-sonnet-20250219", # Latest Sonnet (as of provided info)
        "claude-3-5-sonnet-20241022", # Latest v2 3.5 Sonnet
        "claude-3-5-haiku-20241022",  # Latest 3.5 Haiku
        "claude-3-opus-20240229",   # Opus still relevant
    ],
    # Updated Google Models based on provided info + existing stable ones
    "Google": [
        "gemini-2.5-pro-exp-03-25",     # New experimental model (DEFAULT)
        "models/gemini-1.5-pro-latest", # Previous Pro latest
        "gemini-2.0-flash",             # New Flash model
        "models/gemini-1.5-flash-latest",# Previous Flash latest
        "gemini-2.0-flash-lite",        # New Flash Lite
        "gemini-1.5-flash-8b",          # New Flash 8B
        "models/gemini-1.0-pro"         # Original Pro
    ],
    "OpenRouter": ["Fetching..."] # Placeholder, will be populated by fetch_openrouter_models
}
# Shared cache for fetched OpenRouter models to avoid redundant API calls
OPENROUTER_MODELS_CACHE = None

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache fetched models for 1 hour
def fetch_openrouter_models(api_key):
    """Fetches the list of available models from the OpenRouter API."""
    global OPENROUTER_MODELS_CACHE
    # Return immediately if cache is already populated (and not in an error state)
    if OPENROUTER_MODELS_CACHE is not None and (not OPENROUTER_MODELS_CACHE or not OPENROUTER_MODELS_CACHE[0].startswith("Error:")):
        return OPENROUTER_MODELS_CACHE

    # Proceed with fetching if no valid cache
    if not api_key:
        return ["Error: OpenRouter API Key missing"]

    st.info("Fetching OpenRouter models list...") # Provide user feedback
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=20) # Increased timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        models_data = response.json().get("data", [])

        # Extract model IDs, potentially add filtering here later (e.g., by context length, modality)
        model_ids = sorted([model.get("id") for model in models_data if model.get("id")])

        if not model_ids:
            result = ["Error: No models found in OpenRouter response"]
        else:
            result = model_ids
        st.info(f"OpenRouter models list fetched ({len(result)} models).")
        OPENROUTER_MODELS_CACHE = result # Update cache
        return result

    except requests.exceptions.Timeout:
        st.error("Failed to fetch OpenRouter models: Request timed out.")
        result = ["Error: Request Timed Out"]
        OPENROUTER_MODELS_CACHE = result # Cache error state
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch OpenRouter models: Network error - {e}")
        result = [f"Error: Connection Failed"]
        OPENROUTER_MODELS_CACHE = result
        return result
    except Exception as e: # Catch JSON parsing errors or other issues
        st.error(f"Error processing OpenRouter models response: {e}")
        result = [f"Error: Processing Failed"]
        OPENROUTER_MODELS_CACHE = result
        return result

def load_instruction(file_path):
    """Loads text content from a specified instruction file."""
    try:
        normalized_path = os.path.normpath(file_path) # Normalize path for OS compatibility
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"Instruction file not found: {normalized_path}")
        return "" # Return empty string if file not found
    except Exception as e:
        st.error(f"Error reading instruction file {normalized_path}: {e}")
        return "" # Return empty string on other errors

def call_llm(prompt, provider, model_name): # Removed temperature parameter
    """
    Handles the API call to the selected LLM provider and model using default temperature.
    Returns the generated text content or an error string starting with "Error:".
    """
    st.write(f"Attempting LLM call: Provider='{provider}', Model='{model_name}' (Using Default Temperature)") # Debugging output

    # --- Input Validation ---
    if not provider or not model_name:
        st.error("LLM Call Error: Provider or model name missing.")
        return "Error: Provider or model not specified."

    client = clients.get(provider)
    if not client:
        st.error(f"LLM Call Error: Client for '{provider}' not available. Check API key and initialization status in the sidebar.")
        return f"Error: Client for {provider} not initialized."

    # Prevent attempts with invalid model selections (e.g., during fetching or if fetch failed)
    if model_name.startswith("Error:") or model_name == "Fetching...":
         st.error(f"LLM Call Error: Invalid model selected for {provider}: {model_name}")
         return f"Error: Invalid model '{model_name}' selected."

    # --- API Call Logic ---
    try:
        # --- OpenAI and OpenRouter --- (Share similar API structure via OpenAI SDK)
        if provider == "OpenAI" or provider == "OpenRouter":
            # The client object is already configured with base_url and api_key
            response = client.chat.completions.create(
                model=model_name, # For OpenRouter, expects IDs like 'openai/gpt-4o'
                messages=[{"role": "user", "content": prompt}]
                # REMOVED: temperature parameter - uses model/provider default
                # Note: No explicit max_tokens by default, uses model's default or OpenRouter's handling
            )
            result = response.choices[0].message.content
            st.write(f"LLM call successful ({provider})") # Debugging success
            return result.strip() if result else ""

        # --- Anthropic (Claude) ---
        elif provider == "Anthropic":
            # Heuristic to separate potential system prompt from user prompt
            system_prompt = ""
            user_prompt = prompt
            prompt_lines = prompt.split('\n', 1)
            # Check if the first part seems like instructions
            if len(prompt_lines) > 1 and len(prompt_lines[0]) < 300 and \
               any(keyword in prompt_lines[0].lower() for keyword in ["instructions", "role", "task", "goal"]):
                system_prompt = prompt_lines[0]
                user_prompt = prompt_lines[1]

            # Anthropic API requires max_tokens - Use model defaults (e.g. 4096/8192) or higher if needed.
            # Keeping 4096 for now as a safe default unless user needs longer output.
            # Note: claude-3.7-sonnet can handle much larger outputs (64k/128k with headers/batch)
            max_output_tokens = 4096 # Default, adjust if needed
            if "claude-3-5" in model_name or "claude-3-7" in model_name:
                max_output_tokens = 8192 # Use higher default for 3.5/3.7 models

            response = client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens, # Adjust if needed, models have different limits
                system=system_prompt if system_prompt else None, # Pass system prompt if identified
                messages=[{"role": "user", "content": user_prompt}]
                # REMOVED: temperature parameter - uses model/provider default
            )

            # Handle potential content filtering or other stop reasons
            if response.stop_reason == "max_tokens":
                 st.warning(f"Anthropic response stopped due to max_tokens ({max_output_tokens}). Output might be truncated.")
            # elif response.stop_reason == "stop_sequence": pass # Normal stop
            # elif response.stop_reason: # Log other unexpected stop reasons
            #     st.warning(f"Anthropic response stopped unexpectedly. Reason: {response.stop_reason}")

            # Extract text content safely from potentially multiple blocks
            result_text = ""
            if isinstance(response.content, list):
                 result_text = "".join([block.text for block in response.content if hasattr(block, 'text')])
            elif hasattr(response.content, 'text'): # Check if response itself has text (older API?)
                 result_text = response.content.text
            else: # Fallback if structure is completely different
                 st.warning("Unexpected Anthropic response content structure.")
                 result_text = str(response.content)

            st.write(f"LLM call successful ({provider})") # Debugging success
            return result_text.strip() if result_text else ""

        # --- Google (Gemini) ---
        elif provider == "Google":
            # Define safety settings to be highly permissive (BLOCK_NONE)
            # Useful for legal documents that might contain sensitive topics.
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            # st.info("Applying BLOCK_NONE safety settings for Google API call.") # Optional info message

            # REMOVED: Generation config for temperature

            # Initialize the specific model with safety settings
            # The 'client' here is the configured 'genai' module
            model = client.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings
                # REMOVED: generation_config - uses model default temperature
                # Note: No explicit max_output_tokens here, uses model's default
            )

            # Make the API call
            response = model.generate_content(prompt)

            # --- Handle Google Response ---
            # Check if generation was blocked or failed (even with BLOCK_NONE, severe issues might block)
            # Need to access prompt_feedback safely as it might not exist on all errors
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            block_reason = getattr(prompt_feedback, 'block_reason', None) if prompt_feedback else None

            if block_reason:
                 safety_ratings = getattr(prompt_feedback, 'safety_ratings', [])
                 st.error(f"Google API call blocked. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
                 try: st.warning(f"Full Google Response (Debug): {response}")
                 except Exception: st.warning("Could not display full Google response object.")
                 return f"Error: Google API Blocked Prompt ({block_reason})"

            # If not blocked, try to extract text (might fail if no candidates or other issues)
            try:
                # Accessing response.text can raise ValueError if content is blocked or missing
                result = response.text
            except ValueError as ve:
                 # Check candidates array for more info if .text fails
                 candidates = getattr(response, 'candidates', [])
                 if candidates:
                     st.warning(f"Google response generated candidate(s) but accessing '.text' failed ({ve}). Candidates: {candidates}")
                     # Attempt to extract from parts as fallback
                     if candidates[0].content and candidates[0].content.parts:
                         result = "".join([part.text for part in candidates[0].content.parts if hasattr(part, 'text')])
                         if result: return result.strip() # Return if extraction successful
                 # If no candidates or parts extraction failed
                 st.error(f"Google API call failed: Could not extract text content. Error: {ve}")
                 return f"Error: Google response missing text content ({ve})."
            except Exception as e_text: # Catch other potential errors accessing .text
                 st.error(f"Unexpected error accessing Google response text: {e_text}")
                 return f"Error: Failed to access Google response text ({e_text})."

            st.write(f"LLM call successful ({provider})") # Debugging success
            return result.strip() if result else ""

        # --- Unknown Provider ---
        else:
            st.error(f"LLM Call Error: Provider '{provider}' logic not implemented.")
            return f"Error: Unknown provider {provider}."

    # --- Exception Handling ---
    except OpenAIError as e:
        st.error(f"API Error ({provider} via OpenAI SDK): {e}")
        error_detail = str(e).lower()
        if "context_length_exceeded" in error_detail: return f"Error: Context length exceeded for {model_name}."
        if "model_not_found" in error_detail: return f"Error: Model not found: {model_name}."
        if "authentication" in error_detail: return f"Error: Authentication failed for {provider}. Check API Key."
        return f"Error: OpenAI/OR Error - {e}" # More specific return
    except AnthropicAPIError as e:
        st.error(f"API Error (Anthropic): {e}")
        error_detail = str(e).lower()
        if "authentication" in error_detail: return "Error: Authentication failed for Anthropic. Check API Key."
        # Add check for quota errors if known
        if "quota" in error_detail or "rate_limit" in error_detail: return f"Error: Anthropic Quota/Rate Limit Exceeded - {e}"
        if "invalid_request" in error_detail: return f"Error: Invalid Request for Anthropic model {model_name} - {e}"
        return f"Error: Anthropic Error - {e}"
    except GoogleAPICoreExceptions.PermissionDenied as e:
         st.error(f"API Error (Google): Permission Denied. Check API Key & API enabled status. Details: {e}")
         return f"Error: Google Permission Denied."
    except GoogleAPICoreExceptions.ResourceExhausted as e:
        st.error(f"API Error (Google): Resource Exhausted (Quota Limit?). Details: {e}")
        return f"Error: Google Quota Exceeded."
    except GoogleAPICoreExceptions.InvalidArgument as e:
         st.error(f"API Error (Google): Invalid Argument. Check model name ('{model_name}')/prompt. Details: {e}")
         return f"Error: Google Invalid Argument."
    except GoogleAPICoreExceptions.GoogleAPIError as e: # Catch other specific Google errors
        st.error(f"API Error (Google): {e}")
        return f"Error: Google API Error - {e}"
    except requests.exceptions.RequestException as e: # Catch network errors (e.g., for OpenRouter, or potentially others if they use requests)
         st.error(f"Network Error for {provider}: {e}")
         return f"Error: Network Connection Failed ({provider})."
    except Exception as e: # Catch-all for anything else
        st.error(f"An unexpected error occurred during the LLM call ({provider} - {model_name}): {e}")
        st.error(traceback.format_exc()) # Provide full traceback in logs/UI
        return f"Unexpected Error ({provider}) - {e}"


def parse_uploaded_file(uploaded_file):
    """Parses the uploaded file (.txt, .pdf, .docx) and extracts text content."""
    if uploaded_file is None:
        return None, "No file uploaded."

    file_name = uploaded_file.name
    file_type = ""
    if '.' in file_name:
        file_type = file_name.split('.')[-1].lower()

    st.info(f"Parsing file: '{file_name}' (Detected type: '{file_type}')")

    try:
        # Read file content into bytes
        bytes_data = uploaded_file.getvalue()
        if not bytes_data:
             # It's possible for an empty file to be uploaded
             st.warning(f"Uploaded file '{file_name}' appears to be empty.")
             return "", None # Return empty string, not an error, let downstream handle

        # --- PDF Parsing ---
        if file_type == 'pdf':
            text = ""
            is_ocr_needed = False # Flag if OCR might be required
            try:
                # Open PDF from bytes
                with fitz.open(stream=bytes_data, filetype="pdf") as doc:
                    if not doc.page_count:
                         st.warning(f"PDF file '{file_name}' has zero pages.")
                         return "", None # Empty document

                    # Iterate through pages
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        # Try extracting text with layout preservation first
                        page_text_sorted = page.get_text("text", sort=True).strip()
                        # Fallback to simpler extraction if sorted is empty
                        page_text_simple = page.get_text("simple").strip()
                        # Use sorted if available, otherwise simple
                        page_text = page_text_sorted if page_text_sorted else page_text_simple

                        # If still no text, check if the page contains images (potential OCR needed)
                        if not page_text and page.get_images(full=True):
                             is_ocr_needed = True # Mark that OCR might be needed for the document

                        # Append extracted text for the page
                        if page_text:
                             text += page_text + "\n"

                        # Add a page break marker (optional, can help structure)
                        if page_num < len(doc) - 1:
                              text += "\n--- Page Break ---\n"

            except fitz.fitz.FileDataError as fe:
                 # Error specifically for corrupted/password-protected/invalid PDFs
                 return None, f"Error parsing PDF '{file_name}': File may be corrupt, password protected, or not a valid PDF format. Details: {fe}"
            except Exception as fitz_e:
                 # Catch other unexpected errors during PyMuPDF processing
                 return None, f"Unexpected error during PDF processing: {fitz_e}\n{traceback.format_exc()}"

            # After processing all pages, check final result
            # Remove page breaks for the empty check
            meaningful_text_check = text.replace("--- Page Break ---", "").strip()
            if not meaningful_text_check:
                 if is_ocr_needed:
                      # If no text was extracted AND we detected images, OCR is likely needed
                      return None, f"Failed to extract text from PDF: '{file_name}'. The document appears to be image-based and requires OCR (Optical Character Recognition), which this tool does not perform."
                 else:
                      # If no text and no images detected, it might be truly empty or unusually formatted
                      st.warning(f"Parsed PDF '{file_name}', but no text content was extracted. The file might be empty or have an unusual structure.")
                      return "", None # Return empty string, not an error

            # If text was extracted
            st.success(f"Successfully parsed PDF: '{file_name}'")
            return text, None

        # --- DOCX Parsing ---
        elif file_type == 'docx':
            try:
                # Use BytesIO to read the uploaded file in memory
                doc_io = io.BytesIO(bytes_data)
                doc = docx.Document(doc_io)
                # Extract text from paragraphs, joining with newlines
                # Filter out paragraphs that are purely whitespace
                paragraphs_text = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
                text = "\n".join(paragraphs_text)

                if not text.strip():
                     st.warning(f"DOCX file '{file_name}' parsed, but contained no text in its paragraphs.")
                     return "", None # Return empty string

                st.success(f"Successfully parsed DOCX: '{file_name}'")
                return text, None
            except Exception as docx_e:
                 # Catch errors during docx parsing (e.g., corrupted file, invalid XML)
                 return None, f"Error opening or parsing DOCX file '{file_name}'. It might be corrupted or not a valid DOCX format. Error: {docx_e}\n{traceback.format_exc()}"

        # --- TXT Parsing ---
        elif file_type == 'txt':
            try:
                # Try decoding as UTF-8 first (most common)
                text = bytes_data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    # Fallback to CP1252 (common Windows encoding)
                    text = bytes_data.decode("cp1252")
                    st.warning("Decoded TXT file using CP1252 encoding (UTF-8 failed). Check for garbled characters.")
                except Exception as e_decode:
                   # If both decodings fail
                   st.error(f"Failed to decode TXT file '{file_name}' with UTF-8 or CP1252: {e_decode}")
                   return None, f"TXT decode error: Could not decode file content ({e_decode})."

            # Check if decoded text is empty or just whitespace
            if not text.strip():
                 st.warning(f"TXT file '{file_name}' parsed, but was empty or contained only whitespace.")
                 return "", None # Return empty string

            st.success(f"Successfully parsed TXT: '{file_name}'")
            return text, None

        # --- Unsupported File Type ---
        else:
            error_msg = f"Unsupported file type: '.{file_type}'. Please upload a file with a .txt, .pdf, or .docx extension."
            st.error(error_msg)
            return None, error_msg

    except Exception as e: # Catch-all for unexpected errors during file handling/parsing
        error_msg = f"An unexpected error occurred during file parsing for '{file_name}': {e}\n{traceback.format_exc()}"
        st.error(error_msg)
        return None, error_msg


def chunk_text(text, chunk_size=8000, chunk_overlap=400):
    """Splits the text into overlapping chunks for LLM processing."""
    st.info(f"Chunking text using RecursiveCharacterTextSplitter (Size: {chunk_size}, Overlap: {chunk_overlap})...")

    # Handle empty or whitespace-only input gracefully
    if not text or text.isspace():
        st.warning("Input text for chunking is empty or only whitespace.")
        return [] # Return empty list

    try:
        # Initialize the splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,         # Target size of each chunk
            chunk_overlap=chunk_overlap,   # Number of characters overlap between chunks
            length_function=len,           # Use standard character length function
            is_separator_regex=False,      # Treat separators literally
            # Define separators to split on, in order of preference
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ",", " ", ""],
            keep_separator=True            # Keep separators at the end of chunks where possible
        )

        # Perform the splitting
        chunks = text_splitter.split_text(text)

        # Filter out any chunks that might be purely whitespace after splitting
        # (This can sometimes happen depending on separators and input)
        chunks = [chunk for chunk in chunks if chunk and not chunk.isspace()]

        # Handle cases where splitting results in no chunks (e.g., text smaller than overlap?)
        if not chunks:
            st.warning("Text chunking resulted in zero non-whitespace chunks.")
            # If original text was valid, return it as a single chunk to avoid losing data
            if text and not text.isspace():
                st.info("Returning original text as a single chunk.")
                return [text]
            else:
                return [] # Return empty list if original text was also invalid

        # Report success
        st.write(f"Text divided into {len(chunks)} chunks.")
        return chunks

    except Exception as e: # Catch any unexpected errors during chunking
        st.error(f"An error occurred during text chunking: {e}")
        st.error(traceback.format_exc()) # Log detailed error
        return [] # Return empty list on failure


def generate_docx_bytes(content_dict):
    """
    Generates a DOCX file's content in memory as bytes from a dictionary.
    Keys become headings (Level 1), values become paragraphs.
    Includes improved character cleaning for XML validity.
    """
    try:
        # Create a new Word document object
        document = docx.Document()

        # Iterate through the dictionary items (heading: paragraph_text)
        for title, text in content_dict.items():
            # Ensure title and text are strings, provide defaults if None
            title_str = str(title) if title is not None else "Untitled Section"
            text_str = str(text) if text is not None else ""

            # --- Clean Title (Simpler cleaning for titles) ---
            # Remove non-printable characters for the heading
            clean_title = ''.join(filter(lambda x: x in string.printable, title_str)).strip()
            # Use a default title if cleaning results in an empty string
            if not clean_title: clean_title = "Untitled Section"

            # --- Clean Paragraph Text (More robust XML cleaning) ---
            cleaned_text = ""
            for char in text_str:
                 codepoint = ord(char)
                 # Allow Tab, Newline, Carriage Return
                 if codepoint in (0x9, 0xA, 0xD):
                      cleaned_text += char
                 # Allow valid XML character ranges (excluding control chars except the ones above)
                 # Reference: https://www.w3.org/TR/xml/#charsets
                 elif (0x20 <= codepoint <= 0xD7FF) or \
                      (0xE000 <= codepoint <= 0xFFFD) or \
                      (0x10000 <= codepoint <= 0x10FFFF):
                      cleaned_text += char
                 # else: # Character is invalid for XML 1.0
                 #     # Option 1: Skip the character (do nothing)
                 #     # Option 2: Replace with a placeholder (e.g., '?')
                 #     # cleaned_text += '?'
                 #     pass # Currently skipping

            # --- Add content to document ---
            try:
                # Add the cleaned heading
                document.add_heading(clean_title, level=1)
                # Add the cleaned paragraph text
                document.add_paragraph(cleaned_text)
            except ValueError as ve:
                # This might catch rare cases python-docx still struggles with
                st.warning(f"Problem adding content for section '{clean_title}'. Applying basic printable filter as fallback. Error: {ve}")
                # Extremely basic fallback: filter to just printable ASCII chars
                fallback_text = ''.join(filter(lambda x: x in string.printable, cleaned_text))
                # Add a note indicating potential modification
                document.add_paragraph(fallback_text + "\n\n[Content possibly modified due to invalid characters detected by DOCX library]")

        # --- Save document to memory ---
        # Create an in-memory stream
        bio = io.BytesIO()
        # Save the document to the stream
        document.save(bio)
        # Reset the stream's position to the beginning
        bio.seek(0)
        # Return the byte content of the stream
        return bio.getvalue()

    except Exception as e: # Catch any errors during document creation/saving
        st.error(f"Error generating DOCX file in memory: {e}")
        st.error(traceback.format_exc()) # Log detailed error
        return None # Indicate failure


def save_bytes_to_file(file_bytes, full_path):
    """
    Saves byte content (like a generated DOCX) to a specified file path.
    Avoids overwriting by appending a counter (e.g., _(1), _(2)) if the file exists.
    Returns the final path used for saving, or None if saving fails.
    """
    # Check if input bytes are valid
    if file_bytes is None:
        st.error(f"Attempted to save 'None' to {os.path.basename(full_path)}. Document generation likely failed.")
        return None # Indicate failure

    try:
        # Ensure the target directory exists, create it if necessary
        dir_name = os.path.dirname(full_path)
        if dir_name: # Avoid trying to create directory if path is just a filename
             os.makedirs(dir_name, exist_ok=True)

        # --- Check for existing file and find unique name ---
        base, ext = os.path.splitext(full_path)
        counter = 1
        final_path = full_path
        # Loop while the path exists to find a unique name
        while os.path.exists(final_path):
            # Construct new filename with counter like base_(1).ext, base_(2).ext etc.
            final_path = f"{base}_({counter}){ext}"
            counter += 1
        # --- End check ---

        # Write the bytes to the file in binary write mode ('wb') using the unique path
        with open(final_path, 'wb') as f:
            f.write(file_bytes)

        # Report success using the final path
        st.success(f"✔️ File successfully saved to: {final_path}") # Use final_path
        return final_path # Return the actual path used

    except PermissionError:
        # Specific error for lack of write permissions
        st.error(f"Permission denied: Cannot write to the directory '{os.path.dirname(full_path)}'. Please check folder permissions.")
        return None # Indicate failure
    except OSError as oe:
        # Catch other OS-level errors (e.g., invalid path, disk full)
        st.error(f"Failed to save file to {full_path} (or variant): OS Error - {oe}") # Mention potential variant name
        st.error(traceback.format_exc())
        return None
    except Exception as e:
        # Catch any other unexpected errors during file saving
        st.error(f"An unexpected error occurred while saving file to {full_path} (or variant): {e}") # Mention potential variant name
        st.error(traceback.format_exc())
        return None # Indicate failure


# --- Load Instructions ---
# Define directory where instruction files are located
INSTRUCTIONS_DIR = "instructions"
# Construct full paths to instruction files
global_context_instructions_path = os.path.join(INSTRUCTIONS_DIR, "global_context_instructions.txt")
standard_summary_instructions_path = os.path.join(INSTRUCTIONS_DIR, "standard_summary_instructions.txt")

# Load the content of the main instruction files
# Use updated enhanced instructions from previous turn if applicable
# Assuming enhanced instructions are now in the files:
global_context_instructions = load_instruction(global_context_instructions_path) # Assuming this file exists and is potentially enhanced
standard_instructions = load_instruction(standard_summary_instructions_path) # Assuming this file contains the enhanced version

# Define paths for role-specific outline/instruction files
role_files = {
    "Plaintiff": os.path.join(INSTRUCTIONS_DIR, "plaintiff_outline.txt"),
    "Defendant": os.path.join(INSTRUCTIONS_DIR, "defendant_outline.txt"),
    "Expert Witness": os.path.join(INSTRUCTIONS_DIR, "expert_witness_outline.txt"),
    "Fact Witness": os.path.join(INSTRUCTIONS_DIR, "fact_witness_outline.txt"),
}

# Check the status of instruction file loading/existence for sidebar display
instruction_load_status = {}
instruction_load_status["Global Context"] = bool(global_context_instructions) # True if loaded non-empty
instruction_load_status["Standard Summary"] = bool(standard_instructions) # True if loaded non-empty
# Check existence for role files (content is loaded later when needed)
for role, path in role_files.items():
    instruction_load_status[f"{role} Outline"] = os.path.exists(path)


# --- Sidebar Setup ---
st.sidebar.title("⚙️ Config & Status")
st.sidebar.markdown("---")

# Determine available providers based on successful client initialization
available_providers = [p for p, status in client_load_status.items() if "✅" in status]

# Handle case where no providers are configured/initialized
if not available_providers:
      st.sidebar.error("No LLM providers initialized successfully. Check API keys in the `.env` file.")
      # Set selections to None to prevent errors later in the UI logic
      selected_provider_context = None
      selected_model_context = None
      selected_provider_final = None
      selected_model_final = None
else:
    # --- LLM Selection for Stage 1: Context & Segments ---
    st.sidebar.subheader("1. Context & Segment LLM")
    st.sidebar.caption("Used for initial context generation and summarizing text chunks. Can often be a faster/cheaper model.")

    # Provider selection dropdown for Stage 1
    selected_provider_context = st.sidebar.selectbox(
        "Select Provider (Context/Segments):",
        options=available_providers,
        index=0, # Default to the first available provider
        key="provider_selector_context", # Unique key for this widget
        help="Choose the LLM service for context generation and segment summaries."
    )

    # Dynamically populate model options based on selected provider for Stage 1
    models_for_provider_context = []
    if selected_provider_context:
        if selected_provider_context == "OpenRouter":
            openrouter_key = api_keys.get("OpenRouter")
            # Fetch models (uses cache if available and key exists)
            models_for_provider_context = fetch_openrouter_models(openrouter_key) if openrouter_key else ["Error: OpenRouter Key Missing"]
        else:
            # Use predefined list for other providers
            models_for_provider_context = MODEL_MAPPING.get(selected_provider_context, ["Error: No models defined"])

        # Ensure the result is a list, handle potential errors during fetch/lookup
        if not isinstance(models_for_provider_context, list): models_for_provider_context = ["Error: Invalid model list format"]

        # Model selection dropdown for Stage 1
        if models_for_provider_context:
            default_index_context = 0 # Basic default
            # Attempt to set a more intelligent default
            # Set Google default to gemini-2.5-pro-exp-03-25
            if selected_provider_context == "Google" and "gemini-2.5-pro-exp-03-25" in models_for_provider_context:
                default_index_context = models_for_provider_context.index("gemini-2.5-pro-exp-03-25")
            # Other provider defaults (prioritize faster/cheaper)
            elif selected_provider_context == "OpenAI" and "o1-mini" in models_for_provider_context:
                default_index_context = models_for_provider_context.index("o1-mini")
            elif selected_provider_context == "OpenAI" and "o3-mini" in models_for_provider_context: # Fallback if o1 not there
                 default_index_context = models_for_provider_context.index("o3-mini")
            elif selected_provider_context == "Anthropic" and "claude-3-5-haiku-20241022" in models_for_provider_context:
                default_index_context = models_for_provider_context.index("claude-3-5-haiku-20241022")
            elif selected_provider_context == "OpenRouter" and not models_for_provider_context[0].startswith("Error"):
                cheap_or_models = ['google/gemini-flash-1.5', 'anthropic/claude-3.5-haiku', 'openai/gpt-3.5-turbo', 'mistralai/mistral-7b-instruct']
                for m in cheap_or_models:
                     if m in models_for_provider_context: default_index_context = models_for_provider_context.index(m); break

            selected_model_context = st.sidebar.selectbox(
                 "Select Model (Context/Segments):",
                 options=models_for_provider_context,
                 index=default_index_context,
                 # Dynamic key based on provider to prevent state issues if provider changes
                 key=f"model_selector_context_{selected_provider_context}",
                 help=f"Choose the specific model from {selected_provider_context} for context generation and segment summarization."
            )
        else:
            # Handle case where no models could be loaded for the selected provider
            st.sidebar.warning(f"No models available or loaded for {selected_provider_context}.")
            selected_model_context = None
    else:
        # Handle case where no provider is selected (shouldn't happen with defaults, but good practice)
        selected_model_context = None


    # --- LLM Selection for Stage 2: Final Synthesis ---
    st.sidebar.markdown("---") # Separator
    st.sidebar.subheader("2. Final Synthesis LLM")
    st.sidebar.caption("Used for combining segment summaries into the final structured output. Often benefits from a more capable model.")

    # Provider selection dropdown for Stage 2
    selected_provider_final = st.sidebar.selectbox(
        "Select Provider (Final Synthesis):",
        options=available_providers,
        index=0, # Default to first available
        key="provider_selector_final", # Unique key
        help="Choose the LLM service for the final summary synthesis step."
    )

    # Dynamically populate model options based on selected provider for Stage 2
    models_for_provider_final = []
    if selected_provider_final:
        if selected_provider_final == "OpenRouter":
            openrouter_key = api_keys.get("OpenRouter")
            # Fetch models (uses cache if available and key exists)
            models_for_provider_final = fetch_openrouter_models(openrouter_key) if openrouter_key else ["Error: OpenRouter Key Missing"]
        else:
            # Use predefined list for other providers
            models_for_provider_final = MODEL_MAPPING.get(selected_provider_final, ["Error: No models defined"])

        # Ensure result is a list
        if not isinstance(models_for_provider_final, list): models_for_provider_final = ["Error: Invalid model list format"]

        # Model selection dropdown for Stage 2
        if models_for_provider_final:
            default_index_final = 0 # Basic default
            # Attempt to set a more intelligent default
            # Set Google default to gemini-2.5-pro-exp-03-25
            if selected_provider_final == "Google" and "gemini-2.5-pro-exp-03-25" in models_for_provider_final:
                default_index_final = models_for_provider_final.index("gemini-2.5-pro-exp-03-25")
            # Other provider defaults (prioritize capable models)
            elif selected_provider_final == "OpenAI" and "gpt-4o" in models_for_provider_final:
                default_index_final = models_for_provider_final.index("gpt-4o")
            elif selected_provider_final == "Anthropic" and "claude-3-7-sonnet-20250219" in models_for_provider_final:
                default_index_final = models_for_provider_final.index("claude-3-7-sonnet-20250219")
            elif selected_provider_final == "OpenRouter" and not models_for_provider_final[0].startswith("Error"):
                capable_or_models = ['openai/gpt-4o', 'anthropic/claude-3.7-sonnet', 'google/gemini-pro-1.5', 'mistralai/mixtral-8x22b-instruct']
                for m in capable_or_models:
                     if m in models_for_provider_final: default_index_final = models_for_provider_final.index(m); break

            selected_model_final = st.sidebar.selectbox(
                 "Select Model (Final Synthesis):",
                 options=models_for_provider_final,
                 index=default_index_final,
                 # Dynamic key
                 key=f"model_selector_final_{selected_provider_final}",
                 help=f"Choose the specific model from {selected_provider_final} for the final synthesis step."
            )
        else:
            st.sidebar.warning(f"No models available or loaded for {selected_provider_final}.")
            selected_model_final = None
    else:
        selected_model_final = None

# --- REMOVED LLM Temperature Setting Section ---


# --- Display Selected Models & Statuses ---
st.sidebar.markdown("---")
st.sidebar.subheader("Selected LLMs (Using Default Temperature):") # Updated subheader
# Only display if providers were successfully loaded initially
if available_providers:
    # Display Context/Segment selection status
    if selected_provider_context and selected_model_context and not selected_model_context.startswith("Error:"):
        st.sidebar.write(f"Context/Segments: `{selected_provider_context} / {selected_model_context}`")
    elif selected_provider_context: # Provider selected, but model invalid
        st.sidebar.warning(f"Context/Segments: Invalid model selected for `{selected_provider_context}`.")
    else: # No provider selected (shouldn't happen with default)
         st.sidebar.warning("Context/Segment LLM not selected.")

    # Display Final Synthesis selection status
    if selected_provider_final and selected_model_final and not selected_model_final.startswith("Error:"):
        st.sidebar.write(f"Final Synthesis: `{selected_provider_final} / {selected_model_final}`")
    elif selected_provider_final: # Provider selected, but model invalid
        st.sidebar.warning(f"Final Synthesis: Invalid model selected for `{selected_provider_final}`.")
    else: # No provider selected
         st.sidebar.warning("Final Synthesis LLM not selected.")

# Display API Key & Client Initialization Status
st.sidebar.markdown("---")
st.sidebar.subheader("API Key & Client Status:")
for provider, status in client_load_status.items():
    if "✅" in status: st.sidebar.success(f"{provider}: {status}")
    elif "⚠️" in status: st.sidebar.warning(f"{provider}: {status}")
    else: st.sidebar.error(f"{provider}: {status}") # Assumes errors contain ❌

# Display Instruction File Status
st.sidebar.markdown("---")
st.sidebar.subheader("Instruction Files Status:")
# Check Global Context instructions
if instruction_load_status["Global Context"]: st.sidebar.success("✅ Global context instructions loaded.")
else: st.sidebar.warning("⚠️ Global context instructions missing or empty.")
# Check Standard Summary instructions
if instruction_load_status["Standard Summary"]: st.sidebar.success("✅ Standard summary instructions loaded.")
else: st.sidebar.warning("⚠️ Standard summary instructions missing or empty.")
# Check Role Outline file existence
all_roles_loaded = True
for role, path in role_files.items():
    role_key = f"{role} Outline"
    exists = instruction_load_status.get(role_key, False) # Check status dictionary
    if exists:
        st.sidebar.success(f"✅ {role} outline file found.")
    else:
        st.sidebar.error(f"❌ {role} outline file MISSING ({os.path.basename(path)})")
        all_roles_loaded = False
# Overall message if any role files are missing
if not all_roles_loaded:
    st.sidebar.error("One or more required role outline files are missing from the 'instructions' folder.")


# --- Streamlit UI (Main Area) ---
st.title("📄 Deposition Transcript Summarizer")
st.markdown(f"""
Upload a transcript (.txt, .pdf, .docx), select the deponent's role, configure **two LLMs** in the sidebar (one LLM for context/segments, one for final synthesis), add optional instructions (especially correct name spellings), and click generate. Models will use their default temperature settings.

**Output Location:** `{OUTPUT_DIRECTORY}` (Ensure this directory exists and is writable). Files will not be overwritten; a number like `_(1)` will be appended if a file with the same name exists.
""") # Updated description about file saving

# --- Main Area Inputs (File Upload, Role, Custom Instructions) ---
col1, col2 = st.columns([2, 1]) # Create two columns for layout
with col1:
    # File uploader widget
    uploaded_file = st.file_uploader(
        "1. Upload Transcript",
        type=["txt", "pdf", "docx"], # Allowed file types
        key="file_uploader" # Unique key for the widget
    )
    # Deponent role selection dropdown
    deponent_role = st.selectbox(
        "2. Select Deponent Role:",
        options=list(role_files.keys()), # Get roles from the defined dictionary
        key="role_selector" # Unique key
    )
with col2:
    # <<< CHANGE START >>>
    # Text area for optional custom instructions - updated placeholder and help text
    custom_instructions = st.text_area(
        "3. Optional: Add Custom Instructions (Focus on Names)",
        height=180, # Set height of the text area
        placeholder="Enter correctly spelled names (people, places, companies). The AI *must* use these exact spellings in the output.\n\nExample:\nJohnathan P. Smyth (not Jon Smith)\nACME Corporation Ltd.\n123 Oak Street, Anytown",
        key="custom_instructions", # Unique key
        help="Provide a list of key names with their correct spellings. The LLMs will be explicitly instructed to use these exact spellings for any corresponding names found in the transcript. This helps ensure consistency and accuracy in the summaries. You can also add other general instructions here."
    )
    # <<< CHANGE END >>>


# --- Processing Trigger Button ---
st.markdown("---") # Visual separator

# Determine if the Generate button should be disabled
process_button_disabled = (
    uploaded_file is None or # No file uploaded
    not available_providers or # No LLM providers configured
    # Context/Segment LLM selection is invalid
    selected_provider_context is None or selected_model_context is None or
    selected_model_context.startswith("Error:") or selected_model_context == "Fetching..." or
    # Final Synthesis LLM selection is invalid
    selected_provider_final is None or selected_model_final is None or
    selected_model_final.startswith("Error:") or selected_model_final == "Fetching..."
)
# Tooltip explaining why the button might be disabled
process_button_tooltip = "Please upload a transcript file and select valid LLM providers/models for both stages in the sidebar to enable generation." if process_button_disabled else "Start processing the transcript and generate summaries."

# The main button to start the summarization process
process_button = st.button(
    "🚀 Generate & Save Summaries",
    type="primary", # Makes the button visually prominent
    disabled=process_button_disabled, # Enable/disable based on checks
    key="generate_button", # Unique key
    help=process_button_tooltip # Show tooltip on hover
)

# --- Main Processing Logic ---
# This block executes only when the 'Generate' button is clicked and not disabled
if process_button:

    # --- Final Pre-checks before starting long process ---
    st.info(f"Starting Process...") # Initial feedback
    # Re-verify provider/model selections (belt-and-suspenders check)
    if not available_providers: st.error("Critical Error: No LLM providers configured."); st.stop()
    if not selected_provider_context or not selected_model_context or selected_model_context.startswith("Error:") or selected_model_context == "Fetching...": st.error("Pre-check Failed: Invalid Context/Segment LLM selection."); st.stop()
    if not selected_provider_final or not selected_model_final or selected_model_final.startswith("Error:") or selected_model_final == "Fetching...": st.error("Pre-check Failed: Invalid Final Synthesis LLM selection."); st.stop()
    # Check client initialization status for *selected* providers
    if selected_provider_context not in clients or not client_load_status.get(selected_provider_context, "").startswith("✅"): st.error(f"Client initialization error for Context/Segments provider '{selected_provider_context}'. Check API key/status."); st.stop()
    if selected_provider_final not in clients or not client_load_status.get(selected_provider_final, "").startswith("✅"): st.error(f"Client initialization error for Final Synthesis provider '{selected_provider_final}'. Check API key/status."); st.stop()
    # Check if essential instruction files were loaded successfully
    if not standard_instructions: st.error("Pre-check Failed: Standard summary instructions missing or empty."); st.stop()
    if not global_context_instructions: st.error("Pre-check Failed: Global context instructions missing or empty."); st.stop()
    # Check existence and load content of the selected role's outline file
    selected_role_file = role_files.get(deponent_role)
    if not selected_role_file or not os.path.exists(selected_role_file): st.error(f"Pre-check Failed: Outline file for role '{deponent_role}' is missing."); st.stop()
    role_outline_instructions = load_instruction(selected_role_file)
    if not role_outline_instructions: st.error(f"Pre-check Failed: Outline file for '{deponent_role}' could not be loaded or is empty."); st.stop()
    # Verify output directory exists and is writable
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        # Attempt to create and delete a temporary file to test write permissions
        test_file_path = os.path.join(OUTPUT_DIRECTORY, ".streamlit_permission_test")
        with open(test_file_path, "w") as f_test: f_test.write("test")
        os.remove(test_file_path)
        st.info(f"Output directory checked and seems writable: {OUTPUT_DIRECTORY}")
    except PermissionError:
         st.error(f"Pre-check Failed: Permission denied writing to the output directory: {OUTPUT_DIRECTORY}. Please check folder permissions.")
         st.stop()
    except Exception as e:
        st.error(f"Pre-check Failed: Could not create, access, or write to output directory: {OUTPUT_DIRECTORY}. Error: {e}")
        st.stop()

    # --- Setup Progress Reporting ---
    st.markdown("---") # Separator
    st.subheader("📊 Processing Status:")
    status_placeholder = st.empty() # Placeholder to update status messages
    progress_bar = st.progress(0.0) # Initialize progress bar (uses 0.0 to 1.0 scale)
    saved_path1 = None # Initialize variable to store actual saved path for output 1
    saved_path2 = None # Initialize variable to store actual saved path for output 2
    save_success1_flag = False # Initialize save success flag for output 1
    save_success2_flag = False # Initialize save success flag for output 2


    # Helper function to update progress bar and status message
    def update_status(percentage, text):
        # Clamp percentage between 0 and 100, then scale to 0.0-1.0
        clamped_percentage_float = max(0.0, min(1.0, percentage / 100.0))
        # Ensure text is a string
        text_str = str(text) if text is not None else ""
        # Update progress bar
        progress_bar.progress(clamped_percentage_float, text=text_str)
        # Update status message with appropriate styling based on content
        text_lower = text_str.lower()
        if "error" in text_lower or "failed" in text_lower or "❌" in text_str: status_placeholder.error(text_str)
        elif "warning" in text_lower or "issue" in text_lower or "⚠️" in text_str: status_placeholder.warning(text_str)
        elif "✔️" in text_str: status_placeholder.success(text_str)
        else: status_placeholder.info(text_str) # Default to info

    # --- Main Processing Steps ---
    try:
        # Update initial status message - removed temperature
        update_status(0, f"Starting process... Ctx/Seg Model: {selected_model_context}, Final Model: {selected_model_final}")

        # --- Step 1: Parse Uploaded File ---
        update_status(5, f"Step 1/7: Parsing '{uploaded_file.name}'...")
        transcript_text, error = parse_uploaded_file(uploaded_file)
        # Handle parsing errors
        if error:
            update_status(5, f"❌ Step 1 Failed: File Parsing Error - {error}")
            st.stop() # Stop execution if parsing fails
        # Handle case where parsing succeeds but extracts no text
        if transcript_text is None or not transcript_text.strip():
             update_status(5, f"❌ Step 1 Failed: Parsing '{uploaded_file.name}' resulted in empty text content. Cannot proceed.")
             st.stop() # Stop execution if no text
        update_status(10, "✔️ Step 1: File parsed successfully.")
        # Optionally show a snippet for verification
        # st.text_area("Parsed Text Snippet:", transcript_text[:500]+"...", height=100, disabled=True)

        # --- Step 2: Generate Global Context ---
        # Use the model selected for Context/Segments
        update_status(15, f"Step 2/7: Generating global context ({selected_model_context})...")
        # <<< CHANGE START >>>
        # Construct the prompt using loaded instructions, CUSTOM instructions, and transcript text
        context_prompt = f"""BEGIN GLOBAL CONTEXT INSTRUCTIONS:
---
{global_context_instructions}
---
END GLOBAL CONTEXT INSTRUCTIONS.

BEGIN USER'S CUSTOM INSTRUCTIONS (Pay close attention to required name spellings):
---
{custom_instructions if custom_instructions else 'None provided.'}
---
END USER'S CUSTOM INSTRUCTIONS.

BEGIN TRANSCRIPT TEXT:
---
{transcript_text}
---
END TRANSCRIPT TEXT

TASK: Generate the global context summary according to the 'GLOBAL CONTEXT INSTRUCTIONS'. Critically, if specific name spellings (for people, companies, places, etc.) are provided in the 'USER'S CUSTOM INSTRUCTIONS', you MUST use those exact spellings whenever referring to those entities in your generated context. If no custom instructions are provided or they don't mention specific names, proceed as normal based on the transcript."""
        # <<< CHANGE END >>>

        # Call the LLM - removed temperature argument
        global_context = call_llm(context_prompt, selected_provider_context, selected_model_context)
        # Handle LLM call errors
        if not global_context or global_context.startswith("Error:"):
            update_status(15, f"❌ Step 2 Failed: Global context generation error - {global_context}")
            st.stop() # Stop execution if context generation fails
        update_status(25, "✔️ Step 2: Global context generated.")
        # Display the generated context in an expander
        with st.expander("View Generated Global Context"):
            st.write(global_context)

        # --- Step 3: Chunk Transcript Text ---
        update_status(30, "Step 3/7: Segmenting transcript text...")
        # Define chunk size and overlap (could be made configurable later)
        # Larger chunks are generally better for models with large context windows
        # Smaller chunks might be needed for models with limited context
        chunk_s = 8000 # Target chunk size in characters
        chunk_o = 400  # Overlap between chunks in characters
        text_chunks = chunk_text(transcript_text, chunk_size=chunk_s, chunk_overlap=chunk_o)
        # Handle chunking errors
        if not text_chunks:
            update_status(30, "❌ Step 3 Failed: Failed to segment transcript text after parsing.")
            st.stop() # Stop execution if chunking fails
        update_status(35, f"✔️ Step 3: Transcript divided into {len(text_chunks)} segments.")

        # --- Step 4: Summarize Segments ---
        # Use the model selected for Context/Segments
        update_status(40, f"Step 4/7: Summarizing {len(text_chunks)} segments ({selected_model_context})...")
        segment_summaries = [] # List to store summaries of each segment
        total_chunks = len(text_chunks)
        segment_errors = 0 # Counter for failed segments

        # Loop through each text chunk
        for i, chunk in enumerate(text_chunks):
            # Calculate progress percentage for this step (spread across 40% to 90%)
            progress_percentage = 40 + int(50 * ((i + 1) / total_chunks))
            update_status(progress_percentage, f"Step 4/7: Summarizing segment {i+1}/{total_chunks}...")

            # Construct the prompt for summarizing the current segment
            # Assumes enhanced standard_instructions are loaded
            segment_prompt = f"""BEGIN STANDARD SUMMARY INSTRUCTIONS:
---
{standard_instructions}
---
END STANDARD SUMMARY INSTRUCTIONS.

BEGIN CONTEXT AND DATA FOR SUMMARY:

Overall Case Context:
{global_context}

User's Custom Instructions (Pay close attention to required name spellings):
---
{custom_instructions if custom_instructions else 'None provided.'}
---
END USER'S CUSTOM INSTRUCTIONS.

Transcript Segment To Summarize ({i+1}/{total_chunks}):
--- START SEGMENT ---
{chunk}
--- END SEGMENT ---

END CONTEXT AND DATA FOR SUMMARY.

TASK: Generate a detailed, objective summary for ONLY the 'Transcript Segment To Summarize' provided above. Adhere strictly to the 'STANDARD SUMMARY INSTRUCTIONS'. Use the 'Overall Case Context' and 'User's Custom Instructions' for background and focus, but DO NOT summarize them. Critically, if specific name spellings (for people, companies, places, etc.) are provided in the 'USER'S CUSTOM INSTRUCTIONS', you MUST use those exact spellings whenever referring to those entities in your segment summary. Base your summary *only* on the information present within the 'START SEGMENT' and 'END SEGMENT' markers. Output ONLY the summary text for this segment."""

            # Call the LLM for the current segment - removed temperature argument
            segment_summary = call_llm(segment_prompt, selected_provider_context, selected_model_context) # Use context model

            # Handle LLM call result for the segment
            if segment_summary and not segment_summary.startswith("Error:"):
                segment_summaries.append(segment_summary) # Add successful summary
            else:
                # Log warning and add placeholder if segment summarization fails
                st.warning(f"Failed to summarize segment {i+1}. Using error placeholder. LLM Response: {segment_summary}")
                segment_summaries.append(f"*** Error summarizing segment {i+1}. See warnings above. ***")
                segment_errors += 1

        # Report outcome of segment summarization step
        if segment_errors > 0:
            update_status(90, f"⚠️ Step 4: Completed segment summarization with {segment_errors} error(s).")
        else:
            update_status(90, f"✔️ Step 4: All {len(segment_summaries)} segments processed successfully.")

        # --- Output Generation and Saving ---
        st.markdown("---"); st.subheader("💾 Saving Outputs:")

        # Generate a safe base filename from the uploaded file name
        base_name = os.path.splitext(uploaded_file.name)[0]
        # Remove characters invalid in filenames, replace spaces with underscores
        safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '.', '_', '-')).strip().replace(' ', '_')
        # Provide a fallback name if cleaning results in empty string
        safe_base_name = safe_base_name or "transcript_summary"
        # Sanitize role name for filename
        role_filename_part = deponent_role.lower().replace(' ', '_').replace('/', '_')

        # Create concise representations of the models used for filenames
        # Example: openai_gpt-4o -> ope-gpt / anthropic_claude-3-7... -> ant-claude
        model1_provider_short = selected_provider_context[:3].lower() if selected_provider_context else "unk"
        model1_name_short = selected_model_context.split('/')[-1].split('-')[2].lower() if selected_model_context and '-' in selected_model_context and len(selected_model_context.split('-')) > 2 else selected_model_context.split('/')[-1][:3].lower() # Trying to get 'gpt', 'claude', 'gemini' etc., fallback to first 3 chars
        model1_info = f"{model1_provider_short}-{model1_name_short}"

        model2_provider_short = selected_provider_final[:3].lower() if selected_provider_final else "unk"
        model2_name_short = selected_model_final.split('/')[-1].split('-')[2].lower() if selected_model_final and '-' in selected_model_final and len(selected_model_final.split('-')) > 2 else selected_model_final.split('/')[-1][:3].lower()
        model2_info = f"{model2_provider_short}-{model2_name_short}"


        # --- Step 5: Generate & Save Output 1 (Context + Segment Summaries) ---
        update_status(92, "Step 5/7: Generating segment summaries document...")
        # Prepare content dictionary for DOCX generation
        output1_content = {"Global Context Summary": global_context}
        valid_segment_summary_count = 0
        # Add each segment summary (or error placeholder) to the dictionary
        for i, summary in enumerate(segment_summaries):
            output1_content[f"Segment {i+1} Summary"] = summary
            # Count successfully generated summaries
            if "*** Error summarizing segment" not in summary:
                 valid_segment_summary_count += 1

        # Generate the DOCX file content in memory
        docx_output1_bytes = generate_docx_bytes(output1_content)

        # Construct filename including model info for segments
        output1_filename = f"{safe_base_name}_Segments_{model1_info}.docx"
        output1_fullpath = os.path.join(OUTPUT_DIRECTORY, output1_filename)

        # Save the generated DOCX bytes to file - GET THE FINAL PATH
        saved_path1 = save_bytes_to_file(docx_output1_bytes, output1_fullpath)

        # Update status based on save outcome (check if saved_path1 is None)
        if saved_path1 is None: # Check if saving failed
            update_status(92, f"❌ Step 5 Failed: Could not save the Segment Summaries DOCX file (Initial target: {output1_filename}).")
            save_success1_flag = False # Flag for final message
            # Note: Continue processing to attempt final summary even if this fails
        else:
            # Success message is handled inside save_bytes_to_file now
            # Just update the overall status and flag
            update_status(94, f"✔️ Step 5: Segment summaries document generated (saved as {os.path.basename(saved_path1)}).")
            save_success1_flag = True # Flag for final message


        # --- Step 6: Combine & Refine (Final Synthesis) ---
        # Check if there are any valid summaries to synthesize
        if valid_segment_summary_count == 0:
            update_status(95, "❌ Step 6 Skipped: No valid segment summaries were generated to synthesize.")
            # Report appropriate final status based on whether segment file was saved
            if not save_success1_flag:
                st.error("Processing stopped: No valid segments generated and segment file also failed to save.")
            else:
                st.warning(f"Processing stopped: No valid segments generated, but the segment file (Output 1) was saved as {os.path.basename(saved_path1)}.") # Use actual saved path
            st.stop() # Stop execution

        # Proceed with final synthesis using the selected Final Synthesis LLM
        update_status(95, f"Step 6/7: Generating final synthesis ({selected_model_final})...")

        # Combine only the successfully generated segment summaries into a single text block
        combined_summaries_text = "\n\n--- End of Segment / Start of Next ---\n\n".join(
            [f"Segment {i+1} Summary:\n{s}" for i, s in enumerate(segment_summaries) if "*** Error summarizing segment" not in s]
        )

        # Construct the prompt for the final synthesis step
        # Assumes enhanced role_outline_instructions are loaded
        final_prompt = f"""BEGIN ROLE-SPECIFIC SYNTHESIS INSTRUCTIONS & OUTLINE ({deponent_role}):
---
{role_outline_instructions}
---
END ROLE-SPECIFIC SYNTHESIS INSTRUCTIONS & OUTLINE.

BEGIN DATA TO SYNTHESIZE:

User's Custom Instructions (Pay close attention to required name spellings):
---
{custom_instructions if custom_instructions else 'None provided.'}
---
END USER'S CUSTOM INSTRUCTIONS.

Collection of Segment Summaries:
--- START SUMMARIES ---
{combined_summaries_text}
--- END SUMMARIES ---

END DATA TO SYNTHESIZE.

TASK: You are an experienced civil litigation attorney. Your task is to synthesize the provided 'Collection of Segment Summaries' into a single, cohesive, objective narrative deposition summary. Strictly follow the structure, headings, and requirements outlined in the 'ROLE-SPECIFIC SYNTHESIS INSTRUCTIONS & OUTLINE'. Combine related information from different segments, deduplicate redundant points, and organize the content logically under the specified headings. Use the 'User's Custom Instructions' for focus but ensure the final summary remains objective and based *only* on the provided segment summaries. Critically, if specific name spellings (for people, companies, places, etc.) are provided in the 'USER'S CUSTOM INSTRUCTIONS', you MUST use those exact spellings whenever referring to those entities in your final synthesized summary. Do not invent information. If a required section in the outline cannot be filled from the provided summaries, state that explicitly (e.g., "No testimony regarding X was found in the provided summaries."). Begin your output *directly* with the first heading specified in the outline. Do not include introductory phrases like "Here is the final summary:".
"""
        # Call the LLM using the model selected for Final Synthesis - removed temperature argument
        final_summary = call_llm(final_prompt, selected_provider_final, selected_model_final)

        # Handle errors during final synthesis
        if not final_summary or final_summary.startswith("Error:"):
             update_status(95, f"❌ Step 6 Failed: Final summary generation error - {final_summary}")
             # Report partial success if segments file was saved
             if save_success1_flag:
                 st.warning(f"Final summary generation failed, but the segment summaries document (Output 1) was saved successfully as {os.path.basename(saved_path1)}.") # Use actual path
             else:
                 st.error("Both segment summary saving and final summary generation failed.")
             st.stop() # Stop execution

        update_status(97, "✔️ Step 6: Final cohesive summary generated successfully.")


        # --- Step 7: Generate & Save Output 2 (Final Summary) ---
        update_status(98, "Step 7/7: Generating final summary document...")

        # Prepare content dictionary for the final DOCX
        # Include original filename context in the document title
        doc_title = f"Final Deposition Summary ({deponent_role}) - {safe_base_name}"
        output2_content = {doc_title: final_summary}

        # Generate the DOCX file content in memory
        docx_output2_bytes = generate_docx_bytes(output2_content)

        # Construct filename including role and model info for final summary
        output2_filename = f"{safe_base_name}_FinalSummary_{role_filename_part}_{model2_info}.docx"
        output2_fullpath = os.path.join(OUTPUT_DIRECTORY, output2_filename)

        # Save the generated DOCX bytes to file - GET THE FINAL PATH
        saved_path2 = save_bytes_to_file(docx_output2_bytes, output2_fullpath)

        # Update status based on save outcome (check if saved_path2 is None)
        if saved_path2 is None: # Check if saving failed
            update_status(98, f"❌ Step 7 Failed: Could not save the Final Summary DOCX file (Initial target: {output2_filename}).")
            save_success2_flag = False # Flag for final message
        else:
            # Success message is handled inside save_bytes_to_file now
            # Just update the overall status and flag
            update_status(99, f"✔️ Step 7: Final summary document generated (saved as {os.path.basename(saved_path2)}).")
            save_success2_flag = True # Flag for final message


        # --- Final Completion ---
        progress_bar.progress(1.0) # Ensure progress bar reaches 100%
        # Provide final status message based on which files were saved successfully
        final_message = ""
        if save_success1_flag and save_success2_flag:
            final_message = f"🎉 Processing complete! Summary files saved in '{OUTPUT_DIRECTORY}' (as '{os.path.basename(saved_path1)}' and '{os.path.basename(saved_path2)}')."
            status_placeholder.success(final_message)
            st.balloons() # Fun success indicator
        elif save_success1_flag: # Only segments saved
            final_message = f"⚠️ Processing finished, but the Final Summary file (Output 2) failed to save. Segment summaries (Output 1) were saved (as {os.path.basename(saved_path1)}) to '{OUTPUT_DIRECTORY}'. Check logs/permissions."
            status_placeholder.warning(final_message)
        elif save_success2_flag: # Only final saved
             final_message = f"⚠️ Processing finished, but the Segment Summaries file (Output 1) failed to save. Final summary (Output 2) was saved (as {os.path.basename(saved_path2)}) to '{OUTPUT_DIRECTORY}'. Check logs/permissions."
             status_placeholder.warning(final_message)
        else: # Neither file saved
            final_message = f"❌ Processing finished, but BOTH summary files failed to save. Check logs and permissions for '{OUTPUT_DIRECTORY}'."
            status_placeholder.error(final_message)

    # --- Catch Unexpected Errors during Main Processing ---
    except Exception as e:
        progress_bar.progress(1.0) # Ensure progress bar completes on error
        detailed_error = traceback.format_exc() # Get full traceback
        # Display comprehensive error information in the UI
        st.error(f"An unexpected error terminated the process: {e}")
        st.error("Traceback:")
        st.code(detailed_error, language='text') # Display traceback in a code block
        # Update status placeholder with final error message
        try: update_status(100, f"❌ Processing stopped due to unexpected error: {e}")
        except Exception: print(f"Failed to update final error status: {e}") # Log if status update itself fails


# --- Footer ---
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** AI-generated summaries require careful review by a qualified legal professional. Verify all facts, interpretations, and omissions against the original transcript.")