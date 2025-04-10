# helpers/text_ext_helper.py

import io
import os
import re
import json
import logging
import pytesseract
from PIL import Image
from psycopg2 import sql
from openai import OpenAI
from urllib.parse import urlparse
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from helpers.llm_helpers import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from typing import Union
from io import BytesIO

# ====================================================
# Section: CONFIGURATION
# ====================================================
# Description: Setup Logger
# ====================================================
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================
# Section: FULL PROCESS TRIGGER
# ====================================================
# Description: Invokes the processing of the file
# ====================================================
def read_and_extract_document(user_id, file_input: Union[str, BytesIO], file_type: str) -> str:
    """
    Main function to process the document and return extracted information.
    Supports both file paths and in-memory BytesIO objects.
    """
    # Validate and read the file input
    if isinstance(file_input, (str, os.PathLike)):
        print("File Input is a string")
        # Validate file existence if input is a file path
        if not os.path.exists(file_input):
            logger.error(f"The file at path {file_input} does not exist.")
            raise FileNotFoundError(f"File not found: {file_input}")

        # Read the file as bytes
        try:
            with open(file_input, 'rb') as file:
                print("Reading file as bytes")
                file_bytes = file.read()
        except IOError as io_err:
            logger.error(f"Failed to read file at {file_input}: {io_err}")
            raise io_err

    elif isinstance(file_input, BytesIO):
        # If input is an in-memory file, read its contents directly
        print("File Input is a BytesIO object")
        file_bytes = file_input.getvalue()
    else:
        print("Unsupported input type")
        logger.error("file_input must be a file path or a BytesIO object.")
        raise TypeError("Unsupported input type: file_input must be str or BytesIO.")

    # Process the document to extract text
    try:
        print("Processing document for text extraction")
        extracted_text = process_document(file_bytes, file_type)
        logger.info("Document processed and text extracted.")
        print("Document processed and text extracted.")
    except Exception as e:
        logger.error(f"Error processing document for text extraction: {e}")
        raise RuntimeError(f"Error during document processing: {e}")

    # Process all pages at once
    try:
        print("Processing pages")
        document_outputs = process_pages(user_id, extracted_text)
    except Exception as e:
        logger.error(f"Failed to process pages: {e}")
        raise e

    # Optionally, sort the results by page number
    document_outputs.sort(key=lambda x: x['page'])

    return json.dumps(document_outputs, indent=4)

def process_pages(user_id, page_contents: List[str]) -> List[Dict]:
    """
    Process multiple pages concurrently and extract information.

    Args:
        page_contents (List[str]): A list of page contents.

    Returns:
        List[Dict]: A list of dictionaries containing page number, category, and details.
    """
    try:
        # ====================================================
        # Section: Get Document Types
        # ====================================================
        document_type_infos = detect_document_types(user_id, page_contents)
        logger.info(f"Document types extracted for {len(page_contents)} pages")
        print(f"Document types extracted for {len(page_contents)} pages")

        # Create a mapping from page number to (content, classification)
        page_info = {
            classification.page_number: (page_contents[classification.page_number - 1], classification)
            for classification in document_type_infos.pages
        }

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    process_single_page, user_id, page_num, content, classification
                ): page_num
                for page_num, (content, classification) in page_info.items()
            }

            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unhandled exception for page {page_num}: {e}")
                    results.append({
                        'page': page_num,
                        'category': None,
                        'details': None,
                        'error': str(e)
                    })

        # Sort results by page number to maintain order
        results.sort(key=lambda x: x['page'])
        return results

    except Exception as e:
        logger.error(f"Error processing pages: {e}")
        raise RuntimeError(f"Error during information extraction: {e}") from e

def process_single_page(user_id, page_num: int, page_content: str, classification: PageClassification) -> Dict:
    """
    Process a single page and extract information.

    Args:
        page_num (int): Page number.
        page_content (str): Content of the page.
        classification (PageClassification): Classification info for the page.

    Returns:
        Dict: A dictionary containing page number, category, and details.
    """
    try:
        page_document_type = classification.category
        logger.info(f"Processing page {page_num} with document type {page_document_type}")
        print(f"Processing page {page_num} with document type {page_document_type}")

        # Extract structured information based on document type
        structured_info = process_document_based_on_type(user_id, page_content, page_document_type)
        logger.info(f"Information extracted from page {page_num}")
        print(f"Information extracted from page {page_num}")

        return {
            'page': page_num,
            'category': page_document_type.value,
            'details': structured_info.dict() if structured_info else None
        }
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")
        return {
            'page': page_num,
            'category': page_document_type.value if 'page_document_type' in locals() else None,
            'details': None,
            'error': str(e)
        }

def process_page(user_id, pagenum: int, page_content: str) -> Dict:
    """
    Process a single page and extract information.
    """
    try:
        # ====================================================
        # Section: Get Document Type
        # ====================================================
        document_type_info = detect_document_type(user_id, page_content)
        page_document_type = document_type_info.category
        logger.info(f"Document type extracted from page {pagenum + 1}")
        print(f"Document type extracted from page {pagenum + 1}")

        # ====================================================
        # Section: Extract Structure Information
        # ====================================================
        structured_info = process_document_based_on_type(user_id, page_content, page_document_type)
        logger.info(f"Information extracted from page {pagenum + 1}")
        print(f"Information extracted from page {pagenum + 1}")

        result = {
            'page': pagenum + 1,
            'category': page_document_type.value,
            'details': structured_info.dict() if structured_info else None
        }
        return result

    except KeyError as key_err:
        logger.error(f"Invalid document type on page {pagenum + 1}: {key_err}")
        raise ValueError(f"Unsupported document type on page {pagenum + 1}: {page_document_type}") from key_err
    except Exception as e:
        logger.error(f"Error processing page {pagenum + 1}: {e}")
        raise RuntimeError(f"Error during information extraction on page {pagenum + 1}: {e}") from e

# Non AI Functions
def validate_dd214(file_path):
    pages = convert_from_bytes(open(file_path, 'rb').read())
    texts = [pytesseract.image_to_string(page) for page in pages]
    classifications = detect_document_types(texts)

    if len(classifications.pages) != 1:
        return {"error": "DD214 must contain exactly one page."}
    
    page = classifications.pages[0]
    if page.category != DocumentType.DD214 or page.confidence < 0.9:
        return {"error": "The document is not recognized as a valid DD214."}
    
    return {"message": "Document validated as DD214.", "classification": page}

def preprocess_text(text):
    """
    Preprocesses the input text by cleaning and normalizing.
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: Further normalization or entity extraction can be added here
    
    return text  

def ocr_image(page_num_image_tuple):
    """
    Worker function to perform OCR on a single image.
    
    Args:
        page_num_image_tuple (tuple): A tuple containing the page number and the image object.
    
    Returns:
        str: Extracted text from the image.
    """
    page_num, image = page_num_image_tuple
    try:
        logger.debug(f"Starting OCR for page {page_num}")
        print(f"Starting OCR for page {page_num}")
        text = pytesseract.image_to_string(image)
        logger.debug(f"Completed OCR for page {page_num}")
        print(f"Completed OCR for page {page_num}")
        return f"Page {page_num}:\n{text}"
    except Exception as e:
        logger.error(f"Error during OCR on page {page_num}: {e}")
        raise RuntimeError(f"OCR failed on page {page_num}") from e

def extract_page_num(text: str) -> int:
    """
    Extracts the page number from the given text.

    Args:
        text (str): The text containing the page number.

    Returns:
        int: The extracted page number.

    Raises:
        ValueError: If the page number cannot be found or converted.
    """
    match = re.match(r"Page\s+(\d+):", text)
    if match:
        return int(match.group(1))
    else:
        logger.warning(f"Unable to extract page number from text: {text[:30]}...")
        print(f"Unable to extract page number from text: {text[:30]}...")
        return 0  # Assign a default or handle as needed

def process_image_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes using OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def process_pdf_bytes(pdf_bytes: bytes, batch_size: int = 3) -> List[str]:
    """
    Convert PDF bytes to text by performing OCR on each page in batches.
    
    Args:
        pdf_bytes (bytes): The PDF file content in bytes.
        batch_size (int): Number of pages to process at a time.
    
    Returns:
        List[str]: A list containing the extracted text for each page.
    """
    try:
        logger.info("Obtaining PDF info to determine total pages.")
        info = pdfinfo_from_bytes(pdf_bytes)
        total_pages = info.get("Pages", 0)
        logger.info(f"Total pages in PDF: {total_pages}")
        print(f"Total pages in PDF: {total_pages}")
    except Exception as e:
        logger.error(f"Error obtaining PDF info: {e}")
        raise RuntimeError("Failed to obtain PDF info.") from e

    all_text_content = []

    # Process the PDF in batches of 'batch_size' pages. forcing the batch size to be 3
    for start_page in range(1, total_pages + 1, batch_size):
        end_page = min(start_page + batch_size - 1, total_pages)
        try:
            logger.info(f"Converting pages {start_page} to {end_page} to images.")
            print(f"Converting pages {start_page} to {end_page} to images.")
            batch_images = convert_from_bytes(
                pdf_bytes,
                first_page=start_page,
                last_page=end_page
            )
        except Exception as e:
            logger.error(f"Error converting pages {start_page}-{end_page}: {e}")
            raise RuntimeError(f"Failed to convert pages {start_page}-{end_page} to images.") from e

        batch_text_content = []
        # Create tuples of (page_number, image) for the current batch.
        page_num_image_tuples = list(enumerate(batch_images, start=start_page))
        max_workers = min(len(batch_images), 3)  # up to 10 threads for this batch
        logger.info(f"Starting OCR for pages {start_page} to {end_page} using {max_workers} threads.")
        print(f"Starting OCR for pages {start_page} to {end_page} using {max_workers} threads.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(ocr_image, pair): pair[0] for pair in page_num_image_tuples}
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_text = future.result()
                    batch_text_content.append(page_text)
                    logger.info(f"OCR completed for page {page_num}.")
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")
                    batch_text_content.append(f"\n\nPage {page_num}:\n[Error processing page]")
        # Add the batch results to the overall results.
        all_text_content.extend(batch_text_content)

    # Optionally, sort the results by page number (if order is important).
    text_content_sorted = sorted(all_text_content, key=extract_page_num)
    logger.info("Completed OCR for all pages.")
    print("Completed OCR for all pages.")
    return text_content_sorted

def process_document(file_content: bytes, file_type: str) -> str:
    """Process the document and extract text content."""
    if file_type.lower() == 'pdf':
        return process_pdf_bytes(file_content)
    elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff']:
        return process_image_bytes(file_content)
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or image file.")
