import os
from tempfile import NamedTemporaryFile
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from typing import List

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extracts text from various file formats (PDF, DOCX, PPTX, Images) using the unstructured library.
    Because unstructured often needs a file path to inspect the magic bytes or extension,
    we write the incoming bytes to a temporary file before partitioning.
    """
    # Using original suffix to help unstructured identify the file type
    ext = os.path.splitext(filename)[1]
    
    with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        # Extract elements from the document
        elements = partition(filename=temp_path)
        
        # Optional: Group by title or keep as plain text.
        # Here, we join everything as semantic chunking will happen later.
        text = "\n\n".join([str(el) for el in elements])
        print(f"DEBUG: Extracted text length: {len(text)}")
        return text
    
    except Exception as e:
        print(f"Error extracting text from file: {e}")
        return ""
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
