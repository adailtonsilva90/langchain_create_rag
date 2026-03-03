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
    # Usando o sufixo original para ajudar o unstructured na identificação do tipo
    ext = os.path.splitext(filename)[1]
    
    with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        # Extrai os elementos do documento
        elements = partition(filename=temp_path)
        
        # Opcional: Agrupar por título ou manter texto puro. 
        # Aqui, juntamos tudo pois o chunking semântico ocorrerá depois.
        text = "\n\n".join([str(el) for el in elements])
        return text
    
    except Exception as e:
        print(f"Erro ao extrair texto do arquivo: {e}")
        return ""
    finally:
        # Limpar o arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
