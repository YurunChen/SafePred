"""
Document Parser Module for Risk Rule Extraction.

Supports parsing various document formats (PDF, TXT, DOC, DOCX, etc.)
and extracting text content for LLM-based rule extraction.
"""

from typing import Optional, List, Dict, Any
import os
import re
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger("SafePred.DocumentParser")


class DocumentParser:
    """
    Parse documents in various formats and extract text content.
    
    Supports:
    - PDF files
    - Text files (TXT)
    - Word documents (DOC, DOCX)
    - Markdown files (MD)
    - HTML files
    """
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """
        Parse PDF file and extract text content.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text content
        """
        try:
            # Try PyPDF2
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                pass
            
            # Try pdfplumber
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                pass
            
            # Try pymupdf (fitz)
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text
            except ImportError:
                pass
            
            raise ImportError(
                "No PDF parsing library found. Install one of: PyPDF2, pdfplumber, or PyMuPDF"
            )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {file_path}: {e}")
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        """
        Parse text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            File content as string
        """
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Failed to decode text file {file_path}")
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """
        Parse DOCX file and extract text content.
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            Extracted text content
        """
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            raise ImportError("python-docx library not found. Install with: pip install python-docx")
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX {file_path}: {e}")
    
    @staticmethod
    def parse_doc(file_path: str) -> str:
        """
        Parse DOC file (older Word format).
        
        Args:
            file_path: Path to DOC file
        
        Returns:
            Extracted text content
        """
        try:
            # Try using antiword (requires system installation)
            import subprocess
            result = subprocess.run(
                ['antiword', file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try python-docx (may not work for .doc)
            try:
                return DocumentParser.parse_docx(file_path)
            except:
                raise ValueError(
                    f"Failed to parse DOC {file_path}. "
                    "Install antiword or convert to DOCX format."
                )
    
    @staticmethod
    def parse_markdown(file_path: str) -> str:
        """
        Parse Markdown file.
        
        Args:
            file_path: Path to Markdown file
        
        Returns:
            File content as string
        """
        return DocumentParser.parse_txt(file_path)
    
    @staticmethod
    def parse_html(file_path: str) -> str:
        """
        Parse HTML file and extract text content.
        
        Args:
            file_path: Path to HTML file
        
        Returns:
            Extracted text content
        """
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
        except ImportError:
            # Fallback: simple regex-based extraction
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', content)
                return text
        except Exception as e:
            raise ValueError(f"Failed to parse HTML {file_path}: {e}")
    
    @staticmethod
    def parse_file(file_path: str) -> str:
        """
        Auto-detect file format and parse.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return DocumentParser.parse_pdf(str(file_path))
        elif suffix == '.txt':
            return DocumentParser.parse_txt(str(file_path))
        elif suffix == '.docx':
            return DocumentParser.parse_docx(str(file_path))
        elif suffix == '.doc':
            return DocumentParser.parse_doc(str(file_path))
        elif suffix in ['.md', '.markdown']:
            return DocumentParser.parse_markdown(str(file_path))
        elif suffix in ['.html', '.htm']:
            return DocumentParser.parse_html(str(file_path))
        else:
            # Try as text file
            return DocumentParser.parse_txt(str(file_path))
    
    @staticmethod
    def parse_directory(directory: str, pattern: str = "*.*") -> Dict[str, str]:
        """
        Parse all matching files in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
        
        Returns:
            Dictionary mapping file paths to extracted text
        """
        results = {}
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        for file_path in dir_path.glob(pattern):
            try:
                text = DocumentParser.parse_file(str(file_path))
                results[str(file_path)] = text
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
        
        return results

