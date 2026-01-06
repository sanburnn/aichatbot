from pypdf import PdfReader
import docx
from pathlib import Path


def load_file(file_path: str) -> str:
    """
    Load file content and return plain text.
    Supported: PDF, TXT, DOCX
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix == ".txt":
        return _load_txt(path)

    if suffix == ".docx":
        return _load_docx(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
