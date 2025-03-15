import json
from pypdf import PdfReader
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

def load_and_chunk_pdf(pdf_paths, chunk_size=512):
    """Loads PDFs and chunks them into smaller pieces, handling multiple files."""
    all_documents = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            sentences = nltk.sent_tokenize(text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > chunk_size:
                    all_documents.append({"text": current_chunk.strip(), "source": pdf_path})
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
            if current_chunk:
                all_documents.append({"text": current_chunk.strip(), "source": pdf_path})
    return all_documents

if __name__ == "__main__":
    pdf_paths = [
        "goog-2025.pdf",
        "goog-2024.pdf",
    ]
    chunk_sizes = [128, 256, 512, 1024]

    for chunk_size in chunk_sizes:
        documents = load_and_chunk_pdf(pdf_paths, chunk_size=chunk_size)
        output_file = f"chunks_{chunk_size}.json"
        with open(output_file, "w") as f:
            json.dump(documents, f, indent=4)
        print(f"Chunks saved to {output_file}")