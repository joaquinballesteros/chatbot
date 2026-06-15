# ingest_java.py
import os, re, pathlib
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import streamlit as st

from faiss_local import INDEX_PATH, JAVA_ROOT, MAX_CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL



print("INFO: Cargando variables de entorno desde el fichero .env...")


# === utilidades simples para extraer metadatos Java ===
PKG_RE   = re.compile(r'^\s*package\s+([a-zA-Z_0-9\.]+)\s*;', re.MULTILINE)
CLASS_RE = re.compile(r'^\s*(public\s+|protected\s+|private\s+)?(abstract\s+|final\s+)?class\s+([A-Za-z_0-9]+)', re.MULTILINE)
INTERFACE_RE = re.compile(r'^\s*(public\s+|protected\s+|private\s+)?interface\s+([A-Za-z_0-9]+)', re.MULTILINE)
METHOD_RE = re.compile(r'(public|protected|private|\s)\s+[\w\<\>\[\]]+\s+([A-Za-z_0-9]+)\s*\(', re.MULTILINE)
ABSTRACT_RE = re.compile(r'^\s*(public\s+)?(abstract\s+)class\s+([A-Za-z_0-9]+)', re.MULTILINE)
ENUM_RE = re.compile(r'^\s*(public\s+)?enum\s+([A-Za-z_0-9]+)', re.MULTILINE)

def extract_java_metadata(code: str) -> Dict[str, str]:
    pkg = PKG_RE.search(code)
    cls = CLASS_RE.search(code)
    itf = INTERFACE_RE.search(code)
    abs_cls = ABSTRACT_RE.search(code)
    enum = ENUM_RE.search(code)
    methods = METHOD_RE.findall(code)

    return {
        "package": pkg.group(1) if pkg else "",
        "class_or_interface": (
            cls.group(3) if cls
            else (itf.group(2) if itf
            else (abs_cls.group(3) if abs_cls
            else (enum.group(2) if enum else ""))))
        ,
        "kind": (
            "class" if cls else
            "interface" if itf else
            "abstract_class" if abs_cls else
            "enum" if enum else ""
        ),
        "method_names": ", ".join(sorted(set([m[1] for m in methods]))) if methods else ""
    }


def collect_java_files(root: str) -> List[str]:
    return [str(p) for p in pathlib.Path(root).rglob("*.java")]

def main():
    if not st.secrets.get("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY (exporta la var de entorno o ajusta el script)")

    files = collect_java_files(JAVA_ROOT)
    if not files:
        raise RuntimeError(f"No se encontraron .java en {JAVA_ROOT}")

    # Splitter orientado a Java (mejor que texto plano)
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents: List[Document] = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        meta = extract_java_metadata(code)
        # Troceamos respetando estructura; el splitter devuelve strings
        chunks = splitter.split_text(code)
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": os.path.relpath(fpath, JAVA_ROOT).replace("\\", "/"),
                    "repo_root": os.path.abspath(JAVA_ROOT),
                    "package": meta["package"],
                    "class": meta["class_or_interface"],
                    "methods": meta["method_names"],
                    "chunk_id": idx
                }
            )
            documents.append(doc)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Construir FAISS y persistir
    vs = FAISS.from_documents(documents, embeddings)
    vs.save_local(INDEX_PATH)
    print(f"✅ Índice creado con {len(documents)} chunks en {INDEX_PATH}")

if __name__ == "__main__":
    main()