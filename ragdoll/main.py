import io
import re
from pathlib import Path
from typing import Any, cast

from docling_core.types.doc.document import DoclingDocument
import numpy as np
import tomllib
import rich
from fastembed import TextEmbedding
from pydantic.dataclasses import dataclass
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from pyzotero import zotero
from qdrant_client import QdrantClient
from rich.progress import track

sentence_split_regex = re.compile(r"(?<=[.?!])\s+")


def to_chunks(text: str) -> list[str]:
    # split into groups of 3 sentences
    # merge groups with distances in the 95 percentile
    sentences: list[str] = sentence_split_regex.split(text)

    embedder = TextEmbedding()
    embeddings = np.array(list(embedder.embed(sentences)))
    print(embeddings.dtype)
    print(embeddings.shape)

    a = embeddings[1:, None, :]
    b = embeddings[:-1, :, None]
    n: np.ndarray = np.linalg.norm(embeddings, axis=1)
    distances = 1 - (a @ b).squeeze() / n[1:] / n[:-1]

    threshold = np.percentile(distances, 95)

    # since distances are to previous, indices are shifted by one
    breakpoints = list(np.argwhere(distances > threshold).squeeze() + 1)
    print(breakpoints)

    chunks = [
        " ".join(sentences[i:j])
        for i, j in zip([0] + breakpoints, breakpoints + [None])
    ]
    return chunks


def get_document(zot: zotero.Zotero, key: str) -> DoclingDocument | None:
    item: dict[str, Any]
    for item in zot.children(key):
        if (
            item["data"]["itemType"] == "attachment"
            and item["data"]["contentType"] == "application/pdf"
        ):
            filename: str = item["data"]["filename"]
            with io.BytesIO(zot.file(item["key"])) as file:
                source = DocumentStream(name=filename, stream=file)
                converter = DocumentConverter()
                return converter.convert(source).document

    return None


def sync(zot: zotero.Zotero, client: QdrantClient):
    items = cast(list[dict[str, Any]], zot.top(tag="project-clip-gmm"))
    for item in track(items):
        print(item["data"]["title"])
        document = get_document(zot, item["data"]["key"])

        if document is None:
            continue

        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []

        chunker = HybridChunker()
        for chunk in chunker.chunk(document):
            chunks.append(chunker.contextualize(chunk))
            metadata.append(
                chunk.meta.export_json_dict()
                | {"key": item["data"]["key"], "title": item["data"]["title"]}
            )
            # rich.print(chunker.contextualize(chunk))
            # rich.print(
            #     chunk.meta.export_json_dict()
            #     | {"key": item["data"]["key"], "title": item["data"]["title"]}
            # )

        # chunks = to_chunks(text)

        _ = client.add(collection_name="zotero", documents=chunks, metadata=metadata)


@dataclass
class ZoteroConfig:
    library_id: int
    library_type: str
    api_key: str


@dataclass
class Config:
    zotero: ZoteroConfig


def main():
    with Path("./config.toml").open("rb") as f:
        config = Config(**tomllib.load(f))

    client = QdrantClient("http://localhost:6333")
    client.set_model("BAAI/bge-small-en-v1.5")
    client.set_sparse_model("prithivida/Splade_PP_en_v1")

    zot = zotero.Zotero(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
        local=True,
    )

    sync(zot, client)
    # points = client.query(
    #     collection_name="zotero",
    #     query_text="improving clip embeddings",
    #     limit=10,
    # )

    for i, point in enumerate(points):
        print(point.document)
        print(point.metadata)
        print()


if __name__ == "__main__":
    main()
