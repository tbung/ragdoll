import re
from pathlib import Path

import numpy as np
import tomllib
from fastembed import TextEmbedding
from pydantic.dataclasses import dataclass
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


def get_fulltext(zot: zotero.Zotero, key: str) -> tuple[str, str] | tuple[None, None]:
    for item in zot.children(key):
        if (
            item["data"]["itemType"] == "attachment"
            and item["data"]["contentType"] == "application/pdf"
        ):
            text = zot.fulltext_item(item["key"])
            return item["data"]["filename"], text["content"]

    return None, None


def sync(zot: zotero.Zotero, client: QdrantClient):
    items = zot.top(tag="project-clip-gmm")
    for item in track(items):
        print(item["data"]["title"])
        filename, text = get_fulltext(zot, item["data"]["key"])

        if text is None or filename is None:
            continue

        chunks = to_chunks(text)

        _ = client.add(
            collection_name="zotero",
            documents=chunks,
            metadata=[
                {
                    "key": item["data"]["key"],
                    "title": item["data"]["title"],
                    "filename": filename,
                }
                for _ in chunks
            ],
        )


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
    zot = zotero.Zotero(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
        local=True,
    )

    sync(zot, client)


if __name__ == "__main__":
    main()
