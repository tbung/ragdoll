import io
import logging
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import rich
import tomllib
from docling_core.types.doc.document import DoclingDocument
from fastembed import TextEmbedding
from pydantic.dataclasses import dataclass
from pyzotero import zotero
from qdrant_client import QdrantClient
from qdrant_client.fastembed_common import QueryResponse
from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.progress import track

logger = logging.getLogger(__name__)

sentence_split_regex = re.compile(r"(?<=[.?!])\s+")


@dataclass
class ZoteroConfig:
    library_id: int
    library_type: str
    api_key: str


@dataclass
class Config:
    zotero: ZoteroConfig


def query_grouped(
    self: QdrantClient,
    collection_name: str,
    query_text: str,
    query_filter: models.Filter | None = None,
    limit: int = 10,
    group_size: int = 1,
    **kwargs: Any,
) -> list[QueryResponse]:
    embedding_model_inst = self._get_or_init_model(
        model_name=self.embedding_model_name, deprecated=True
    )
    embeddings = list(embedding_model_inst.query_embed(query=query_text))
    query_vector = embeddings[0].tolist()

    if self.sparse_embedding_model_name is None:
        return self._scored_points_to_query_responses(
            self.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name=self.get_vector_field_name(), vector=query_vector
                ),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                **kwargs,
            )
        )

    sparse_embedding_model_inst = self._get_or_init_sparse_model(
        model_name=self.sparse_embedding_model_name, deprecated=True
    )
    sparse_vector = list(sparse_embedding_model_inst.query_embed(query=query_text))[0]
    sparse_query_vector = models.SparseVector(
        indices=sparse_vector.indices.tolist(),
        values=sparse_vector.values.tolist(),
    )

    dense_group_response = [
        p
        for g in self.search_groups(
            collection_name,
            query_vector=models.NamedVector(
                name=self.get_vector_field_name(),
                vector=query_vector,
            ),
            group_by="key",
            query_filter=query_filter,
            limit=limit,
            group_size=group_size,
            with_payload=True,
            **kwargs,
        ).groups
        for p in g.hits
    ]

    sparse_group_response = [
        p
        for g in self.search_groups(
            collection_name,
            query_vector=models.NamedSparseVector(
                name=self.get_sparse_vector_field_name() or "",
                vector=sparse_query_vector,
            ),
            group_by="key",
            query_filter=query_filter,
            limit=limit,
            group_size=group_size,
            with_payload=True,
            **kwargs,
        ).groups
        for p in g.hits
    ]

    return self._scored_points_to_query_responses(
        reciprocal_rank_fusion(
            [dense_group_response, sparse_group_response], limit=limit
        )
    )


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
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.io import DocumentStream

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # pick what you need
    item: dict[str, Any]
    for item in zot.children(key):  # type: ignore
        if (
            item["data"]["itemType"] == "attachment"
            and item["data"]["contentType"] == "application/pdf"
        ):
            filename: str = item["data"]["filename"]
            with io.BytesIO(zot.file(item["key"])) as file:  # type: ignore
                source = DocumentStream(name=filename, stream=file)
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        )
                    }
                )
                return converter.convert(source).document

    return None


def sync(config: Config, client: QdrantClient):
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    logger.info("Starting Zotero Client")
    zot = zotero.Zotero(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
        local=True,
    )
    logger.info("Zotero Client started")

    items = cast(list[dict[str, Any]], zot.top(tag="project-clip-gmm"))
    for item in track(items):
        logger.info(
            f'Loading Zotero entry "{item["data"]["title"]}" ({item["data"]["key"]})'
        )
        if (
            client.count(
                "zotero",
                count_filter=models.Filter(
                    must=models.FieldCondition(
                        key="key",
                        match=models.MatchValue(value=item["data"]["key"]),
                    )
                ),
            ).count
            > 0
        ):
            logger.info(f"Zotero item {item['data']['key']} already indexed")
            continue

        document = get_document(zot, item["data"]["key"])

        if document is None:
            logger.info(f"Zotero item {item['data']['key']} has no PDF attachment")
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

        _ = client.add(collection_name="zotero", documents=chunks, metadata=metadata)


def main():
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    with Path("./config.toml").open("rb") as f:
        config = Config(**tomllib.load(f))

    logger.info("Starting Qdrant Client")
    client = QdrantClient("http://localhost:6333")
    client.set_model("BAAI/bge-small-en-v1.5")
    client.set_sparse_model("prithivida/Splade_PP_en_v1")
    logger.info("Qdrant Client started")

    sync(config, client)

    points = query_grouped(
        client,
        collection_name="zotero",
        query_text="improving clip embeddings",
        limit=10,
    )

    for i, point in enumerate(points):
        md = Markdown(f"""
# {point.metadata["title"]}, page {point.metadata["doc_items"][0]["prov"][0]["page_no"]}

## {point.document.replace("\n", "\n\n")}

---
                      """)
        rich.print(md)


if __name__ == "__main__":
    main()
