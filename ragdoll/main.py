import io
import logging
import re
import urllib.parse
from typing import Any, Generator, cast

import rich
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.document import DoclingDocument
from jsonargparse import ArgumentParser
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


def _docling_convert_batched(filename: str, file: bytes) -> list[DoclingDocument]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.io import DocumentStream

    documents: list[DoclingDocument] = []
    source = DocumentStream(name=filename, stream=io.BytesIO(file))
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=False)
            )
        }
    )

    first_page = 1
    batch_size = 100
    conversion = converter.convert(
        source,
        page_range=(first_page, first_page + batch_size - 1),
        raises_on_error=False,
    )
    while conversion.status in [
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ]:
        documents.append(conversion.document)
        first_page += batch_size
        source = DocumentStream(name=filename, stream=io.BytesIO(file))
        conversion = converter.convert(
            source,
            page_range=(first_page, first_page + batch_size - 1),
            raises_on_error=False,
        )

    return documents


def get_document(
    zot: zotero.Zotero, key: str
) -> tuple[list[DoclingDocument], str] | tuple[None, None]:
    item: dict[str, Any]
    for item in zot.children(key):  # type: ignore
        if (
            item["data"]["itemType"] == "attachment"
            and item["data"]["contentType"] == "application/pdf"
        ):
            filename: str = item["data"]["filename"]
            file: bytes = zot.file(item["key"])  # type: ignore
            return _docling_convert_batched(filename, file), item["key"]

    return None, None


def _get_qdrant() -> QdrantClient:
    logger.info("Starting Qdrant Client")
    client = QdrantClient("http://localhost:6333")
    client.set_model("BAAI/bge-small-en-v1.5")
    client.set_sparse_model("prithivida/Splade_PP_en_v1")
    logger.info("Qdrant Client started")
    return client


def zotero_items_iter(zot: zotero.Zotero) -> Generator[dict[str, Any], Any, Any]:
    start = 0
    items: list[dict[str, Any]] = cast(list[dict[str, Any]], zot.top(limit=20, tag="-no-index"))
    while len(items) > 0:
        for item in items:
            yield item
        start += 20
        items = cast(list[dict[str, Any]], zot.top(start=start, limit=20, tag="-no-index"))


def sync(config: Config):
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    client = _get_qdrant()
    logger.info("Starting Zotero Client")
    zot = zotero.Zotero(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
        local=True,
    )
    logger.info("Zotero Client started")

    num_items = zot.num_items()
    for item in track(zotero_items_iter(zot), total=num_items):
        logger.info(
            f'Loading Zotero entry "{item["data"]["title"]}" ({item["data"]["key"]})'
        )

        if client.collection_exists("zotero") and (
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

        documents, attachment_key = get_document(zot, item["data"]["key"])

        if documents is None:
            logger.info(f"Zotero item {item['data']['key']} has no PDF attachment")
            continue

        chunker = HybridChunker()
        for document in documents:
            for chunk in chunker.chunk(document):
                _ = client.add(
                    collection_name="zotero",
                    documents=[chunker.contextualize(chunk)],
                    metadata=[
                        chunk.meta.export_json_dict()
                        | {
                            "key": item["data"]["key"],
                            "title": item["data"]["title"],
                            "attachment_key": attachment_key,
                        }
                    ],
                )


def query(message: str, limit: int = 10, group_size: int = 1):
    client = _get_qdrant()
    points = query_grouped(
        client,
        collection_name="zotero",
        query_text=message,
        limit=limit,
        group_size=group_size,
    )

    for i, point in enumerate(points):
        link = urllib.parse.quote(
            f"/home/tillb/Zotero/storage/{point.metadata['attachment_key']}/{point.metadata['origin']['filename']}"
        )
        md = Markdown(
            f"""
# {point.metadata["title"]}, page {point.metadata["doc_items"][0]["prov"][0]["page_no"]}

## {point.document.replace("\n", "\n\n")}

---
                      """,
            hyperlinks=True,
        )
        rich.print(
            f"[blue][underline][link=file://{link}]Source: {point.metadata['title']}[/link][/underline][/blue]"
        )
        rich.print(md)


def main():
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = ArgumentParser(parser_mode="toml")
    subcommands = parser.add_subcommands()

    subparser = ArgumentParser(parser_mode="toml")
    subparser.add_function_arguments(sync)
    subcommands.add_subcommand("sync", subparser)

    subparser = ArgumentParser(parser_mode="toml")
    subparser.add_function_arguments(query, as_positional=True)
    subcommands.add_subcommand("query", subparser)

    args = parser.parse_args()
    if args.subcommand == "sync":
        config = Config(**args.sync.config.as_dict())
        sync(config)
    elif args.subcommand == "query":
        query(args.query.message)


if __name__ == "__main__":
    main()
