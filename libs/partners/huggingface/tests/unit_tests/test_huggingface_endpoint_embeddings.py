"""Tests for Hugging Face endpoint embedding secrets in representations."""

from unittest.mock import patch

from langchain_core.load import dumpd
from langchain_huggingface.embeddings.huggingface_endpoint import (
    HuggingFaceEndpointEmbeddings,
)


def test_api_token_is_excluded_from_embedding_representations() -> None:
    token = "hf_test_secret_token"

    with (
        patch("huggingface_hub.InferenceClient"),
        patch("huggingface_hub.AsyncInferenceClient"),
    ):
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=token,
        )

    assert token not in repr(embeddings)
    assert token not in str(embeddings)
    assert token not in str(dumpd(embeddings))
