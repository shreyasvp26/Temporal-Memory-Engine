"""Contract tests for LLM client/gateway and Embedder protocols."""

from typing import TYPE_CHECKING

from memory_engine.embed.cached import CachedEmbedder
from memory_engine.embed.fastembed_adapter import FastEmbedAdapter
from memory_engine.llm.gateway import DefaultLLMGateway
from memory_engine.llm.gemini import GeminiClient
from memory_engine.llm.groq import GroqClient
from memory_engine.llm.ollama import OllamaClient

if TYPE_CHECKING:
    from memory_engine.ports.embed import Embedder
    from memory_engine.ports.llm import LLMClient, LLMGateway


def test_llm_client_protocol_conformance() -> None:
    """Verify concrete clients conform to the LLMClient protocol."""
    groq: LLMClient = GroqClient()
    gemini: LLMClient = GeminiClient()
    ollama: LLMClient = OllamaClient()

    for client in (groq, gemini, ollama):
        assert hasattr(client, 'name') and isinstance(client.name, str)
        assert hasattr(client, 'complete') and callable(client.complete)
        assert hasattr(client, 'stream') and callable(client.stream)


def test_llm_gateway_protocol_conformance() -> None:
    """Verify DefaultLLMGateway conforms to the LLMGateway protocol."""
    gateway: LLMGateway = DefaultLLMGateway(providers={})
    assert hasattr(gateway, 'complete_text') and callable(gateway.complete_text)
    assert hasattr(gateway, 'complete_json') and callable(gateway.complete_json)
    assert hasattr(gateway, 'stream_text') and callable(gateway.stream_text)


def test_embedder_protocol_conformance() -> None:
    """Verify FastEmbedAdapter and CachedEmbedder conform to Embedder protocol."""
    fastembed: Embedder = FastEmbedAdapter()
    cached: Embedder = CachedEmbedder(inner=fastembed)

    for embedder in (fastembed, cached):
        assert hasattr(embedder, 'model_id') and isinstance(embedder.model_id, str)
        assert hasattr(embedder, 'dim') and isinstance(embedder.dim, int)
        assert hasattr(embedder, 'embed_documents') and callable(embedder.embed_documents)
        assert hasattr(embedder, 'embed_queries') and callable(embedder.embed_queries)
