"""Configuration settings for Temporal Memory Engine."""

from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True, slots=True)
class IngestSettings:
    """Settings subset for ingestion pipelines and gates."""

    debounce_s: int = 15
    notes_debounce_s: int = 2
    extract_assistant: bool = False

    @property
    def ingest_debounce_s(self) -> int:
        """Alias for debounce_s."""
        return self.debounce_s


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Settings subset for background workers and task execution."""

    concurrency: int = 2
    lease_s: int = 120
    heartbeat_s: int = 30
    max_attempts: int = 3
    decay_hour_utc: int = 2

    @property
    def worker_concurrency(self) -> int:
        """Alias for concurrency."""
        return self.concurrency


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Settings subset for retrieval pipelines, lanes, fusion, and decay."""

    plan_timeout_ms: int = 1500
    lane_timeout_ms: int = 1200
    source_lane: bool = True
    graph_expansion: bool = True
    rerank: bool = False
    rrf_k: int = 60
    decay_enabled: bool = True
    decay_floor: float = 0.3
    halflife_slow: int = 365
    halflife_normal: int = 90
    halflife_fast: int = 14
    abstain_min_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """Settings subset for context assembly budgets."""

    fact_tokens: int = 2200
    source_tokens: int = 800
    eval_tokens: int = 3000

    @property
    def ctx_fact_tokens(self) -> int:
        """Alias for fact_tokens."""
        return self.fact_tokens

    @property
    def ctx_source_tokens(self) -> int:
        """Alias for source_tokens."""
        return self.source_tokens

    @property
    def eval_ctx_tokens(self) -> int:
        """Alias for eval_tokens."""
        return self.eval_tokens


@dataclass(frozen=True, slots=True)
class ChatSettings:
    """Settings subset for interactive chat orchestration."""

    plan_timeout_ms: int = 1500
    access_tracking: bool = True


@dataclass(frozen=True, slots=True)
class ExtractSettings:
    """Settings subset for fact extraction."""

    context_turns: int = 4
    min_confidence: float = 0.5
    extract_assistant: bool = False


@dataclass(frozen=True, slots=True)
class EntitySettings:
    """Settings subset for entity resolution and ambiguity thresholds."""

    auto_merge: float = 0.92
    ambig_low: float = 0.80
    ambiguity_judge: bool = False

    @property
    def ent_auto_merge(self) -> float:
        """Alias for auto_merge."""
        return self.auto_merge

    @property
    def ent_ambig_low(self) -> float:
        """Alias for ambig_low."""
        return self.ambig_low


@dataclass(frozen=True, slots=True)
class JwtSettings:
    """Settings subset for JWT authentication."""

    mode: str = 'jwks'
    jwks_url: str | None = None
    secret: str | None = None
    issuer: str | None = None
    audience: str = 'authenticated'

    @property
    def jwt_mode(self) -> str:
        """Alias for mode."""
        return self.mode

    @property
    def jwt_jwks_url(self) -> str | None:
        """Alias for jwks_url."""
        return self.jwks_url

    @property
    def jwt_secret(self) -> str | None:
        """Alias for secret."""
        return self.secret

    @property
    def jwt_issuer(self) -> str | None:
        """Alias for issuer."""
        return self.issuer

    @property
    def jwt_audience(self) -> str:
        """Alias for audience."""
        return self.audience


class Settings(BaseSettings):
    """Global immutable configuration loaded from environment variables (SB_ prefix)."""

    model_config = SettingsConfigDict(
        env_prefix='SB_',
        frozen=True,
        extra='ignore',
    )

    # General / Infrastructure
    env: str = 'dev'
    database_url_api: str | None = None
    database_url_worker: str | None = None
    web_origin: str | None = None

    # Logging
    log_level: str = 'INFO'
    log_content: bool = False

    # Mode / Embedding
    local_only: bool = False
    embedding_model_id: str = 'nomic-ai/nomic-embed-text-v1.5'
    embedding_mode: str = 'local'
    embed_remote_url: str | None = None
    embed_remote_key: str | None = None

    # Auth / JWT
    jwt_mode: str = 'jwks'
    jwt_jwks_url: str | None = None
    jwt_secret: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str = 'authenticated'

    # LLM Chains & Providers
    llm_fast_chain: str = 'groq,gemini,ollama'
    llm_synth_chain: str = 'gemini,groq'
    groq_api_key: str | None = None
    groq_model_fast: str = 'llama-3.1-8b-instant'
    gemini_api_key: str | None = None
    gemini_model_fast: str = 'gemini-1.5-flash'
    gemini_model_synth: str = 'gemini-1.5-flash'
    ollama_url: str = 'http://localhost:11434'
    ollama_model: str = 'llama3.1:8b'

    # Provider Rate Limits (token buckets)
    rate_groq_rpm: int = 30
    rate_groq_tpm: int = 30_000
    rate_groq_rpd: int = 14_400
    rate_gemini_rpm: int = 15
    rate_gemini_tpm: int = 1_000_000
    rate_gemini_rpd: int = 1_500

    # Ingestion & Extraction
    ingest_debounce_s: int = 15
    notes_debounce_s: int = 2
    context_turns: int = 4
    min_confidence: float = 0.5
    extract_assistant: bool = False
    ent_auto_merge: float = 0.92
    ent_ambig_low: float = 0.80
    ambiguity_judge: bool = False

    # Worker
    worker_concurrency: int = 2
    lease_s: int = 120
    heartbeat_s: int = 30
    max_attempts: int = 3
    decay_hour_utc: int = 2

    # Retrieval, Lanes, Decay & Fusion
    plan_timeout_ms: int = 1500
    lane_timeout_ms: int = 1200
    source_lane: bool = True
    graph_expansion: bool = True
    rerank: bool = False
    rrf_k: int = 60
    decay_enabled: bool = True
    decay_floor: float = 0.3
    halflife_slow: int = 365
    halflife_normal: int = 90
    halflife_fast: int = 14

    # Context & Abstention
    ctx_fact_tokens: int = 2200
    ctx_source_tokens: int = 800
    abstain_min_score: float = 0.0
    access_tracking: bool = True

    # Database Pool & Partitioning
    pool_api_max: int = 10
    pool_worker_max: int = 6
    read_conn_limit: int = 6
    write_conn_limit: int = 4

    # Per-user API Rate Limits (per minute)
    rate_chat_per_min: int = 30
    rate_search_per_min: int = 120
    rate_ingest_per_min: int = 30
    rate_default_per_min: int = 120

    # Observability
    metrics_enabled: bool = False
    trace_backend: str = 'none'
    trace_redact: bool = True

    # Evaluation
    eval_ctx_tokens: int = 3000
    eval_cache_dir: str = 'eval/.cache'

    @property
    def fast_chain(self) -> tuple[str, ...]:
        """Parsed tuple of provider names for fast LLM operations."""
        return tuple(p.strip() for p in self.llm_fast_chain.split(',') if p.strip())

    @property
    def synth_chain(self) -> tuple[str, ...]:
        """Parsed tuple of provider names for synthesis operations."""
        return tuple(p.strip() for p in self.llm_synth_chain.split(',') if p.strip())

    @property
    def ingest(self) -> IngestSettings:
        """Ingest-specific sub-settings view."""
        return IngestSettings(
            debounce_s=self.ingest_debounce_s,
            notes_debounce_s=self.notes_debounce_s,
            extract_assistant=self.extract_assistant,
        )

    @property
    def worker(self) -> WorkerSettings:
        """Worker-specific sub-settings view."""
        return WorkerSettings(
            concurrency=self.worker_concurrency,
            lease_s=self.lease_s,
            heartbeat_s=self.heartbeat_s,
            max_attempts=self.max_attempts,
            decay_hour_utc=self.decay_hour_utc,
        )

    @property
    def retrieval(self) -> RetrievalSettings:
        """Retrieval-specific sub-settings view."""
        return RetrievalSettings(
            plan_timeout_ms=self.plan_timeout_ms,
            lane_timeout_ms=self.lane_timeout_ms,
            source_lane=self.source_lane,
            graph_expansion=self.graph_expansion,
            rerank=self.rerank,
            rrf_k=self.rrf_k,
            decay_enabled=self.decay_enabled,
            decay_floor=self.decay_floor,
            halflife_slow=self.halflife_slow,
            halflife_normal=self.halflife_normal,
            halflife_fast=self.halflife_fast,
            abstain_min_score=self.abstain_min_score,
        )

    @property
    def context(self) -> ContextSettings:
        """Context assembly sub-settings view."""
        return ContextSettings(
            fact_tokens=self.ctx_fact_tokens,
            source_tokens=self.ctx_source_tokens,
            eval_tokens=self.eval_ctx_tokens,
        )

    @property
    def chat(self) -> ChatSettings:
        """Chat orchestrator sub-settings view."""
        return ChatSettings(
            plan_timeout_ms=self.plan_timeout_ms,
            access_tracking=self.access_tracking,
        )

    @property
    def extract(self) -> ExtractSettings:
        """Extraction sub-settings view."""
        return ExtractSettings(
            context_turns=self.context_turns,
            min_confidence=self.min_confidence,
            extract_assistant=self.extract_assistant,
        )

    @property
    def entity(self) -> EntitySettings:
        """Entity resolution sub-settings view."""
        return EntitySettings(
            auto_merge=self.ent_auto_merge,
            ambig_low=self.ent_ambig_low,
            ambiguity_judge=self.ambiguity_judge,
        )

    @property
    def jwt(self) -> JwtSettings:
        """JWT authentication sub-settings view."""
        return JwtSettings(
            mode=self.jwt_mode,
            jwks_url=self.jwt_jwks_url,
            secret=self.jwt_secret,
            issuer=self.jwt_issuer,
            audience=self.jwt_audience,
        )
