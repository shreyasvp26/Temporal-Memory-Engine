"""Unit tests for configuration and observability/logging (M0-06)."""

import hashlib
import io
import json
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest
from memory_engine.config import (
    ChatSettings,
    ContextSettings,
    EntitySettings,
    ExtractSettings,
    IngestSettings,
    JwtSettings,
    RetrievalSettings,
    Settings,
    WorkerSettings,
)
from memory_engine.domain.errors import ValidationFailed
from memory_engine.obs.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_context,
    get_logger,
    redact_content,
    user_hash,
)
from pydantic import ValidationError


class TestConfigSettings:
    """Verify default Settings, sub-settings views, immutability, and env overrides."""

    def test_default_settings(self) -> None:
        """Verify defaults match LLD Section 21."""
        cfg = Settings()

        # Core / DB / Web
        assert cfg.env == 'dev'
        assert cfg.database_url_api is None
        assert cfg.database_url_worker is None
        assert cfg.web_origin is None

        # Logging & Mode
        assert cfg.log_level == 'INFO'
        assert cfg.log_content is False
        assert cfg.local_only is False

        # Embeddings & Auth
        assert cfg.embedding_model_id == 'nomic-ai/nomic-embed-text-v1.5'
        assert cfg.embedding_mode == 'local'
        assert cfg.embed_remote_url is None
        assert cfg.embed_remote_key is None
        assert cfg.jwt_mode == 'jwks'
        assert cfg.jwt_audience == 'authenticated'

        # LLM Chains & Providers
        assert cfg.llm_fast_chain == 'groq,gemini,ollama'
        assert cfg.fast_chain == ('groq', 'gemini', 'ollama')
        assert cfg.llm_synth_chain == 'gemini,groq'
        assert cfg.synth_chain == ('gemini', 'groq')
        assert cfg.groq_model_fast == 'llama-3.1-8b-instant'
        assert cfg.gemini_model_fast == 'gemini-1.5-flash'
        assert cfg.gemini_model_synth == 'gemini-1.5-flash'
        assert cfg.ollama_url == 'http://localhost:11434'
        assert cfg.ollama_model == 'llama3.1:8b'

        # Provider Rate Limits
        assert cfg.rate_groq_rpm == 30
        assert cfg.rate_groq_tpm == 30_000
        assert cfg.rate_groq_rpd == 14_400
        assert cfg.rate_gemini_rpm == 15
        assert cfg.rate_gemini_tpm == 1_000_000
        assert cfg.rate_gemini_rpd == 1_500

        # Ingest, Extraction & Entity
        assert cfg.ingest_debounce_s == 15
        assert cfg.notes_debounce_s == 2
        assert cfg.context_turns == 4
        assert cfg.min_confidence == 0.5
        assert cfg.extract_assistant is False
        assert cfg.ent_auto_merge == 0.92
        assert cfg.ent_ambig_low == 0.80
        assert cfg.ambiguity_judge is False

        # Worker
        assert cfg.worker_concurrency == 2
        assert cfg.lease_s == 120
        assert cfg.heartbeat_s == 30
        assert cfg.max_attempts == 3
        assert cfg.decay_hour_utc == 2

        # Retrieval, Fusion & Decay
        assert cfg.plan_timeout_ms == 1500
        assert cfg.lane_timeout_ms == 1200
        assert cfg.source_lane is True
        assert cfg.graph_expansion is True
        assert cfg.rerank is False
        assert cfg.rrf_k == 60
        assert cfg.decay_enabled is True
        assert cfg.decay_floor == 0.3
        assert cfg.halflife_slow == 365
        assert cfg.halflife_normal == 90
        assert cfg.halflife_fast == 14

        # Context Budget & API limits
        assert cfg.ctx_fact_tokens == 2200
        assert cfg.ctx_source_tokens == 800
        assert cfg.abstain_min_score == 0.0
        assert cfg.access_tracking is True
        assert cfg.pool_api_max == 10
        assert cfg.pool_worker_max == 6
        assert cfg.read_conn_limit == 6
        assert cfg.write_conn_limit == 4
        assert cfg.rate_chat_per_min == 30
        assert cfg.rate_search_per_min == 120
        assert cfg.rate_ingest_per_min == 30
        assert cfg.rate_default_per_min == 120

        # Observability & Eval
        assert cfg.metrics_enabled is False
        assert cfg.trace_backend == 'none'
        assert cfg.trace_redact is True
        assert cfg.eval_ctx_tokens == 3000
        assert cfg.eval_cache_dir == 'eval/.cache'

    def test_sub_settings_views(self) -> None:
        """Verify sub-settings objects expose correct slices and aliases."""
        cfg = Settings()

        assert isinstance(cfg.ingest, IngestSettings)
        assert cfg.ingest.debounce_s == 15
        assert cfg.ingest.notes_debounce_s == 2
        assert cfg.ingest.ingest_debounce_s == 15

        assert isinstance(cfg.worker, WorkerSettings)
        assert cfg.worker.concurrency == 2
        assert cfg.worker.worker_concurrency == 2
        assert cfg.worker.lease_s == 120
        assert cfg.worker.heartbeat_s == 30

        assert isinstance(cfg.retrieval, RetrievalSettings)
        assert cfg.retrieval.plan_timeout_ms == 1500
        assert cfg.retrieval.lane_timeout_ms == 1200
        assert cfg.retrieval.rrf_k == 60
        assert cfg.retrieval.decay_floor == 0.3

        assert isinstance(cfg.context, ContextSettings)
        assert cfg.context.fact_tokens == 2200
        assert cfg.context.ctx_fact_tokens == 2200
        assert cfg.context.source_tokens == 800
        assert cfg.context.ctx_source_tokens == 800
        assert cfg.context.eval_tokens == 3000
        assert cfg.context.eval_ctx_tokens == 3000

        assert isinstance(cfg.chat, ChatSettings)
        assert cfg.chat.plan_timeout_ms == 1500
        assert cfg.chat.access_tracking is True

        assert isinstance(cfg.extract, ExtractSettings)
        assert cfg.extract.context_turns == 4
        assert cfg.extract.min_confidence == 0.5
        assert cfg.extract.extract_assistant is False

        assert isinstance(cfg.entity, EntitySettings)
        assert cfg.entity.auto_merge == 0.92
        assert cfg.entity.ent_auto_merge == 0.92
        assert cfg.entity.ambig_low == 0.80
        assert cfg.entity.ent_ambig_low == 0.80

        assert isinstance(cfg.jwt, JwtSettings)
        assert cfg.jwt.mode == 'jwks'
        assert cfg.jwt.jwt_mode == 'jwks'
        assert cfg.jwt.audience == 'authenticated'

    def test_settings_immutability(self) -> None:
        """Verify that Settings and sub-settings are frozen/immutable."""
        cfg = Settings()
        with pytest.raises(ValidationError):
            cfg.env = 'prod'

        with pytest.raises(FrozenInstanceError):
            cfg.ingest.debounce_s = 30  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            cfg.worker.concurrency = 10  # type: ignore[misc]

    def test_env_var_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify environment variables with prefix SB_ override defaults."""
        monkeypatch.setenv('SB_ENV', 'prod')
        monkeypatch.setenv('SB_DATABASE_URL_API', 'postgresql://app_api:secret@db:5432/secondbrain')
        monkeypatch.setenv('SB_LOG_LEVEL', 'DEBUG')
        monkeypatch.setenv('SB_LOG_CONTENT', 'true')
        monkeypatch.setenv('SB_LOCAL_ONLY', 'true')
        monkeypatch.setenv('SB_WORKER_CONCURRENCY', '8')
        monkeypatch.setenv('SB_DECAY_FLOOR', '0.45')
        monkeypatch.setenv('SB_ENT_AUTO_MERGE', '0.96')
        monkeypatch.setenv('SB_RERANK', 'true')
        monkeypatch.setenv('SB_LLM_FAST_CHAIN', 'gemini,ollama')

        cfg = Settings()
        assert cfg.env == 'prod'
        assert cfg.database_url_api == 'postgresql://app_api:secret@db:5432/secondbrain'
        assert cfg.log_level == 'DEBUG'
        assert cfg.log_content is True
        assert cfg.local_only is True
        assert cfg.worker_concurrency == 8
        assert cfg.worker.concurrency == 8
        assert cfg.decay_floor == 0.45
        assert cfg.retrieval.decay_floor == 0.45
        assert cfg.ent_auto_merge == 0.96
        assert cfg.entity.auto_merge == 0.96
        assert cfg.rerank is True
        assert cfg.fast_chain == ('gemini', 'ollama')


class TestLoggingAndObservability:
    """Verify user_hash, context propagation, log formatting, and redaction."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_user_hash_returns_12_hex_chars(self) -> None:
        """Verify user_hash returns first 12 hex chars of sha256."""
        uid_str = '4a8b7573-0545-4235-8663-899478f69201'
        uid_obj = UUID(uid_str)

        expected = hashlib.sha256(uid_str.encode('utf-8')).hexdigest()[:12]
        assert len(expected) == 12

        assert user_hash(uid_str) == expected
        assert user_hash(uid_obj) == expected
        assert user_hash(uid_str) == user_hash(uid_obj)

        other_uid = uuid4()
        assert user_hash(other_uid) != expected

    def test_context_binding_and_clearing(self) -> None:
        """Verify bind_context, get_context, and clear_context behavior."""
        req_id = 'req-1234'
        u_id = uuid4()
        j_id = uuid4()
        c_id = uuid4()

        assert get_context() == {
            'request_id': None,
            'user_id': None,
            'job_id': None,
            'conversation_id': None,
        }

        bind_context(
            request_id=req_id,
            user_id=u_id,
            job_id=j_id,
            conversation_id=c_id,
        )

        ctx = get_context()
        assert ctx['request_id'] == req_id
        assert ctx['user_id'] == str(u_id)
        assert ctx['job_id'] == str(j_id)
        assert ctx['conversation_id'] == str(c_id)

        clear_context()
        assert get_context() == {
            'request_id': None,
            'user_id': None,
            'job_id': None,
            'conversation_id': None,
        }

    def test_structured_logging_json_output(self) -> None:
        """Verify JSON output contains mandatory fields and bound context."""
        stream = io.StringIO()
        configure_logging(service='api', level='INFO', file=stream)
        log = get_logger('test')

        u_id = uuid4()
        bind_context(request_id='req-999', user_id=u_id)

        log.info('message_handled', duration_ms=45)

        raw = stream.getvalue().strip()
        record = json.loads(raw)

        assert record['event'] == 'message_handled'
        assert record['level'] == 'info'
        assert record['service'] == 'api'
        assert record['request_id'] == 'req-999'
        assert record['user_hash'] == user_hash(u_id)
        assert record['duration_ms'] == 45
        assert 'ts' in record
        # Raw user_id must NOT be exposed
        assert 'user_id' not in record

    def test_worker_job_id_in_log_output(self) -> None:
        """Verify worker service logs contain job_id when bound."""
        stream = io.StringIO()
        configure_logging(service='worker', level='INFO', file=stream)
        log = get_logger('worker')

        j_id = uuid4()
        bind_context(job_id=j_id)

        log.info('job_claimed', status='running')

        record = json.loads(stream.getvalue().strip())
        assert record['service'] == 'worker'
        assert record['job_id'] == str(j_id)
        assert record['status'] == 'running'

    def test_memory_content_redacted_at_info_level(self) -> None:
        """Verify memory content fields are stripped and lengths recorded at INFO."""
        stream = io.StringIO()
        configure_logging(service='api', level='INFO', log_content=False, file=stream)
        log = get_logger('test')

        secret = 'Secret personal memory text: Alice moved to Pune'
        log.info('fact_extracted', text=secret, fact_id='f-100')

        raw = stream.getvalue().strip()
        record = json.loads(raw)

        # Raw secret text must NOT appear anywhere in the line
        assert secret not in raw
        assert 'text' not in record
        assert record['text_length'] == len(secret)
        assert record['fact_id'] == 'f-100'

    def test_memory_content_preserved_at_debug_when_allowed(self) -> None:
        """Verify memory content is kept at DEBUG only when log_content=True."""
        stream = io.StringIO()
        configure_logging(service='api', level='DEBUG', log_content=True, file=stream)
        log = get_logger('test')

        msg = 'Debug fact inspection'
        log.debug('evaluating_candidate', content=msg)

        raw = stream.getvalue().strip()
        record = json.loads(raw)

        assert record['content'] == msg

    def test_memory_content_redacted_at_debug_when_disallowed(self) -> None:
        """Verify memory content is still redacted at DEBUG if log_content=False."""
        stream = io.StringIO()
        configure_logging(service='api', level='DEBUG', log_content=False, file=stream)
        log = get_logger('test')

        msg = 'Sensitive debug note'
        log.debug('evaluating_candidate', content=msg)

        raw = stream.getvalue().strip()
        record = json.loads(raw)

        assert 'content' not in record
        assert record['content_length'] == len(msg)

    def test_redact_content_helper(self) -> None:
        """Verify standalone redact_content function."""
        d = {'text': 'hello world', 'fact_id': '123'}
        redacted = redact_content(dict(d), allow_content=False)
        assert 'text' not in redacted
        assert redacted['text_length'] == 11
        assert redacted['fact_id'] == '123'

        preserved = redact_content(dict(d), allow_content=True)
        assert preserved['text'] == 'hello world'

    def test_error_logging_includes_code_and_class(self) -> None:
        """Verify error logging extracts error_code and exception_class."""
        stream = io.StringIO()
        configure_logging(service='api', level='INFO', file=stream)
        log = get_logger('test')

        err = ValidationFailed(message='Invalid turn data', code='E-VAL-01')
        log.error('request_failed', exc_info=err)

        record = json.loads(stream.getvalue().strip())
        assert record['exception_class'] == 'ValidationFailed'
        assert record['error_code'] == 'E-VAL-01'
