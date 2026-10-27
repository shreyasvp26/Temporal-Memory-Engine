"""Unit tests for SQL query catalog and row-to-domain mappers.

Verifies:
- All named query constants Q-01 through Q-18 exist and contain expected SQL.
- CR-02 coalesce_key sealing at 12 sources in Q-04.
- CR-03 per-hop CROSS JOIN LATERAL limit 15 cap in Q-11.
- Proper parameter splicing ($4, $5, etc.) for TemporalFilter in Q-07, Q-08, Q-09, Q-11.
- Accurate mapping of records/dictionaries into frozen domain entities and DTOs.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from memory_engine.db.mappers import (
    map_entity,
    map_evidence_snippet,
    map_fact,
    map_fact_evidence,
    map_fact_relation,
    map_job,
    map_message,
    map_predicate,
    map_scored_fact,
    map_scored_source,
    map_source,
)
from memory_engine.db.sql import (
    INSERT_AUDIT_LOG_SQL,
    INSERT_ENTITY_SQL,
    INSERT_FACT_EVIDENCE_SQL,
    INSERT_FACT_RELATION_SQL,
    INSERT_FACT_SQL,
    INSERT_MESSAGE_SQL,
    INSERT_SOURCE_SQL,
    Q01_CLAIM_JOBS,
    Q02_HEARTBEAT_JOB,
    Q02_REAP_EXPIRED_LEASES,
    Q03_COMPLETE_JOB,
    Q03_FAIL_JOB,
    Q03_HEARTBEAT,
    Q04_UPSERT_COALESCED_EXTRACT_JOB,
    Q05_LOAD_TIMELINE,
    Q05_LOCK_USER,
    Q06_ACTIVE_CANDIDATES,
    Q06_CLOSE_FACT,
    Q07_VECTOR_SEARCH_FACTS,
    Q08_FTS_SEARCH_FACTS,
    Q08_KEYWORD_SEARCH_FACTS,
    Q09_STRUCTURED_LOOKUP,
    Q09_TEMPORAL_RANGE_FACTS,
    Q10_VECTOR_SEARCH_SOURCES,
    Q11_GRAPH_EXPANSION,
    Q12_RECENT_MESSAGES,
    Q12_SUPERSEDES_CHAIN,
    Q13_EVIDENCE_SNIPPETS,
    Q13_UNPROCESSED_USER_SOURCES,
    Q14_ACCESS_RECORDING,
    Q14_EVIDENCE_FOR_FACTS,
    Q15_FACTS_ONLY_SUPPORTED,
    Q15_SUPERSESSION_NEIGHBORS,
    Q16_ENTITY_NEAREST,
    Q16_PURGE_USER,
    Q17_PENDING_USER_MESSAGES,
    Q17_REAP_EXPIRED_LEASES,
    Q18_HARD_DELETE_TOMBSTONES,
    inject_temporal_filter,
    inject_temporal_fragment,
)
from memory_engine.domain.enums import (
    Cardinality,
    DecayClass,
    EntityType,
    FactOrigin,
    FactStatus,
    JobStatus,
    JobType,
    ObjectKind,
    RelationType,
    Role,
    SourceKind,
)
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.intervals import TimeInterval
from memory_engine.domain.models import (
    Entity,
    Fact,
    FactEvidence,
    FactRelation,
    Job,
    Message,
    Predicate,
    Source,
)
from memory_engine.ports.repositories import (
    EvidenceSnippet,
    FilterMode,
    ScoredFact,
    ScoredSource,
    TemporalFilter,
)


def test_sql_constants_presence_and_types() -> None:
    constants = [
        Q01_CLAIM_JOBS,
        Q02_HEARTBEAT_JOB,
        Q03_COMPLETE_JOB,
        Q03_FAIL_JOB,
        Q04_UPSERT_COALESCED_EXTRACT_JOB,
        Q05_LOCK_USER,
        Q06_ACTIVE_CANDIDATES,
        Q07_VECTOR_SEARCH_FACTS,
        Q08_FTS_SEARCH_FACTS,
        Q09_TEMPORAL_RANGE_FACTS,
        Q10_VECTOR_SEARCH_SOURCES,
        Q11_GRAPH_EXPANSION,
        Q12_RECENT_MESSAGES,
        Q13_UNPROCESSED_USER_SOURCES,
        Q14_EVIDENCE_FOR_FACTS,
        Q15_SUPERSESSION_NEIGHBORS,
        Q16_PURGE_USER,
        Q17_REAP_EXPIRED_LEASES,
        Q18_HARD_DELETE_TOMBSTONES,
        # Section 6.8 aliases
        Q02_REAP_EXPIRED_LEASES,
        Q03_HEARTBEAT,
        Q05_LOAD_TIMELINE,
        Q06_CLOSE_FACT,
        Q08_KEYWORD_SEARCH_FACTS,
        Q09_STRUCTURED_LOOKUP,
        Q12_SUPERSEDES_CHAIN,
        Q13_EVIDENCE_SNIPPETS,
        Q14_ACCESS_RECORDING,
        Q15_FACTS_ONLY_SUPPORTED,
        Q16_ENTITY_NEAREST,
        Q17_PENDING_USER_MESSAGES,
        # Helpers
        INSERT_SOURCE_SQL,
        INSERT_ENTITY_SQL,
        INSERT_FACT_SQL,
        INSERT_FACT_EVIDENCE_SQL,
        INSERT_FACT_RELATION_SQL,
        INSERT_MESSAGE_SQL,
        INSERT_AUDIT_LOG_SQL,
    ]
    for c in constants:
        assert isinstance(c, str)
        assert len(c.strip()) > 0


def test_cr02_coalesce_key_sealing_in_q04() -> None:
    """CR-02: Q-04 must seal coalesce_key (setting to null) at 12 sources."""
    sql = Q04_UPSERT_COALESCED_EXTRACT_JOB
    assert "on conflict (coalesce_key) where status = 'queued' and coalesce_key is not null" in sql
    assert "jsonb_array_length(jobs.payload->'source_ids') + 1 >= 12 then null" in sql
    assert "jsonb_array_length(jobs.payload->'source_ids') + 1 >= 12 then now()" in sql
    assert 'greatest(jobs.run_after, now() + make_interval(secs => $6))' in sql
    assert 'returning id, (xmax = 0) as inserted;' in sql


def test_cr03_graph_expansion_cap_in_q11() -> None:
    """CR-03: Q-11 must cap per-hop expansion using CROSS JOIN LATERAL (... LIMIT 15)."""
    sql = Q11_GRAPH_EXPANSION
    assert 'cross join lateral (' in sql
    assert 'limit 15' in sql
    assert 'not e_curr.is_self' in sql
    assert 'not e.is_self' in sql
    assert '{temporal}' in sql


def test_q01_claim_jobs_structure() -> None:
    sql = Q01_CLAIM_JOBS
    assert "status = 'running'" in sql
    assert 'for update skip locked limit $2' in sql
    assert 'lease_expires_at = now() + make_interval(secs => $3)' in sql
    assert 'returning *;' in sql


def test_q05_lock_user_structure() -> None:
    sql = Q05_LOCK_USER
    assert 'pg_advisory_xact_lock' in sql
    assert "hashtext('sb_fact:' || $1::text)" in sql


def test_temporal_filter_splicing_q09_current() -> None:
    flt = TemporalFilter(mode=FilterMode.CURRENT)
    built_sql, params = inject_temporal_filter(Q09_TEMPORAL_RANGE_FACTS, flt, param_offset=4)
    assert '{temporal}' not in built_sql
    assert "f.status = 'active' and f.valid_to is null and f.invalidated_at is null" in built_sql
    assert 'order by f.valid_from desc limit $4;' in built_sql
    assert params == ()


def test_temporal_filter_splicing_q09_overlap_offsets() -> None:
    t_start = datetime(2026, 1, 1, tzinfo=UTC)
    t_end = datetime(2026, 6, 1, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=t_start, end=t_end))

    built_sql, params = inject_temporal_filter(Q09_TEMPORAL_RANGE_FACTS, flt, param_offset=4)
    assert '{temporal}' not in built_sql
    # OVERLAP with start and end produces valid_from < $4 and (valid_to is null or valid_to > $5)
    assert 'f.valid_from < $4' in built_sql
    assert 'f.valid_to > $5' in built_sql
    # The suffix limit parameter must be shifted from $4 to $6!
    assert 'limit $6;' in built_sql
    assert params == (t_end, t_start)


def test_temporal_filter_splicing_q11_overlap() -> None:
    t_start = datetime(2026, 2, 1, tzinfo=UTC)
    t_end = datetime(2026, 5, 1, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.OVERLAP, range=TimeInterval(start=t_start, end=t_end))

    built_sql, params = inject_temporal_filter(Q11_GRAPH_EXPANSION, flt, param_offset=4)
    assert '{temporal}' not in built_sql
    assert 'f.valid_from < $4' in built_sql
    assert 'f.valid_to > $5' in built_sql
    assert 'limit $6;' in built_sql
    assert params == (t_end, t_start)


def test_temporal_filter_splicing_q07_and_q08() -> None:
    t_as_of = datetime(2026, 4, 1, tzinfo=UTC)
    flt = TemporalFilter(mode=FilterMode.AS_OF, as_of=t_as_of)

    # Q07 has prefix with $1, $2, so param_offset defaults to 3
    built_sql, params = inject_temporal_filter(Q07_VECTOR_SEARCH_FACTS, flt)
    assert '{temporal}' not in built_sql
    assert 'f.valid_from <= $3' in built_sql
    # suffix had limit $3 -> shifted to limit $4
    assert 'limit $4;' in built_sql
    assert params == (t_as_of,)

    # Q08 with explicit param_offset and inject_temporal_fragment helper
    frag = flt.to_sql(alias='f', first_param=3)
    frag_sql = inject_temporal_fragment(Q08_FTS_SEARCH_FACTS, frag, param_offset=3)
    assert '{temporal}' not in frag_sql
    assert 'f.valid_from <= $3' in frag_sql
    assert 'limit $4;' in frag_sql


def test_inject_temporal_filter_missing_placeholder_raises() -> None:
    with pytest.raises(ValueError, match="does not contain '{temporal}'"):
        inject_temporal_filter('select * from facts', TemporalFilter(mode=FilterMode.CURRENT))


def test_map_source() -> None:
    sid = uuid4()
    uid = uuid4()
    now_utc = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    row = {
        'id': str(sid),
        'user_id': uid,
        'kind': 'chat',
        'role': 'user',
        'session_id': 'sess-123',
        'turn_index': 3,
        'content': 'I love working with PostgreSQL and Python.',
        'source_ts': now_utc.isoformat(),
        'content_hash': 'hash123',
        'deleted_at': None,
    }
    src = map_source(row)
    assert isinstance(src, Source)
    assert src.id == sid
    assert src.user_id == uid
    assert src.kind == SourceKind.CHAT
    assert src.role == Role.USER
    assert src.session_id == 'sess-123'
    assert src.turn_index == 3
    assert src.source_ts == now_utc
    assert src.deleted_at is None


def test_map_entity() -> None:
    eid = uuid4()
    uid = uuid4()
    row = {
        'id': eid,
        'user_id': str(uid),
        'canonical_name': 'Alice Smith',
        'type': 'person',
        'is_self': False,
    }
    entity = map_entity(row)
    assert isinstance(entity, Entity)
    assert entity.id == eid
    assert entity.user_id == uid
    assert entity.canonical_name == 'Alice Smith'
    assert entity.type == EntityType.PERSON
    assert entity.is_self is False


def test_map_predicate() -> None:
    row = {
        'name': 'lives_in',
        'cardinality': 'single',
        'decay_class': 'slow',
        'object_kind': 'entity',
        'template': '{subject} lives in {object}',
        'synonyms': ['resides in', 'stays in'],
    }
    pred = map_predicate(row)
    assert isinstance(pred, Predicate)
    assert pred.name == 'lives_in'
    assert pred.cardinality == Cardinality.SINGLE
    assert pred.decay_class == DecayClass.SLOW
    assert pred.object_kind == ObjectKind.ENTITY
    assert pred.template == '{subject} lives in {object}'
    assert pred.synonyms == ('resides in', 'stays in')


def test_map_fact_literal() -> None:
    fid = uuid4()
    uid = uuid4()
    sub_id = uuid4()
    t_from = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_rec = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    row = {
        'id': fid,
        'user_id': uid,
        'subject_id': sub_id,
        'predicate': 'likes_coffee',
        'cardinality': 'single',
        'object_entity_id': None,
        'object_value': 'dark roast',
        'object_norm': 'dark roast',
        'fact_text': 'Alice likes coffee dark roast',
        'valid_from': t_from,
        'valid_to': None,
        'time_precision': 'day',
        'valid_from_explicit': True,
        'recorded_at': t_rec,
        'invalidated_at': None,
        'confidence': 0.95,
        'salience': 1.0,
        'access_count': 3,
        'last_accessed_at': None,
        'status': 'active',
        'origin': 'extracted',
        'event_hash': None,
    }
    fact = map_fact(row)
    assert isinstance(fact, Fact)
    assert fact.id == fid
    assert fact.object_value == 'dark roast'
    assert fact.object_entity_id is None
    assert fact.confidence == 0.95
    assert fact.status == FactStatus.ACTIVE
    assert fact.origin == FactOrigin.EXTRACTED
    assert fact.is_current is True


def test_map_fact_entity() -> None:
    fid = uuid4()
    uid = uuid4()
    sub_id = uuid4()
    obj_id = uuid4()
    t_from = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    t_to = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    t_rec = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    t_inv = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    row = {
        'id': fid,
        'user_id': uid,
        'subject_id': sub_id,
        'predicate': 'employed_by',
        'cardinality': 'single',
        'object_entity_id': obj_id,
        'object_value': None,
        'object_norm': 'acme corp',
        'fact_text': 'Alice is employed by Acme Corp',
        'valid_from': t_from,
        'valid_to': t_to,
        'time_precision': 'month',
        'valid_from_explicit': False,
        'recorded_at': t_rec,
        'invalidated_at': t_inv,
        'confidence': 0.9,
        'salience': 0.8,
        'access_count': 5,
        'last_accessed_at': t_rec,
        'status': 'archived',
        'origin': 'extracted',
        'event_hash': 'evthash',
    }
    fact = map_fact(row)
    assert isinstance(fact, Fact)
    assert fact.object_entity_id == obj_id
    assert fact.object_value is None
    assert fact.valid_to == t_to
    assert fact.status == FactStatus.ARCHIVED
    assert fact.is_current is False


def test_map_job_with_json_and_dict_payload() -> None:
    jid = uuid4()
    uid = uuid4()
    wid = uuid4()
    now_utc = datetime(2026, 10, 27, 14, 0, tzinfo=UTC)

    # Test with string JSON payload
    payload_dict = {'session_id': 's1', 'source_ids': ['id1', 'id2']}
    row_json = {
        'id': str(jid),
        'user_id': str(uid),
        'type': 'extract',
        'payload': json.dumps(payload_dict),
        'idempotency_key': 'extract:id1',
        'coalesce_key': 'extract:user:sess',
        'status': 'running',
        'attempts': 1,
        'max_attempts': 3,
        'deferrals': 0,
        'run_after': now_utc,
        'worker_id': str(wid),
        'lease_expires_at': now_utc,
        'last_error': None,
    }
    job = map_job(row_json)
    assert isinstance(job, Job)
    assert job.id == jid
    assert job.user_id == uid
    assert job.type == JobType.EXTRACT
    assert job.payload == payload_dict
    assert job.status == JobStatus.RUNNING
    assert job.worker_id == wid

    # Test with dict payload directly
    row_dict = dict(row_json)
    row_dict['payload'] = payload_dict
    job2 = map_job(row_dict)
    assert job2.payload == payload_dict


def test_map_message() -> None:
    mid = uuid4()
    cid = uuid4()
    uid = uuid4()
    sid = uuid4()
    now_utc = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)
    row = {
        'id': mid,
        'conversation_id': cid,
        'user_id': uid,
        'role': 'assistant',
        'content': 'You live in Seattle.',
        'client_message_id': 'msg-1',
        'source_id': sid,
        'partial': False,
        'created_at': now_utc,
    }
    msg = map_message(row)
    assert isinstance(msg, Message)
    assert msg.id == mid
    assert msg.conversation_id == cid
    assert msg.role == Role.ASSISTANT
    assert msg.content == 'You live in Seattle.'
    assert msg.source_id == sid


def test_map_scored_fact_and_scored_source() -> None:
    fid = uuid4()
    uid = uuid4()
    sid = uuid4()
    t_from = datetime(2026, 1, 1, tzinfo=UTC)
    t_rec = datetime(2026, 1, 2, tzinfo=UTC)

    fact_row = {
        'id': fid,
        'user_id': uid,
        'subject_id': sid,
        'predicate': 'name',
        'cardinality': 'single',
        'object_entity_id': None,
        'object_value': 'Bob',
        'object_norm': 'bob',
        'fact_text': 'Bob name',
        'valid_from': t_from,
        'valid_to': None,
        'time_precision': 'day',
        'valid_from_explicit': False,
        'recorded_at': t_rec,
        'invalidated_at': None,
        'confidence': 0.8,
        'salience': 1.0,
        'access_count': 0,
        'last_accessed_at': None,
        'status': 'active',
        'origin': 'extracted',
        'event_hash': None,
        'score': 0.887,
    }
    sf = map_scored_fact(fact_row)
    assert isinstance(sf, ScoredFact)
    assert sf.score == pytest.approx(0.887)
    assert sf.fact.id == fid

    source_row = {
        'id': sid,
        'user_id': uid,
        'kind': 'note',
        'role': 'user',
        'session_id': 'sess-note',
        'turn_index': 0,
        'content': 'Note content',
        'source_ts': t_rec,
        'content_hash': 'h1',
        'score': 0.92,
    }
    ss = map_scored_source(source_row)
    assert isinstance(ss, ScoredSource)
    assert ss.score == pytest.approx(0.92)
    assert ss.source.id == sid


def test_map_evidence_snippet() -> None:
    sid = uuid4()
    fid = uuid4()
    t = datetime(2026, 10, 27, 8, 30, tzinfo=UTC)
    row = {
        'source_id': sid,
        'source_ts': t,
        'role': 'user',
        'snippet': '...moved to Zurich last month...',
        'fact_id': fid,
    }
    ev = map_evidence_snippet(row)
    assert isinstance(ev, EvidenceSnippet)
    assert ev.source_id == sid
    assert ev.source_ts == t
    assert ev.role == Role.USER
    assert ev.snippet == '...moved to Zurich last month...'
    assert ev.fact_id == fid


def test_map_fact_evidence_and_relation() -> None:
    fid1 = uuid4()
    fid2 = uuid4()
    sid = uuid4()

    ev_row = {
        'fact_id': fid1,
        'source_id': sid,
        'span_start': 10,
        'span_end': 45,
    }
    fe = map_fact_evidence(ev_row)
    assert isinstance(fe, FactEvidence)
    assert fe.fact_id == fid1
    assert fe.source_id == sid
    assert fe.span_start == 10
    assert fe.span_end == 45

    rel_row = {
        'from_fact_id': fid1,
        'to_fact_id': fid2,
        'relation': 'supersedes',
    }
    fr = map_fact_relation(rel_row)
    assert isinstance(fr, FactRelation)
    assert fr.from_fact_id == fid1
    assert fr.to_fact_id == fid2
    assert fr.relation == RelationType.SUPERSEDES


def test_mapper_error_handling() -> None:
    # Invalid UUID raises InvariantViolation
    with pytest.raises(InvariantViolation, match='must be a valid UUID'):
        map_source({'id': 'not-a-uuid'})

    # Missing mandatory datetime raises InvariantViolation
    with pytest.raises(InvariantViolation, match='source_ts cannot be None'):
        map_source(
            {
                'id': uuid4(),
                'user_id': uuid4(),
                'kind': 'chat',
                'role': 'user',
                'session_id': 's',
                'turn_index': 0,
                'content': 'c',
                'source_ts': None,
                'content_hash': 'h',
            }
        )

    # Non-dict JSON in job payload raises InvariantViolation
    with pytest.raises(InvariantViolation, match='must be a JSON object'):
        map_job(
            {
                'id': uuid4(),
                'user_id': uuid4(),
                'type': 'extract',
                'payload': json.dumps(['not', 'an', 'object']),
                'idempotency_key': 'k',
                'coalesce_key': None,
                'status': 'queued',
                'attempts': 0,
                'max_attempts': 3,
                'deferrals': 0,
                'run_after': datetime.now(UTC),
                'worker_id': None,
                'lease_expires_at': None,
                'last_error': None,
            }
        )
