"""SQL query catalog and query builder helpers for PostgreSQL.

Implements queries Q-01 through Q-18 matching docs/LLD.md Section 6.8 verbatim,
including CR-01, CR-02 (coalesce_key sealing at 12 sources), and CR-03 (degree cap
of 15 per hop in graph expansion).
"""

import re
from typing import Any

from memory_engine.ports.repositories import SqlFragment, TemporalFilter

# -----------------------------------------------------------------------------
# Q-01 to Q-18 SQL catalog statements (Section 6.8 verbatim)
# -----------------------------------------------------------------------------

# Q-01: Claim jobs (worker, no user context)
Q01_CLAIM_JOBS: str = """\
update jobs set status = 'running', worker_id = $1, locked_at = now(),
       lease_expires_at = now() + make_interval(secs => $3), attempts = attempts + 1
where id in (select id from jobs where status = 'queued' and run_after <= now()
             order by run_after, created_at for update skip locked limit $2)
returning *;"""

# Q-02 / Q-17: Heartbeat / Reap expired leases
Q02_HEARTBEAT_JOB: str = """\
update jobs set lease_expires_at = now() + make_interval(secs => $3)
where id = $1 and worker_id = $2 and status = 'running';"""

# Q-03: Complete job
Q03_COMPLETE_JOB: str = """\
update jobs set status = 'done', finished_at = $3, worker_id = null, lease_expires_at = null
where id = $1 and worker_id = $2 and status = 'running';"""

# Q-03: Fail / retry job
Q03_FAIL_JOB: str = """\
update jobs set status = case when attempts >= max_attempts then 'dead' else 'queued' end,
       worker_id = null, lease_expires_at = null, run_after = $3,
       last_error = $4, error_code = $5,
       finished_at = case when attempts >= max_attempts then now() end
where id = $1 and worker_id = $2 and status = 'running';"""

# Q-04: Upsert coalesced extract job (seals at 12 sources per CR-02)
Q04_UPSERT_COALESCED_EXTRACT_JOB: str = """\
insert into jobs (user_id, type, payload, idempotency_key, coalesce_key, run_after)
values ($1, 'extract', jsonb_build_object(
          'session_id', $2::text,
          'source_ids', jsonb_build_array($3::text)
        ),
        $4, $5, now() + make_interval(secs => $6))
on conflict (coalesce_key) where status = 'queued' and coalesce_key is not null
do update set
  payload = jsonb_set(
    jobs.payload, '{source_ids}',
    jobs.payload->'source_ids' || jsonb_build_array($3::text)
  ),
  coalesce_key = case
    when jsonb_array_length(jobs.payload->'source_ids') + 1 >= 12 then null
    else jobs.coalesce_key
  end,
  run_after = case
    when jsonb_array_length(jobs.payload->'source_ids') + 1 >= 12 then now()
    else greatest(jobs.run_after, now() + make_interval(secs => $6))
  end
returning id, (xmax = 0) as inserted;"""

# Q-05: Lock user (transaction-level advisory lock)
Q05_LOCK_USER: str = """\
select pg_advisory_xact_lock(hashtext('sb_fact:' || $1::text));"""

# Q-06: Active candidates / load timeline (inside UoW, after lock_user)
Q06_ACTIVE_CANDIDATES: str = """\
select * from facts where user_id = $1 and subject_id = $2 and predicate = $3
  and status in ('active','needs_review') order by valid_from, recorded_at for update;"""

# Q-06 (alternate): Close fact
Q06_CLOSE_FACT: str = """\
update facts set valid_to = $3, invalidated_at = $4
where id = $2 and user_id = $1 and (valid_to is null or valid_to > $3);"""

# Q-07: Vector search over facts
Q07_VECTOR_SEARCH_FACTS: str = """\
select f.*, 1 - (f.embedding <=> $2) as score from facts f
where f.user_id = $1 and f.embedding is not null and {temporal}
order by f.embedding <=> $2 limit $3;"""

# Q-08: Keyword search over facts (FTS)
Q08_FTS_SEARCH_FACTS: str = """\
select f.*, ts_rank_cd(f.tsv, q) as score from facts f, websearch_to_tsquery('english', $2) q
where f.user_id = $1 and f.tsv @@ q and {temporal} order by score desc limit $3;"""

# Q-09: Structured lookup / Temporal range facts
Q09_TEMPORAL_RANGE_FACTS: str = """\
select f.*, 1.0 as score from facts f
where f.user_id = $1
  and ($2::uuid[] is null or f.subject_id = any($2) or f.object_entity_id = any($2))
  and ($3::text[] is null or f.predicate = any($3)) and {temporal}
order by f.valid_from desc limit $4;"""

# Q-10: Vector search over sources
Q10_VECTOR_SEARCH_SOURCES: str = """\
select s.*, 1 - (s.embedding <=> $2) as score from sources s
where s.user_id = $1 and s.embedding is not null and s.deleted_at is null
  and ($4::timestamptz is null or s.source_ts <= $4)
order by s.embedding <=> $2 limit $3;"""

# Q-11: Graph expansion (CR-03 per-hop CROSS JOIN LATERAL limit 15 cap)
Q11_GRAPH_EXPANSION: str = """\
with recursive walk(entity_id, depth) as (
  select unnest($2::uuid[]), 0
  union
  select hop.next_entity_id, w.depth + 1
  from walk w
  join entities e_curr on e_curr.id = w.entity_id and not e_curr.is_self
  cross join lateral (
    select case
      when f.subject_id = w.entity_id then f.object_entity_id
      else f.subject_id
    end as next_entity_id
    from facts f
    where f.user_id = $1 and (f.subject_id = w.entity_id or f.object_entity_id = w.entity_id)
      and f.object_entity_id is not null and f.status = 'active'
    order by f.valid_from desc
    limit 15
  ) hop
  where w.depth < $3
)
select f.*, 1.0 / (1 + w.depth) as score from (
  select entity_id, min(depth) as depth from walk group by entity_id
) w
join entities e on e.id = w.entity_id and not e.is_self
join facts f on f.user_id = $1 and (f.subject_id = w.entity_id or f.object_entity_id = w.entity_id)
where {temporal} order by score desc, f.valid_from desc limit $4;"""

# Q-12: Recent messages
Q12_RECENT_MESSAGES: str = """\
select * from messages
where user_id = $1 and conversation_id = $2
order by created_at desc limit $3;"""

# Q-13 / Q-17: Pending user messages / unprocessed user sources (read-your-writes buffer)
Q13_UNPROCESSED_USER_SOURCES: str = """\
select m.* from messages m
join jobs j on j.user_id = m.user_id and j.type = 'extract'
  and j.status in ('queued','running')
  and j.payload->'source_ids' ? m.source_id::text
where m.user_id = $1 and m.role = 'user' order by m.created_at desc limit $2;"""

# Q-14 / Q-13 (LLD catalog): Evidence snippets for facts
Q14_EVIDENCE_FOR_FACTS: str = """\
select distinct on (fs.fact_id) fs.fact_id, s.id as source_id, s.source_ts, s.role,
       substring(s.content from greatest(coalesce(fs.span_start, 1) - 80, 1) for 320) as snippet
from fact_sources fs join sources s on s.id = fs.source_id
where fs.user_id = $1 and fs.fact_id = any($2) and s.deleted_at is null
order by fs.fact_id, s.source_ts desc;"""

# Q-15: Supersession neighbors / 1-hop chain
Q15_SUPERSESSION_NEIGHBORS: str = """\
select distinct f.* from facts f
join fact_relations fr on (fr.from_fact_id = f.id or fr.to_fact_id = f.id)
where f.user_id = $1 and fr.relation = 'supersedes'
  and (fr.from_fact_id = any($2) or fr.to_fact_id = any($2))
  and f.id <> all($2);"""

# Q-16: Purge user
Q16_PURGE_USER: str = """\
delete from auth.users where id = $1;"""

# Q-17: Reap expired leases (Q-02 in Section 6.8)
Q17_REAP_EXPIRED_LEASES: str = """\
update jobs set status = case when attempts >= max_attempts then 'dead' else 'queued' end,
       worker_id = null, lease_expires_at = null, run_after = now(),
       last_error = coalesce(last_error, 'lease expired'),
       error_code = coalesce(error_code, 'E-JOB-LEASE'),
       finished_at = case when attempts >= max_attempts then now() end
where status = 'running' and lease_expires_at < now();"""

# Q-18: Hard delete tombstones
Q18_HARD_DELETE_TOMBSTONES: str = """\
delete from facts where status = 'deleted' and invalidated_at < $1;"""

# -----------------------------------------------------------------------------
# Section 6.8 literal Q-01 to Q-17 catalog aliases
# -----------------------------------------------------------------------------
Q02_REAP_EXPIRED_LEASES = Q17_REAP_EXPIRED_LEASES
Q03_HEARTBEAT = Q02_HEARTBEAT_JOB
Q05_LOAD_TIMELINE = Q06_ACTIVE_CANDIDATES
Q08_KEYWORD_SEARCH_FACTS = Q08_FTS_SEARCH_FACTS
Q09_STRUCTURED_LOOKUP = Q09_TEMPORAL_RANGE_FACTS
Q12_SUPERSEDES_CHAIN: str = """\
with recursive chain as (
  select f.*, 0 as depth
  from facts f
  where f.user_id = $1 and f.id = $2
  union
  select f.*, c.depth + 1
  from chain c
  join fact_relations fr on (
    (fr.from_fact_id = c.id and fr.relation = 'supersedes') or
    (fr.to_fact_id = c.id and fr.relation = 'supersedes')
  )
  join facts f on f.user_id = $1 and f.id = case
    when fr.from_fact_id = c.id then fr.to_fact_id
    else fr.from_fact_id
  end
  where c.depth < 20
)
select * from chain order by valid_from;"""
Q13_EVIDENCE_SNIPPETS = Q14_EVIDENCE_FOR_FACTS
Q14_ACCESS_RECORDING: str = """\
update facts set access_count = access_count + 1, last_accessed_at = $3
where user_id = $1 and id = any($2);"""
Q15_FACTS_ONLY_SUPPORTED: str = """\
select f.id from facts f
where f.user_id = $1
  and f.id in (select fact_id from fact_sources where source_id = any($2))
  and not exists (
    select 1 from fact_sources fs2 where fs2.fact_id = f.id and fs2.source_id <> all($2)
  );"""
Q16_ENTITY_NEAREST: str = """\
select e.*, 1 - (e.embedding <=> $3) as sim from entities e
where e.user_id = $1 and e.type = $2 and not e.is_self order by e.embedding <=> $3 limit $4;"""
Q17_PENDING_USER_MESSAGES = Q13_UNPROCESSED_USER_SOURCES

# -----------------------------------------------------------------------------
# Insert and update SQL helpers
# -----------------------------------------------------------------------------
INSERT_SOURCE_SQL: str = """\
insert into sources (
  id, user_id, kind, role, session_id, turn_index, content, source_ts, content_hash
) values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
on conflict (user_id, content_hash) do nothing
returning id;"""

INSERT_ENTITY_SQL: str = """\
insert into entities (id, user_id, canonical_name, type, is_self, embedding)
values ($1, $2, $3, $4, $5, $6)
on conflict (user_id, canonical_name, type) do nothing
returning id;"""

INSERT_ENTITY_ALIAS_SQL: str = """\
insert into entity_aliases (user_id, type, alias_norm, alias, entity_id)
values ($1, $2, $3, $4, $5)
on conflict (user_id, type, alias_norm) do nothing;"""

INSERT_FACT_SQL: str = """\
insert into facts (
  id, user_id, subject_id, predicate, cardinality, object_entity_id, object_value, object_norm,
  fact_text, embedding, embedding_model, valid_from, valid_to, time_precision, valid_from_explicit,
  recorded_at, confidence, salience, status, origin, event_hash
) values (
  $1, $2, $3, $4, $5, $6, $7, $8,
  $9, $10, $11, $12, $13, $14, $15,
  $16, $17, $18, $19, $20, $21
) returning id;"""

INSERT_FACT_EVIDENCE_SQL: str = """\
insert into fact_sources (fact_id, source_id, user_id, span_start, span_end)
values ($1, $2, $3, $4, $5)
on conflict (fact_id, source_id) do nothing;"""

INSERT_FACT_RELATION_SQL: str = """\
insert into fact_relations (user_id, from_fact_id, to_fact_id, relation, job_id)
values ($1, $2, $3, $4, $5)
on conflict (from_fact_id, to_fact_id, relation) do nothing;"""

INSERT_CONVERSATION_SQL: str = """\
insert into conversations (id, user_id, title, created_at, updated_at)
values ($1, $2, $3, $4, $4)
on conflict (id) do nothing
returning id;"""

INSERT_MESSAGE_SQL: str = """\
insert into messages (
  id, conversation_id, user_id, role, content, client_message_id, source_id, partial, trace,
  created_at
) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
on conflict (conversation_id, client_message_id) where client_message_id is not null do nothing
returning id;"""

INSERT_AUDIT_LOG_SQL: str = """\
insert into audit_log (user_id, actor, action, target_type, target_id, before, after)
values ($1, $2, $3, $4, $5, $6, $7);"""

UPDATE_FACT_STATUS_SQL: str = """\
update facts set status = $3, invalidated_at = $4 where user_id = $1 and id = $2;"""

UPDATE_FACT_VALID_FROM_SQL: str = """\
update facts set valid_from = $3 where user_id = $1 and id = $2;"""

UPDATE_FACT_SALIENCE_SQL: str = """\
update facts set salience = $3 where user_id = $1 and id = $2;"""

UPDATE_FACT_CONFIDENCE_SQL: str = """\
update facts set confidence = $3 where user_id = $1 and id = $2;"""

SOFT_DELETE_SOURCE_SQL: str = """\
update sources set deleted_at = $3 where user_id = $1 and id = any($2) and deleted_at is null;"""

SOFT_DELETE_FACT_SQL: str = """\
update facts set status = 'deleted', invalidated_at = $3
where user_id = $1 and id = any($2) and status <> 'deleted';"""

SET_SOURCE_EMBEDDING_SQL: str = """\
update sources set embedding = $3, embedding_model = $4 where user_id = $1 and id = $2;"""

SET_FACT_EMBEDDING_SQL: str = """\
update facts set embedding = $3, embedding_model = $4 where user_id = $1 and id = $2;"""

# -----------------------------------------------------------------------------
# Query builder helpers for TemporalFilter injection
# -----------------------------------------------------------------------------


def inject_temporal_filter(
    sql_template: str,
    temporal: TemporalFilter | SqlFragment,
    param_offset: int | None = None,
    alias: str = 'f',
) -> tuple[str, tuple[Any, ...]]:
    """Inject a TemporalFilter or SqlFragment safely into a query containing {temporal}.

    Slices parameter placeholders occurring in the suffix after {temporal} so that
    bound parameters ($4, $5, etc.) match their positional offsets cleanly.
    """
    if isinstance(param_offset, str) and isinstance(alias, int):
        alias, param_offset = param_offset, alias

    if '{temporal}' not in sql_template:
        raise ValueError("sql_template does not contain '{temporal}' placeholder")

    prefix, _, suffix = sql_template.partition('{temporal}')

    prefix_param_matches = [int(m.group(1)) for m in re.finditer(r'\$([0-9]+)', prefix)]
    inferred_offset = (max(prefix_param_matches) + 1) if prefix_param_matches else 1
    effective_offset = param_offset if param_offset is not None else inferred_offset

    if isinstance(temporal, TemporalFilter):
        frag = temporal.to_sql(alias=alias, first_param=effective_offset)
    elif isinstance(temporal, SqlFragment):
        frag = temporal
    else:
        raise TypeError(f'Expected TemporalFilter or SqlFragment, got {type(temporal)!r}')

    num_added_params = len(frag.params)

    def _shift_suffix_param(match: re.Match[str]) -> str:
        param_num = int(match.group(1))
        if param_num >= effective_offset:
            return f'${param_num + num_added_params}'
        return match.group(0)

    shifted_suffix = re.sub(r'\$([0-9]+)', _shift_suffix_param, suffix)
    final_sql = f'{prefix}{frag.sql}{shifted_suffix}'
    return final_sql, frag.params


def inject_temporal_fragment(
    sql_template: str,
    fragment: SqlFragment,
    param_offset: int | None = None,
) -> str:
    """Helper to inject a SqlFragment directly into {temporal} and return the SQL string."""
    sql, _ = inject_temporal_filter(sql_template, fragment, param_offset=param_offset)
    return sql


__all__ = [
    'INSERT_AUDIT_LOG_SQL',
    'INSERT_CONVERSATION_SQL',
    'INSERT_ENTITY_ALIAS_SQL',
    'INSERT_ENTITY_SQL',
    'INSERT_FACT_EVIDENCE_SQL',
    'INSERT_FACT_RELATION_SQL',
    'INSERT_FACT_SQL',
    'INSERT_MESSAGE_SQL',
    'INSERT_SOURCE_SQL',
    'Q01_CLAIM_JOBS',
    'Q02_HEARTBEAT_JOB',
    'Q02_REAP_EXPIRED_LEASES',
    'Q03_COMPLETE_JOB',
    'Q03_FAIL_JOB',
    'Q03_HEARTBEAT',
    'Q04_UPSERT_COALESCED_EXTRACT_JOB',
    'Q05_LOAD_TIMELINE',
    'Q05_LOCK_USER',
    'Q06_ACTIVE_CANDIDATES',
    'Q06_CLOSE_FACT',
    'Q07_VECTOR_SEARCH_FACTS',
    'Q08_FTS_SEARCH_FACTS',
    'Q08_KEYWORD_SEARCH_FACTS',
    'Q09_STRUCTURED_LOOKUP',
    'Q09_TEMPORAL_RANGE_FACTS',
    'Q10_VECTOR_SEARCH_SOURCES',
    'Q11_GRAPH_EXPANSION',
    'Q12_RECENT_MESSAGES',
    'Q12_SUPERSEDES_CHAIN',
    'Q13_EVIDENCE_SNIPPETS',
    'Q13_UNPROCESSED_USER_SOURCES',
    'Q14_ACCESS_RECORDING',
    'Q14_EVIDENCE_FOR_FACTS',
    'Q15_FACTS_ONLY_SUPPORTED',
    'Q15_SUPERSESSION_NEIGHBORS',
    'Q16_ENTITY_NEAREST',
    'Q16_PURGE_USER',
    'Q17_PENDING_USER_MESSAGES',
    'Q17_REAP_EXPIRED_LEASES',
    'Q18_HARD_DELETE_TOMBSTONES',
    'SET_FACT_EMBEDDING_SQL',
    'SET_SOURCE_EMBEDDING_SQL',
    'SOFT_DELETE_FACT_SQL',
    'SOFT_DELETE_SOURCE_SQL',
    'UPDATE_FACT_CONFIDENCE_SQL',
    'UPDATE_FACT_SALIENCE_SQL',
    'UPDATE_FACT_STATUS_SQL',
    'UPDATE_FACT_VALID_FROM_SQL',
    'inject_temporal_filter',
    'inject_temporal_fragment',
]
