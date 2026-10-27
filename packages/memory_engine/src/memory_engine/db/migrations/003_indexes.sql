create index facts_hnsw    on facts    using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index sources_hnsw  on sources  using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index entities_hnsw on entities using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index facts_tsv     on facts    using gin (tsv);
create index sources_tsv   on sources  using gin (tsv);
create unique index entities_one_self on entities (user_id) where is_self;
create unique index facts_single_open on facts (user_id, subject_id, predicate)
  where cardinality = 'single' and valid_to is null and status = 'active';
create unique index facts_event_hash on facts (user_id, event_hash)
  where event_hash is not null and status <> 'deleted';
create index facts_timeline on facts (user_id, subject_id, predicate, valid_from);
create index facts_current  on facts (user_id, predicate)
  where valid_to is null and invalidated_at is null and status = 'active';
create index facts_validity on facts (user_id, valid_from, valid_to) where status in ('active','archived');
create index facts_object_entity on facts (user_id, object_entity_id) where object_entity_id is not null;
create index facts_review on facts (user_id) where status = 'needs_review';
create index fact_sources_source on fact_sources (source_id);
create index sources_session on sources (user_id, session_id, turn_index);
create unique index jobs_coalesce_queued on jobs (coalesce_key) where status = 'queued' and coalesce_key is not null;
create index jobs_poll  on jobs (run_after, created_at) where status = 'queued';
create index jobs_lease on jobs (lease_expires_at) where status = 'running';
create index jobs_user  on jobs (user_id, status);
create index messages_conv on messages (conversation_id, created_at);
