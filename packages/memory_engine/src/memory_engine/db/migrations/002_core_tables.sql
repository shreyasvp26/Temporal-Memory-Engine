create table profiles (
  user_id uuid primary key references auth.users on delete cascade,
  timezone text not null default 'UTC',
  created_at timestamptz not null default now()
);

create table predicates (
  name text primary key,
  cardinality text not null check (cardinality in ('single','multi','event')),
  decay_class text not null check (decay_class in ('permanent','slow','normal','fast')),
  object_kind text not null check (object_kind in ('entity','literal')),
  template text not null,
  synonyms text[] not null default '{}',
  description text not null default ''
);

create table sources (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  kind text not null check (kind in ('chat','note','doc')),
  role text not null default 'user' check (role in ('user','assistant','system')),
  session_id text not null,
  turn_index int not null default 0,
  content text not null check (length(content) between 1 and 20000),
  source_ts timestamptz not null,
  content_hash text not null,        -- sha256 hex of kind|session_id|dedupe_key|role|content (dedupe_key = client_message_id or chunk_index)
  embedding vector(768),
  embedding_model text,
  tsv tsvector generated always as (to_tsvector('english', content)) stored,
  deleted_at timestamptz,
  created_at timestamptz not null default now(),
  unique (user_id, content_hash)
);

create table entities (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  canonical_name text not null,
  type text not null check (type in ('person','place','org','thing','topic')),
  is_self boolean not null default false,
  embedding vector(768),
  created_at timestamptz not null default now(),
  unique (user_id, canonical_name, type)
);

create table entity_aliases (
  user_id uuid not null references auth.users on delete cascade,
  type text not null,
  alias_norm text not null,
  alias text not null,
  entity_id uuid not null references entities on delete cascade,
  primary key (user_id, type, alias_norm)
);

create table facts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  subject_id uuid not null references entities,
  predicate text not null references predicates,
  cardinality text not null check (cardinality in ('single','multi','event')),
  object_entity_id uuid references entities,
  object_value text,
  object_norm text not null,
  fact_text text not null,
  embedding vector(768),
  embedding_model text,
  tsv tsvector generated always as (to_tsvector('english', fact_text)) stored,
  valid_from timestamptz not null,
  valid_to timestamptz,
  time_precision text not null default 'day' check (time_precision in ('year','month','day','exact')),
  valid_from_explicit boolean not null default false,
  recorded_at timestamptz not null default now(),
  invalidated_at timestamptz,
  confidence real not null default 0.8 check (confidence between 0 and 1),
  salience real not null default 1.0,
  access_count int not null default 0,
  last_accessed_at timestamptz,
  status text not null default 'active' check (status in ('active','archived','deleted','needs_review')),
  origin text not null default 'extracted' check (origin in ('extracted','user_edit')),
  event_hash text,
  check (valid_to is null or valid_to > valid_from),
  check ((object_entity_id is null) <> (object_value is null)),
  check (cardinality <> 'event' or valid_to is not null)
);

create table fact_sources (
  fact_id uuid not null references facts on delete cascade,
  source_id uuid not null references sources on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  span_start int, span_end int,
  primary key (fact_id, source_id)
);

create table fact_relations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  from_fact_id uuid not null references facts on delete cascade,
  to_fact_id uuid not null references facts on delete cascade,
  relation text not null check (relation in ('supersedes','supports','clarifies','derived_from')),
  job_id uuid,
  created_at timestamptz not null default now(),
  unique (from_fact_id, to_fact_id, relation)
);
-- direction: (from=new fact) supersedes (to=old fact)

create table jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  type text not null check (type in ('extract','purge','decay','reindex')),
  payload jsonb not null,
  idempotency_key text not null unique,
  coalesce_key text,
  status text not null default 'queued' check (status in ('queued','running','done','dead')),
  attempts int not null default 0,
  max_attempts int not null default 3,
  deferrals int not null default 0,
  run_after timestamptz not null default now(),
  worker_id uuid,
  locked_at timestamptz,
  lease_expires_at timestamptz,
  last_error text,
  error_code text,
  created_at timestamptz not null default now(),
  finished_at timestamptz
);

create table conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  client_message_id text,
  source_id uuid references sources on delete set null,
  partial boolean not null default false,
  trace jsonb,
  created_at timestamptz not null default now(),
  unique (conversation_id, client_message_id)
);

create table audit_log (
  id bigserial primary key,
  user_id uuid not null,
  actor text not null,             -- user | system | worker
  action text not null,            -- fact.edit fact.delete fact.purge fact.resolve fact.supersede ...
  target_type text not null, target_id uuid,
  before jsonb, after jsonb,
  created_at timestamptz not null default now()
);
