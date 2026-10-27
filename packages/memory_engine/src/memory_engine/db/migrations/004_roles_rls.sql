do $$
begin
  if not exists (select from pg_roles where rolname = 'app_api') then
    create role app_api login;
  end if;
  if not exists (select from pg_roles where rolname = 'app_worker') then
    create role app_worker login;
  end if;
  if not exists (select from pg_roles where rolname = 'anon') then
    create role anon;
  end if;
  if not exists (select from pg_roles where rolname = 'authenticated') then
    create role authenticated;
  end if;
end $$;

revoke all on all tables in schema public from anon, authenticated;     -- C-2: no PostgREST access
grant usage on schema public to app_api, app_worker;
grant select, insert, update, delete on all tables in schema public to app_api, app_worker;
grant usage on all sequences in schema public to app_api, app_worker;
revoke insert, update, delete on predicates, app_meta from app_api, app_worker;

-- tenant policy applied to every user-scoped table
do $$ declare t text; begin
  foreach t in array array['sources','entities','entity_aliases','facts','fact_sources','fact_relations',
                           'conversations','messages','audit_log'] loop
    execute format('alter table %I enable row level security', t);
    execute format('alter table %I force row level security', t);
    execute format($p$create policy %I on %I for all to app_api, app_worker
      using (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
      with check (user_id = nullif(current_setting('app.user_id', true), '')::uuid)$p$, t || '_tenant', t);
  end loop;
end $$;

-- profiles: api sees its own row, worker may list users for the nightly scheduler
alter table profiles enable row level security; alter table profiles force row level security;
create policy profiles_api on profiles for all to app_api
  using (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
  with check (user_id = nullif(current_setting('app.user_id', true), '')::uuid);
create policy profiles_worker_read on profiles for select to app_worker using (true);

-- jobs: api is tenant scoped, worker sees all job rows (payload holds ids only, never content)
alter table jobs enable row level security; alter table jobs force row level security;
create policy jobs_api on jobs for all to app_api
  using (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
  with check (user_id = nullif(current_setting('app.user_id', true), '')::uuid);
create policy jobs_worker on jobs for all to app_worker using (true) with check (true);
