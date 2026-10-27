create extension if not exists vector;
create extension if not exists pgcrypto;

create table app_meta (
  key text primary key,
  value text not null
);
