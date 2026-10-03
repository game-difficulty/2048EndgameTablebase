CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX forum_topic_title_search ON forum_topics USING gin(title gin_trgm_ops);
CREATE INDEX forum_post_body_search ON forum_posts USING gin(body_text gin_trgm_ops) WHERE status='published';
CREATE INDEX forum_topic_tags_search ON forum_topics USING gin(tags);
CREATE FUNCTION forum_short_terms(value text) RETURNS text[] LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT coalesce(array_agg(DISTINCT lower(substr(value,p,n))),ARRAY[]::text[])
    FROM generate_series(1,length(value)) p CROSS JOIN generate_series(1,2) n
    WHERE p+n-1<=length(value)
$$;
CREATE INDEX forum_topic_short_search ON forum_topics USING gin(forum_short_terms(title));
CREATE INDEX forum_post_short_search ON forum_posts USING gin(forum_short_terms(body_text)) WHERE status='published';
ALTER TABLE forum_topics ADD COLUMN kind text NOT NULL DEFAULT 'discussion' CHECK(kind IN ('discussion','question','poll'));
ALTER TABLE forum_topics ADD COLUMN question_status text NOT NULL DEFAULT 'open' CHECK(question_status IN ('open','solved','closed'));
ALTER TABLE forum_topics ADD COLUMN accepted_post_id bigint REFERENCES forum_posts(id);
ALTER TABLE forum_topics ADD COLUMN duplicate_of bigint REFERENCES forum_topics(id);
CREATE TABLE forum_polls (
    topic_id bigint PRIMARY KEY REFERENCES forum_topics(id), options jsonb NOT NULL,
    max_choices integer NOT NULL CHECK(max_choices BETWEEN 1 AND 10),
    closes_at timestamptz NOT NULL, results text NOT NULL CHECK(results IN ('always','voted','closed')),
    closed boolean NOT NULL DEFAULT false
);
CREATE TABLE forum_votes (
    topic_id bigint NOT NULL REFERENCES forum_polls(topic_id), user_id bigint NOT NULL,
    choices jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(topic_id,user_id)
);
CREATE TABLE forum_blocks (
    user_id bigint NOT NULL, kind text NOT NULL CHECK(kind IN ('user','board')), target_id bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(user_id,kind,target_id)
);
ALTER TABLE forum_notification_preferences ADD COLUMN categories jsonb NOT NULL DEFAULT '{}';
CREATE TABLE forum_system_notifications (
    id bigint PRIMARY KEY DEFAULT nextval('forum_notifications_id_seq'), recipient_id bigint NOT NULL,
    kind text NOT NULL CHECK(kind IN ('moderation','system')), title text NOT NULL, body text NOT NULL,
    path text NOT NULL DEFAULT '/community', event_key text NOT NULL,
    read_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(recipient_id,event_key)
);
CREATE INDEX forum_system_inbox ON forum_system_notifications(recipient_id,id DESC);
CREATE TABLE forum_announcements (
    id bigserial PRIMARY KEY, author_id bigint NOT NULL REFERENCES forum_profiles(user_id),
    payload jsonb NOT NULL, sites jsonb NOT NULL DEFAULT '["forum"]',
    publish_at timestamptz, expires_at timestamptz,
    status text NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','scheduled','published','expired','retracted')),
    topic_id bigint REFERENCES forum_topics(id), allow_replies boolean NOT NULL DEFAULT true,
    notify_all boolean NOT NULL DEFAULT false, revision integer NOT NULL DEFAULT 1,
    publish_key uuid NOT NULL UNIQUE, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE forum_privacy_requests (
    id bigserial PRIMARY KEY, user_id bigint NOT NULL REFERENCES forum_profiles(user_id), reason text NOT NULL,
    status text NOT NULL DEFAULT 'open' CHECK(status IN ('open','completed','rejected')),
    decision text, created_at timestamptz NOT NULL DEFAULT now(), decided_at timestamptz
);
CREATE UNIQUE INDEX forum_privacy_pending ON forum_privacy_requests(user_id) WHERE status='open';
CREATE TABLE forum_ip_windows (
    key text PRIMARY KEY, window_started timestamptz NOT NULL DEFAULT now(), count integer NOT NULL
);
CREATE TABLE forum_external_cards (
    id bigserial PRIMARY KEY, source text NOT NULL CHECK(source IN ('competition','live')),
    source_id text NOT NULL, revision integer NOT NULL, title text NOT NULL, summary text NOT NULL,
    path text NOT NULL, status text NOT NULL CHECK(status IN ('published','withdrawn')),
    topic_id bigint REFERENCES forum_topics(id), updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source,source_id)
);
CREATE FUNCTION forum_can_notify(recipient bigint, actor bigint, board bigint, category text) RETURNS boolean LANGUAGE sql STABLE AS $$
    SELECT NOT EXISTS(SELECT 1 FROM forum_notification_preferences p WHERE p.user_id=recipient
        AND (NOT p.enabled OR p.categories->>category='false'))
    AND NOT EXISTS(SELECT 1 FROM forum_blocks b WHERE
        (b.user_id=recipient AND ((b.kind='user' AND b.target_id=actor) OR (b.kind='board' AND b.target_id=board)))
        OR (b.user_id=actor AND b.kind='user' AND b.target_id=recipient))
$$;
