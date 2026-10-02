CREATE TABLE forum_follows (
    user_id bigint NOT NULL REFERENCES forum_profiles(user_id),
    followed_id bigint NOT NULL REFERENCES forum_profiles(user_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(user_id,followed_id), CHECK(user_id<>followed_id)
);
CREATE INDEX forum_follows_author ON forum_follows(followed_id,created_at);
CREATE TABLE forum_mentions (
    post_id bigint NOT NULL REFERENCES forum_posts(id),
    user_id bigint NOT NULL REFERENCES forum_profiles(user_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(post_id,user_id)
);
ALTER TABLE forum_notifications ADD COLUMN kind text NOT NULL DEFAULT 'reply'
    CHECK(kind IN ('reply','subscription','follow','mention'));
CREATE TABLE forum_reading (
    user_id bigint NOT NULL, topic_id bigint NOT NULL REFERENCES forum_topics(id),
    post_id bigint NOT NULL REFERENCES forum_posts(id), post_number integer NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(user_id,topic_id)
);
CREATE INDEX forum_reading_recent ON forum_reading(user_id,updated_at DESC);
CREATE TABLE forum_reply_drafts (
    user_id bigint NOT NULL, topic_id bigint NOT NULL REFERENCES forum_topics(id),
    text text NOT NULL DEFAULT '', reply_to bigint REFERENCES forum_posts(id),
    revision integer NOT NULL DEFAULT 1, updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(user_id,topic_id), CHECK(length(text)<=20000)
);
CREATE TABLE forum_appeals (
    id bigserial PRIMARY KEY, user_id bigint NOT NULL REFERENCES forum_profiles(user_id),
    action_id bigint NOT NULL REFERENCES forum_moderation_actions(id),
    reason text NOT NULL, status text NOT NULL DEFAULT 'open' CHECK(status IN ('open','accepted','rejected')),
    decision text, reviewer_id bigint REFERENCES forum_profiles(user_id),
    created_at timestamptz NOT NULL DEFAULT now(), decided_at timestamptz,
    UNIQUE(user_id,action_id)
);
CREATE INDEX forum_appeals_open ON forum_appeals(id DESC) WHERE status='open';
