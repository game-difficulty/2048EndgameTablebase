CREATE TABLE forum_media (
 id uuid PRIMARY KEY,
 owner_id bigint NOT NULL REFERENCES forum_profiles(user_id),
 kind varchar(16) NOT NULL CHECK(kind IN ('image','replay','play')),
 mime varchar(64) NOT NULL,
 data bytea NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}',
 size integer NOT NULL CHECK(size>=0),
 status varchar(16) NOT NULL DEFAULT 'active' CHECK(status IN ('active','removed')),
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX forum_media_owner ON forum_media(owner_id,created_at DESC);
CREATE TABLE forum_post_media (
 post_id bigint NOT NULL REFERENCES forum_posts(id),
 media_id uuid NOT NULL REFERENCES forum_media(id),
 PRIMARY KEY(post_id,media_id)
);
CREATE INDEX forum_post_media_asset ON forum_post_media(media_id);
CREATE TABLE forum_subscriptions (
 user_id bigint NOT NULL,
 kind varchar(8) NOT NULL CHECK(kind IN ('topic','board')),
 target_id bigint NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(user_id,kind,target_id)
);
CREATE TABLE forum_notification_preferences (
 user_id bigint PRIMARY KEY,
 enabled boolean NOT NULL DEFAULT true
);
ALTER TABLE forum_outbox ADD COLUMN attempts integer NOT NULL DEFAULT 0;
ALTER TABLE forum_outbox ADD COLUMN next_attempt_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE forum_outbox ADD COLUMN last_error varchar(100);
ALTER TABLE forum_topics ADD COLUMN pinned boolean NOT NULL DEFAULT false;
ALTER TABLE forum_moderation_actions ALTER COLUMN topic_id DROP NOT NULL;
ALTER TABLE forum_moderation_actions ADD COLUMN target_type varchar(20) NOT NULL DEFAULT 'topic';
ALTER TABLE forum_moderation_actions ADD COLUMN target_id varchar(80);
