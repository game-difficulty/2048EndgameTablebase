CREATE TABLE forum_boards (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 slug varchar(40) NOT NULL UNIQUE,
 name varchar(80) NOT NULL,
 description text NOT NULL DEFAULT '',
 position integer NOT NULL,
 staff_only boolean NOT NULL DEFAULT false
);
CREATE TABLE forum_profiles (
 user_id bigint PRIMARY KEY CHECK (user_id > 0),
 display_name varchar(120) NOT NULL,
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE forum_roles (
 user_id bigint NOT NULL,
 board_id bigint NOT NULL REFERENCES forum_boards(id),
 role varchar(20) NOT NULL CHECK (role IN ('moderator')),
 PRIMARY KEY(user_id,board_id)
);
CREATE TABLE forum_sanctions (
 user_id bigint PRIMARY KEY,
 reason text NOT NULL,
 expires_at timestamptz NOT NULL,
 operator_id bigint NOT NULL
);
CREATE TABLE forum_topics (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 board_id bigint NOT NULL REFERENCES forum_boards(id),
 author_id bigint NOT NULL REFERENCES forum_profiles(user_id),
 title varchar(120) NOT NULL,
 tags jsonb NOT NULL DEFAULT '[]',
 status varchar(16) NOT NULL DEFAULT 'published' CHECK(status IN ('published','hidden')),
 locked boolean NOT NULL DEFAULT false,
 revision integer NOT NULL DEFAULT 1,
 next_post_number integer NOT NULL DEFAULT 2,
 created_at timestamptz NOT NULL DEFAULT now(),
 last_activity timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX forum_topics_feed ON forum_topics(board_id,status,last_activity DESC,id DESC);
CREATE INDEX forum_topics_global_feed ON forum_topics(status,last_activity DESC,id DESC);
CREATE TABLE forum_posts (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 topic_id bigint NOT NULL REFERENCES forum_topics(id),
 author_id bigint NOT NULL REFERENCES forum_profiles(user_id),
 post_number integer NOT NULL CHECK(post_number > 0),
 reply_to bigint REFERENCES forum_posts(id),
 body jsonb NOT NULL,
 body_text text NOT NULL,
 revision integer NOT NULL DEFAULT 1,
 status varchar(16) NOT NULL DEFAULT 'published' CHECK(status IN ('published','deleted')),
 created_at timestamptz NOT NULL DEFAULT now(),
 edited_at timestamptz,
 UNIQUE(topic_id,post_number)
);
CREATE TABLE forum_post_revisions (
 post_id bigint NOT NULL REFERENCES forum_posts(id),
 revision integer NOT NULL,
 body jsonb NOT NULL,
 editor_id bigint NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(post_id,revision)
);
CREATE TABLE forum_drafts (
 user_id bigint NOT NULL,
 id uuid NOT NULL,
 revision integer NOT NULL DEFAULT 1,
 payload jsonb NOT NULL,
 updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(user_id,id)
);
CREATE TABLE forum_reactions (
 user_id bigint NOT NULL,
 post_id bigint NOT NULL REFERENCES forum_posts(id),
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(user_id,post_id)
);
CREATE TABLE forum_bookmarks (
 user_id bigint NOT NULL,
 topic_id bigint NOT NULL REFERENCES forum_topics(id),
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(user_id,topic_id)
);
CREATE TABLE forum_notifications (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 recipient_id bigint NOT NULL,
 actor_id bigint NOT NULL REFERENCES forum_profiles(user_id),
 post_id bigint NOT NULL REFERENCES forum_posts(id),
 read_at timestamptz,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(recipient_id,post_id)
);
CREATE INDEX forum_notifications_inbox ON forum_notifications(recipient_id,id DESC);
CREATE TABLE forum_reports (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 reporter_id bigint NOT NULL,
 post_id bigint NOT NULL REFERENCES forum_posts(id),
 reason text NOT NULL,
 status varchar(16) NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved')),
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(reporter_id,post_id)
);
CREATE TABLE forum_moderation_actions (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 operator_id bigint NOT NULL,
 topic_id bigint NOT NULL REFERENCES forum_topics(id),
 action varchar(30) NOT NULL,
 reason text NOT NULL,
 previous jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE forum_idempotency (
 user_id bigint NOT NULL,
 key uuid NOT NULL,
 fingerprint varchar(64) NOT NULL,
 result jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(user_id,key)
);
CREATE TABLE forum_outbox (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 kind varchar(50) NOT NULL,
 aggregate_id bigint NOT NULL,
 payload jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 delivered_at timestamptz
);
CREATE INDEX forum_outbox_pending ON forum_outbox(id) WHERE delivered_at IS NULL;
CREATE TABLE forum_rate_windows (
 user_id bigint PRIMARY KEY,
 window_started timestamptz NOT NULL DEFAULT now(),
 count integer NOT NULL DEFAULT 0
);
INSERT INTO forum_boards(slug,name,description,position,staff_only) VALUES
 ('announcements','官方公告','站点更新、维护与社区规则',10,true),
 ('general','综合交流','分享日常、经验和你的 2048 故事',20,false),
 ('showcase','高分与复盘','展示高光，研究每一个关键选择',30,false),
 ('strategy','攻略与提问','从一个局面开始，一起进步',40,false),
 ('events','赛事与直播','赛事讨论、组队和观赛交流',50,false),
 ('feedback','建议与反馈','问题报告、功能建议与处理进展',60,false);
