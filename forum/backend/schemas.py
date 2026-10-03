from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Paragraph(StrictModel):
    type: Literal["paragraph"]
    text: str = Field(min_length=1, max_length=20000)


class BoardBlock(StrictModel):
    type: Literal["board"]
    rows: int = Field(ge=2, le=4, strict=True)
    cols: int = Field(ge=3, le=4, strict=True)
    cells: list[Annotated[int, Field(strict=True, ge=0, le=1048576)]]
    caption: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def valid_board(self):
        if (self.rows, self.cols) not in {(4, 4), (3, 4), (3, 3), (2, 4)}:
            raise ValueError("Unsupported board dimensions")
        if len(self.cells) != self.rows * self.cols or any(
            v == 1 or (v and v & (v - 1)) for v in self.cells
        ):
            raise ValueError(
                "Cells must be zero or powers of two and match the dimensions"
            )
        return self


Block = Annotated[Paragraph | BoardBlock, Field(discriminator="type")]


class Document(StrictModel):
    version: Literal[1] = 1
    blocks: list[Block] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def bounded(self):
        if len(self.model_dump_json().encode()) > 80000:
            raise ValueError("Document too large")
        return self


class PollInput(StrictModel):
    options: list[str] = Field(min_length=2, max_length=10)
    max_choices: int = Field(default=1, ge=1, le=10)
    closes_at: str = Field(max_length=40)
    results: Literal["always", "voted", "closed"] = "always"

    @model_validator(mode="after")
    def valid_options(self):
        from datetime import datetime

        date = datetime.fromisoformat(self.closes_at)
        if not date.tzinfo:
            raise ValueError("Deadline must include timezone")
        self.options = [o.strip() for o in self.options]
        if (
            any(not o or len(o) > 120 for o in self.options)
            or len(set(self.options)) != len(self.options)
            or self.max_choices > len(self.options)
        ):
            raise ValueError("Invalid poll choices")
        return self


class NewTopic(StrictModel):
    board_slug: str = Field(pattern=r"^[a-z][a-z0-9-]{1,39}$")
    title: str = Field(min_length=3, max_length=120)
    body: Document
    tags: list[str] = Field(default_factory=list, max_length=5)
    kind: Literal["discussion", "question", "poll"] = "discussion"
    poll: PollInput | None = None

    @model_validator(mode="after")
    def poll_matches_kind(self):
        if (self.kind == "poll") != (self.poll is not None):
            raise ValueError("Poll topics require poll configuration")
        return self

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, tags):
        if any(not t.strip() or len(t) > 24 for t in tags):
            raise ValueError("Invalid tags")
        return list(dict.fromkeys(t.strip() for t in tags))


class NewPost(StrictModel):
    body: Document
    reply_to: int | None = Field(default=None, ge=1)


class EditPost(StrictModel):
    body: Document
    revision: int = Field(ge=1)


class PollDraft(StrictModel):
    options: str = Field(default="", max_length=1300)
    closes_at: str = Field(default="", max_length=40)
    max_choices: int = Field(default=1, ge=1, le=10)
    results: Literal["always", "voted", "closed"] = "always"


class DraftInput(StrictModel):
    revision: int = Field(ge=0)
    title: str = Field(default="", max_length=120)
    board_slug: str = Field(default="general", max_length=40)
    text: str = Field(default="", max_length=20000)
    board: BoardBlock | None = None
    kind: Literal["discussion", "question", "poll"] = "discussion"
    tags: list[str] = Field(default_factory=list, max_length=5)
    poll: PollDraft | None = None


class ReportInput(StrictModel):
    reason: str = Field(min_length=3, max_length=1000)


class ModerationInput(StrictModel):
    action: Literal["hide", "restore", "lock", "unlock", "pin", "unpin"]
    reason: str = Field(min_length=3, max_length=1000)


class ReasonInput(StrictModel):
    reason: str = Field(min_length=3, max_length=1000)


class PreferenceInput(StrictModel):
    enabled: bool


class TopicInput(StrictModel):
    title: str = Field(min_length=3, max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=5)
    revision: int = Field(ge=1)

    @field_validator("tags")
    @classmethod
    def valid_tags(cls, tags):
        return NewTopic.clean_tags(tags)


class AdminUserInput(ReasonInput):
    action: Literal["grant", "revoke", "mute", "unmute"]
    board_id: int | None = Field(default=None, ge=1, le=9007199254740991)
    hours: int = Field(default=24, ge=1, le=8760)


class BoardInput(ReasonInput):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(max_length=1000)
    position: int = Field(ge=0, le=10000)
    staff_only: bool


class ReplyDraftInput(StrictModel):
    text: str = Field(default="", max_length=20000)
    reply_to: int | None = Field(default=None, ge=1, le=9007199254740991)
    revision: int = Field(ge=0, le=2147483646)


class ReadingInput(StrictModel):
    post_id: int = Field(ge=1, le=9007199254740991)


class AppealInput(ReasonInput):
    action_id: int = Field(ge=1, le=9007199254740991)


class AppealDecision(ReasonInput):
    decision: Literal["accepted", "rejected"]


class VoteInput(StrictModel):
    choices: list[int] = Field(min_length=1, max_length=10)


class QuestionInput(ReasonInput):
    status: Literal["open", "solved", "closed"]
    accepted_post_id: int | None = Field(default=None, ge=1, le=9007199254740991)
    duplicate_of: int | None = Field(default=None, ge=1, le=9007199254740991)
    revision: int = Field(ge=1)


class MoveTopicInput(ReasonInput):
    board_id: int = Field(ge=1, le=9007199254740991)
    revision: int = Field(ge=1)


class NotificationSettings(StrictModel):
    enabled: bool = True
    categories: dict[
        Literal["reply", "mention", "subscription", "follow", "moderation", "system"],
        bool,
    ] = Field(default_factory=dict)


class AnnouncementInput(StrictModel):
    topic: NewTopic
    sites: list[Literal["forum", "main", "play", "competition", "live"]] = Field(
        default_factory=lambda: ["forum"], min_length=1, max_length=5
    )
    publish_at: str | None = Field(default=None, max_length=40)
    expires_at: str | None = Field(default=None, max_length=40)
    allow_replies: bool = True
    notify_all: bool = False
    schedule: bool = False
    revision: int = Field(default=0, ge=0)


class PrivacyDecision(ReasonInput):
    decision: Literal["completed", "rejected"]


class ExternalCardInput(ReasonInput):
    source: Literal["competition", "live"]
    source_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    revision: int = Field(ge=1)
    title: str = Field(min_length=3, max_length=120)
    summary: str = Field(max_length=2000)
    path: str = Field(pattern=r"^/[a-zA-Z0-9/_?=&%.-]*$", max_length=300)
    status: Literal["published", "withdrawn"]
    topic_id: int | None = Field(default=None, ge=1, le=9007199254740991)
