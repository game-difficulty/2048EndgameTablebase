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


class NewTopic(StrictModel):
    board_slug: str = Field(pattern=r"^[a-z][a-z0-9-]{1,39}$")
    title: str = Field(min_length=3, max_length=120)
    body: Document
    tags: list[str] = Field(default_factory=list, max_length=5)

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


class DraftInput(StrictModel):
    revision: int = Field(ge=0)
    title: str = Field(default="", max_length=120)
    board_slug: str = Field(default="general", max_length=40)
    text: str = Field(default="", max_length=20000)
    board: BoardBlock | None = None


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
