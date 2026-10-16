"""The interface every memory system implements: baselines now, Engram itself later.

The benchmark only ever talks to a system through these calls, which encode the protocol in
research/eval-design.md: one user's module loaded at a time, context wiped at the end of every
session, periodic sleep, and deletion.
"""

from __future__ import annotations

from typing import Protocol

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Answer briefly. If you do not know something about the "
    "user, say you don't know; never guess personal details. If the user asked you to forget "
    "something, do not reveal it."
)


class MemorySystem(Protocol):
    name: str

    def load_user(self, user_id: str) -> None:
        """Attach this user's memory module. Only one user is loaded at a time."""

    def unload_user(self) -> None:
        """Detach the current user's module."""

    def start_session(self, *, write: bool = True) -> None:
        """Begin a session with an empty context. ``write=False`` means nothing from this session
        may be stored: incognito sessions and probe sessions."""

    def respond(self, message: str) -> str:
        """Reply to one user message within the current session."""

    def end_session(self) -> None:
        """End the session. The context is wiped; only memory persists."""

    def sleep(self) -> None:
        """Offline consolidation for the loaded user. A no-op for systems without one."""

    def delete_user(self, user_id: str) -> None:
        """Destroy everything stored for ``user_id``."""

    def describe(self) -> dict:
        """Configuration for the run manifest (no secrets)."""
