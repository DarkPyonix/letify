"""Exception hierarchy.

The split between infrastructure failure and user code failure decides retry
behaviour. A ``RuntimeFailure`` means the session misbehaved and the same call
may succeed on a fresh runtime. A ``RemoteError`` means the shipped function
itself raised, so retrying only reproduces it.
"""

from __future__ import annotations


class LetifyError(Exception):
    """Base class for every error this package raises."""


class ConfigError(LetifyError):
    """A .letify config.toml is malformed, or names an unknown provider kind."""


class ProviderUnavailable(LetifyError):
    """A provider cannot be used, usually because its optional dependency is absent."""

    def __init__(self, kind: str, reason: str, extra: str | None = None):
        self.kind = kind
        self.reason = reason
        self.extra = extra
        hint = f' Install it with: uv add "letify[{extra}]"' if extra else ""
        super().__init__(f"provider {kind!r} is unavailable: {reason}.{hint}")


class UnknownProvider(LetifyError):
    """No provider is registered under the requested alias."""


class UnknownInstance(LetifyError):
    """The provider does not offer an instance under the requested name."""


class NotRunning(LetifyError):
    """A declared function was invoked outside a ``with let.run():`` scope."""


class RuntimeFailure(LetifyError):
    """The remote session failed. Retryable on a fresh runtime."""

    def __init__(self, message: str, *, command: str = "", stderr: str = ""):
        self.command = command
        self.stderr = stderr
        super().__init__(message)


class RuntimeLost(RuntimeFailure):
    """A runtime believed to be alive is gone. Always retryable."""


class RemoteError(LetifyError):
    """The shipped function raised on the remote side. Carries its traceback."""

    def __init__(self, message: str, remote_traceback: str = ""):
        self.remote_traceback = remote_traceback
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if not self.remote_traceback:
            return base
        return f"{base}\n\n--- remote traceback ---\n{self.remote_traceback}"


class ProtocolError(LetifyError):
    """The remote side produced output that could not be parsed.

    The usual cause is a dead kernel: an out of memory kill, a preempted
    session, or a crash below the Python level.
    """


class InsufficientDevices(LetifyError):
    """The devices a call needs cannot be allocated, and nothing running would free them.

    Raised instead of waiting when every holder of the accelerator is an idle session kept by
    ``let.keep_alive()``, when the cards are taken by another process, or when a call asks for
    more devices than the account declares. Not retried, because a retry asks for the same
    devices from the same inventory.
    """


class UnsupportedMode(LetifyError):
    """The requested execution mode cannot work on this provider.

    Raised, for example, when ``host="local"`` is asked for on a provider with no
    low-latency data path. letify never downgrades silently.
    """


class EnvironmentFailure(RuntimeFailure):
    """uv could not be installed on a runtime, or ``uv sync`` failed there. Retryable."""


class InterpreterMismatch(LetifyError):
    """The runtime's Python major.minor differs from this process, or would.

    Not retried, because a fresh runtime builds the same interpreter.
    """
