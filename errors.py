class UnknownVerbError(ValueError):
    """Raised when a command verb is not recognized."""

    def __init__(self, verb: str, suggestion: str | None = None) -> None:
        message = f"Sorry, I don't understand: '{verb}'"
        if suggestion:
            message += f". Did you mean: '{suggestion}'?"
        super().__init__(message)
        self.verb = verb
        self.suggestion = suggestion


class MissingParameterError(ValueError):
    """Raised when required parameters are missing."""

    def __init__(self, params: list[str], suggestion: str | None = None) -> None:
        missing = ", ".join(params)
        message = f"Missing parameters: {missing}"
        if suggestion:
            message += f". Did you mean: '{suggestion}'?"
        super().__init__(message)
        self.params = params
        self.suggestion = suggestion


class UnsupportedLibraryError(ValueError):
    """Raised when an optional library required for a task isn't available."""

    def __init__(self, library: str) -> None:
        super().__init__(f"Unsupported library: {library}")
        self.library = library
