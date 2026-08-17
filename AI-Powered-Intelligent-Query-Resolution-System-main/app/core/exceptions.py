"""Authentication domain exceptions."""


class AuthError(Exception):
    """Base authentication error."""


class EmailAlreadyRegisteredError(AuthError):
    """Raised when signup email already exists."""


class InvalidCredentialsError(AuthError):
    """Raised when login credentials are invalid."""


class UnauthorizedError(AuthError):
    """Raised when an access token is missing or invalid."""
