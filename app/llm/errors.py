"""Erreurs du wrapper Claude et du chargement des prompts."""


class PromptError(Exception):
    """Prompt introuvable, mal formé ou variable manquante."""


class ClaudeError(Exception):
    """Erreur métier d'un appel à Claude. Les erreurs réseau restent celles du SDK."""


class ClaudeCredentialsError(ClaudeError):
    """Aucun identifiant Anthropic disponible."""


class ClaudeInvalidOutputError(ClaudeError):
    """La sortie reste invalide après le retry."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ClaudeRefusalError(ClaudeError):
    """Requête refusée, y compris par le modèle de fallback."""

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class ClaudeTruncatedError(ClaudeError):
    """Sortie coupée par max_tokens : relancer à l'identique ne changerait rien."""
