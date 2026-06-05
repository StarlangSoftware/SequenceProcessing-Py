from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Converts raw text to a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of token IDs back into raw text."""
        pass