"""
TabiBERT tokenizer wrapper.

The "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
paper (Bayram et al., 2025) compares TurkishTokenizer against three
matched baselines, one of which is TabiBERT — a large-scale ModernBERT
foundation model for Turkish NLP developed by the BOUN TABILab
(Safaya et al., "TabiBERT: A Large-Scale ModernBERT Foundation for
Turkish NLP", 2025, huggingface.co/boun-tabilab/TabiBERT).

This class is a thin :class:`BaseTokenizer` wrapper around the
HuggingFace ``transformers.AutoTokenizer`` so the rest of the project
can treat all four tokenizers uniformly. The underlying model is loaded
lazily on the first call to :meth:`encode` / :meth:`decode`.

Example
-------
>>> tok = TabiTokenizer()
>>> ids = tok.encode("Türkiye Cumhuriyeti'nin başkenti Ankara'dır.")
>>> tok.decode(ids)
"Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
"""

from .BaseTokenizer import BaseTokenizer

class TabiTokenizer(BaseTokenizer):
    """Wrapper for the TabiBERT ModernBERT tokenizer."""

    #: HuggingFace repository ID. Override via the constructor argument
    #: when pointing at a local snapshot or a different checkpoint.
    DEFAULT_MODEL_ID = "boun-tabilab/TabiBERT"

    def __init__(
        self,
        model_name_or_path: str = None,
        add_special_tokens: bool = False,
        cache_dir: str = None,
        revision: str = None,
        local_files_only: bool = False,
        **from_pretrained_kwargs,
    ):
        """
        Parameters
        ----------
        model_name_or_path
            HuggingFace repo ID or path to a local tokenizer snapshot.
            Defaults to :attr:`DEFAULT_MODEL_ID`.
        add_special_tokens
            Whether ``encode`` should prepend/append CLS/SEP tokens.
            Defaults to False for clean encode/decode round-trips.
        cache_dir, revision, local_files_only
            Forwarded to ``AutoTokenizer.from_pretrained``.
        from_pretrained_kwargs
            Any additional keyword arguments forwarded to
            ``AutoTokenizer.from_pretrained``.
        """
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_ID
        self.add_special_tokens = add_special_tokens
        self._cache_dir = cache_dir
        self._revision = revision
        self._local_files_only = local_files_only
        self._extra_kwargs = from_pretrained_kwargs
        self._tokenizer = None  # lazy

    # ---------------------------------------------------------
    # Lazy loading
    # ---------------------------------------------------------

    @property
    def tokenizer(self):
        """The lazily-instantiated HuggingFace tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "TabiTokenizer needs the 'transformers' package. "
                    "Install it with `pip install transformers`."
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self._cache_dir,
                revision=self._revision,
                local_files_only=self._local_files_only,
                **self._extra_kwargs,
            )
        return self._tokenizer

    # ---------------------------------------------------------
    # BaseTokenizer API
    # ---------------------------------------------------------

    def encode(self, text: str) -> list:
        """Encode ``text`` to a list of integer token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

    def decode(self, token_ids) -> str:
        """Decode a list of integer token IDs back to a string."""
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=not self.add_special_tokens,
        )

    # ---------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------

    def tokenize(self, text: str) -> list:
        """Return raw subword token strings (no ID conversion)."""
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, token_ids) -> list:
        return self.tokenizer.convert_ids_to_tokens(list(token_ids))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    def __repr__(self):
        loaded = "loaded" if self._tokenizer is not None else "lazy"
        return f"TabiTokenizer(model={self.model_name_or_path!r}, {loaded})"


if __name__ == "__main__":  # small demo
    sentence = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
    tok = TabiTokenizer()
    print(f"Model  : {tok.model_name_or_path}")
    print(f"Input  : {sentence}")
    ids = tok.encode(sentence)
    print(f"IDs    ({len(ids)}): {ids}")
    print(f"Pieces : {tok.convert_ids_to_tokens(ids)}")
    print(f"Decoded: {tok.decode(ids)}")
