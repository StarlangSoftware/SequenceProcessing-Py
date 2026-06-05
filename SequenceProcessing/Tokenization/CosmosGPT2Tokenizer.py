"""
CosmosGPT2 tokenizer wrapper.

The "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
paper (Bayram et al., 2025) compares TurkishTokenizer against three
matched baselines, one of which is CosmosGPT2 — the BPE tokenizer that
ships with the Cosmos research group's Turkish GPT-2 models
(Kesgin et al., 2024, "Introducing cosmosGPT: Monolingual Training for
Turkish Language Models", arXiv:2404.17336).

This class is a thin :class:`BaseTokenizer` wrapper around the
HuggingFace ``transformers.AutoTokenizer`` so the rest of the project
can treat all four tokenizers uniformly. The underlying model is loaded
lazily on the first call to :meth:`encode` / :meth:`decode`, which keeps
``import`` cheap and lets callers run unit tests without needing
network access.

Example
-------
>>> tok = CosmosGPT2Tokenizer()
>>> ids = tok.encode("Türkiye Cumhuriyeti'nin başkenti Ankara'dır.")
>>> tok.decode(ids)
"Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
"""

from .BaseTokenizer import BaseTokenizer


class CosmosGPT2Tokenizer(BaseTokenizer):
    """HuggingFace-backed wrapper for the CosmosGPT2 BPE tokenizer."""

    #: HuggingFace repository ID. Override via the constructor argument
    #: when pointing at a local snapshot or a different checkpoint.
    DEFAULT_MODEL_ID = "ytu-ce-cosmos/turkish-gpt2-medium"

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
        Constructor for CosmosGPT2Tokenizer.

        :param model_name_or_path: HuggingFace repo ID or path to a local
            tokenizer snapshot. Defaults to :attr:`DEFAULT_MODEL_ID`
            (``"ytu-ce-cosmos/turkish-gpt2-medium"``).
        :param add_special_tokens: Whether ``encode`` should prepend/append
            the tokenizer's special tokens (BOS/EOS). The CosmosGPT2 paper
            trains its GPT-2 models without explicit BOS/EOS, so the default
            is ``False`` to keep encode/decode round-tripping cleanly.
        :param cache_dir: Directory to cache downloaded tokenizer files.
            Forwarded to ``AutoTokenizer.from_pretrained``.
        :param revision: Specific model version (branch, tag, or commit hash).
            Forwarded to ``AutoTokenizer.from_pretrained``.
        :param local_files_only: If ``True``, only use locally cached files
            and raise an error if the model is not already downloaded.
            Forwarded to ``AutoTokenizer.from_pretrained``.
        :param from_pretrained_kwargs: Any additional keyword arguments
            forwarded to ``AutoTokenizer.from_pretrained``
            (e.g. ``trust_remote_code=True``).
        """
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_ID
        self.add_special_tokens = add_special_tokens
        self._cache_dir = cache_dir
        self._revision = revision
        self._local_files_only = local_files_only
        self._extra_kwargs = from_pretrained_kwargs
        self._tokenizer = None  # lazy

    # ------------------------------------------------------------------ LOAD

    @property
    def tokenizer(self):
        """
        The lazily-instantiated HuggingFace tokenizer.

        The underlying ``AutoTokenizer`` is created on first access and then
        cached in :attr:`_tokenizer`. This ensures that importing the module
        or constructing the wrapper does not trigger a network request.

        :raises ImportError: If the ``transformers`` package is not installed.
        :return: Loaded ``AutoTokenizer`` instance.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "CosmosGPT2Tokenizer needs the 'transformers' package. "
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

    # --------------------------------------------------------- BaseTokenizer

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of integer BPE token IDs.

        Delegates directly to the underlying HuggingFace tokenizer. Whether
        BOS/EOS special tokens are prepended/appended is controlled by the
        :attr:`add_special_tokens` flag set at construction time.

        :param text: Input string to tokenize.
        :return: List of integer token IDs.
        """
        return self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

    def decode(self, token_ids) -> str:
        """
        Decode a list of integer token IDs back to a string.

        Delegates to the underlying HuggingFace tokenizer. Special tokens
        (BOS/EOS) are stripped when :attr:`add_special_tokens` is ``False``.

        :param token_ids: Sequence of integer token IDs (list, tuple, or
            any iterable).
        :return: Decoded surface-form string.
        """
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=not self.add_special_tokens,
        )

    # --------------------------------------------------------- CONVENIENCE

    def tokenize(self, text: str) -> list:
        """
        Return the raw BPE subword token strings without converting to IDs.

        Useful for inspecting how the text is split before the vocabulary
        lookup step.

        :param text: Input string to tokenize.
        :return: List of subword token strings (e.g. ``['Türk', 'iye']``).
        """
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, token_ids) -> list:
        """
        Map a sequence of integer token IDs to their string representations.

        :param token_ids: Sequence of integer token IDs.
        :return: List of token strings corresponding to each ID.
        """
        return self.tokenizer.convert_ids_to_tokens(list(token_ids))

    @property
    def vocab_size(self) -> int:
        """
        Total number of tokens in the BPE vocabulary.

        :return: integer vocab size
        """
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        """
        ID of the padding token, or ``None`` if the tokenizer has no pad token.

        :return: integer pad token ID, or ``None``
        """
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        """
        ID of the end-of-sequence token, or ``None`` if not defined.

        :return: integer EOS token ID, or ``None``
        """
        return self.tokenizer.eos_token_id

    def __repr__(self):
        """
        Return a short string representation showing the model path and
        whether the underlying tokenizer has been loaded yet.

        :return: String of the form
            ``CosmosGPT2Tokenizer(model='...', loaded)`` or
            ``CosmosGPT2Tokenizer(model='...', lazy)``.
        """
        loaded = "loaded" if self._tokenizer is not None else "lazy"
        return f"CosmosGPT2Tokenizer(model={self.model_name_or_path!r}, {loaded})"


if __name__ == "__main__":  # small demo
    sentence = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
    tok = CosmosGPT2Tokenizer()
    print(f"Model: {tok.model_name_or_path}")
    print(f"Input: {sentence}")
    ids = tok.encode(sentence)
    print(f"IDs   ({len(ids)}): {ids}")
    print(f"Pieces: {tok.convert_ids_to_tokens(ids)}")
    print(f"Decoded: {tok.decode(ids)}")
