"""
Mursit tokenizer wrapper.

The "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
paper (Bayram et al., 2025) compares TurkishTokenizer against three
matched baselines, one of which is Mursit — the ModernBERT-based
bidirectional encoder trained from scratch on a Turkish-dominant corpus
released as part of the Mecellem project
(Ugur et al., 2026, "Mecellem Models: Turkish Models Trained from Scratch
and Continually Pre-trained for the Legal Domain", arXiv:2601.16018).

This wrapper provides a unified BaseTokenizer-compatible interface
while additionally supporting dynamic MLM masking using the
standard 80/10/10 masking strategy from the original BERT paper
(Devlin et al., 2019), which is directly applicable to ModernBERT-style
encoders like Mursit.

The tokenizer backend is loaded lazily on first use.

Example
-------
>>> tok = MursitTokenizer()
>>> ids = tok.encode("Türkiye Cumhuriyeti'nin başkenti Ankara'dır.")
>>> tok.decode(ids)
"Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
"""

import random
import torch

from .BaseTokenizer import BaseTokenizer

class MursitTokenizer(BaseTokenizer):
    """Wrapper for the Mursit ModernBERT tokenizer with optional MLM masking."""

    #: HuggingFace repository ID. Override via the constructor argument
    #: when pointing at a local snapshot or a different checkpoint.
    DEFAULT_MODEL_ID = "newmindai/Mursit-Base"

    def __init__(
        self,
        model_name_or_path: str = None,
        mlm_probability: float = 0.15,
        add_special_tokens: bool = False,
        cache_dir: str = None,
        revision: str = None,
        local_files_only: bool = False,
        return_tensors: str = "pt",
        **from_pretrained_kwargs,
    ):
        """
        Parameters
        ----------
        model_name_or_path
            HuggingFace repo ID or path to a local tokenizer snapshot.
            Defaults to :attr:`DEFAULT_MODEL_ID`.
        mlm_probability
            Fraction of tokens to mask in :meth:`mlm_encode`.
            Default is 0.15 (15%) as in the original BERT paper.
        add_special_tokens
            Whether ``encode`` should prepend/append CLS/SEP tokens.
            Defaults to False for clean encode/decode round-trips.
        cache_dir, revision, local_files_only
            Forwarded to ``AutoTokenizer.from_pretrained``.
        return_tensors
            Tensor format for :meth:`mlm_encode` (``"pt"``, ``"np"``).
        from_pretrained_kwargs
            Any additional keyword arguments forwarded to
            ``AutoTokenizer.from_pretrained``.
        """
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_ID
        self.mlm_probability = mlm_probability
        self.add_special_tokens = add_special_tokens
        self.return_tensors = return_tensors
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
                    "MursitTokenizer needs the 'transformers' package. "
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
    # MLM processing (BERT 80/10/10 strategy)
    # ---------------------------------------------------------

    def mlm_encode(self, text: str) -> dict:
        """
        Tokenize ``text`` and apply BERT-style MLM masking.

        Returns
        -------
        dict with keys ``input_ids``, ``labels``, ``attention_mask``
            ``input_ids``  — token IDs with masked positions replaced.
            ``labels``     — original token IDs; -100 at unmasked positions
                             (ignored by cross-entropy loss).
            ``attention_mask`` — 1 for real tokens, 0 for padding.
        """
        inputs = self.tokenizer(text, return_tensors=self.return_tensors)
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = input_ids.clone()

        # Ignore special tokens (CLS, SEP, PAD, …)
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                input_ids, already_has_special_tokens=True
            ),
            dtype=torch.bool,
        )

        # Sample masked positions
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only masked tokens contribute to loss

        # 80 / 10 / 10 strategy
        for i in range(len(input_ids)):
            if not masked_indices[i]:
                continue
            rand = random.random()
            if rand < 0.8:                               # 80%: [MASK]
                input_ids[i] = self.tokenizer.mask_token_id
            elif rand < 0.9:                             # 10%: random token
                input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
            # else 10%: unchanged

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

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
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        loaded = "loaded" if self._tokenizer is not None else "lazy"
        return f"MursitTokenizer(model={self.model_name_or_path!r}, {loaded})"


if __name__ == "__main__":  # small demo
    sentence = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."
    tok = MursitTokenizer()

    print(f"Model  : {tok.model_name_or_path}")
    print(f"Input  : {sentence}")
    ids = tok.encode(sentence)
    print(f"IDs    ({len(ids)}): {ids}")
    print(f"Pieces : {tok.convert_ids_to_tokens(ids)}")
    print(f"Decoded: {tok.decode(ids)}")

    print("\n--- MLM masking demo ---")
    batch = tok.mlm_encode(sentence)
    print("Masked input :", tok.convert_ids_to_tokens(batch["input_ids"]))
    print("Target tokens:", tok.convert_ids_to_tokens(
        batch["labels"][batch["labels"] != -100]
    ))
