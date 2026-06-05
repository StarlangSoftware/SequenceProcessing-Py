"""
TurkishTokenizer implementation following Algorithm 1 (encode) and
Algorithm 2 (decode) of

    Bayram, M. A., Fincan, A. A., Gümüş, A. S., Karakaş, S., Diri, B.,
    Yıldırım, S., Çelik, D. "Tokens with Meaning: A Hybrid Tokenization
    Approach for Turkish", 2025.
"""

import string

from .BaseTokenizer import BaseTokenizer


class TurkishTokenizer(BaseTokenizer):
    """
    Hybrid morphological tokenizer for Turkish.

    Encodes text into a sequence of linguistically meaningful token IDs
    by decomposing each word into a root token and zero or more suffix
    tokens using a morphological analyzer. Capitalization is encoded as
    a separate ``<uppercase>`` token so it survives the lowercase
    normalization step. Punctuation characters are wrapped as ``<P:x>``
    tokens. Words that the analyzer cannot parse fall back to a supplied
    BPE encoder, or to ``<unk>`` if no fallback is provided.

    Class-level special token constants
    ------------------------------------
    ID_UPPERCASE : str
        Token emitted before a capitalised word. Value ``"<uppercase>"``.
    ID_UNK : str
        Token emitted for out-of-vocabulary words with no BPE fallback.
        Value ``"<unk>"``.
    ID_SPACE : str
        Surface form used internally to represent a leading space on a
        root form. Value ``"<space>"``.
    """

    ID_UPPERCASE = "<uppercase>"
    ID_UNK = "<unk>"
    ID_SPACE = "<space>"
    _PUNCT_PREFIX = "<P:"
    _PUNCT_SUFFIX = ">"

    def __init__(self, morph_analyzer=None, bpe_fallback=None):
        """
        Constructor for TurkishTokenizer.

        :param morph_analyzer: A :class:`MorphologicalAnalyzer` instance that
            provides ``analyze(word)`` and ``reverse_lookup(token_id)`` etc.
            Required for :meth:`encode`; raises ``ValueError`` if ``None``
            when encoding is attempted.
        :param bpe_fallback: Any object with an ``encode(text) -> list``
            method used for words the morphological analyzer cannot parse.
            If ``None``, unrecognised words are mapped to :attr:`ID_UNK`.
        """
        self.morph_analyzer = morph_analyzer
        self.bpe_fallback = bpe_fallback

    # ------------------------------------------------------------------ ENCODE

    def _preprocess(self, raw_text: str) -> str:
        """
        Isolate every punctuation character with surrounding spaces so that
        the subsequent ``split()`` treats each punctuation mark as its own
        token candidate.

        :param raw_text: Original input string.
        :return: String with spaces inserted around every punctuation character.
        """
        text = raw_text
        for p in string.punctuation:
            text = text.replace(p, " " + p + " ")
        return text

    def encode(self, text: str) -> list:
        """
        Encode a Turkish string into a list of token IDs (Algorithm 1).

        Processing steps for each whitespace-delimited word:

        1. If the word is a single punctuation character, emit ``<P:x>``.
        2. If the word starts with an uppercase letter, emit
           ``<uppercase>`` and lowercase the word (Turkish-aware).
        3. Run the morphological analyzer; on success emit the root ID
           followed by all suffix IDs.
        4. If the analyzer fails and a BPE fallback is present, delegate
           to it; otherwise emit ``<unk>``.

        :param text: Raw Turkish text to tokenize.
        :return: List of token IDs (strings for morphological tokens,
            integers if the BPE fallback returns integers).
        :raises ValueError: If :attr:`morph_analyzer` is ``None``.
        """
        if self.morph_analyzer is None:
            raise ValueError("TurkishTokenizer requires a morph_analyzer.")

        token_ids = []
        processed = self._preprocess(text)

        for w in processed.split():
            if len(w) == 1 and w in string.punctuation:
                token_ids.append(self._punct_id(w))
                continue

            if self._is_capitalized(w):
                token_ids.append(self.ID_UPPERCASE)
                w = self._turkish_lower(w)

            root_id, suffix_ids = self.morph_analyzer.analyze(w)
            if root_id is not None:
                token_ids.append(root_id)
                token_ids.extend(suffix_ids)
            elif self.bpe_fallback is not None:
                token_ids.extend(self.bpe_fallback.encode(w))
            else:
                token_ids.append(self.ID_UNK)

        return token_ids

    # ------------------------------------------------------------------ DECODE

    def decode(self, token_ids: list) -> str:
        """
        Reconstruct a surface-form string from a list of token IDs
        (Algorithm 2).

        Processing steps for each token ID:

        1. If the token is ``<uppercase>``, resolve the *next* token to its
           surface form and capitalize the first character (Turkish-aware),
           then advance the index by two.
        2. Otherwise, look up all candidate surface forms via
           ``reverse_lookup``. If there is more than one candidate (i.e. the
           token is an ambiguous suffix allomorph), call ``apply_phonology``
           to select the correct form based on vowel-harmony context.
        3. Unwrap special tokens (``<P:x>`` → ``x``, ``<space>`` → ``" "``).

        :param token_ids: Sequence of token IDs as produced by :meth:`encode`.
        :return: Reconstructed Turkish string.
        """
        parts    = []
        part_ids = []   # parallel list — token ID that produced each part
        i = 0
        n = len(token_ids)

        while i < n:
            tid = token_ids[i]

            if tid == self.ID_UPPERCASE and i + 1 < n:
                next_tid = token_ids[i + 1]
                base = self._surface_of(next_tid, parts)
                lead = ""
                while base.startswith(" "):
                    lead += " "
                    base = base[1:]
                parts.append(lead + self._turkish_capitalize(base))
                part_ids.append(next_tid)
                i += 2
                continue

            candidates = self.morph_analyzer.reverse_lookup(tid)

            # Decide whether to call apply_phonology:
            #   (a) Multiple affix allomorphs → vowel-harmony selection needed.
            #   (b) Any affix whose preceding ROOT has allomorphs → the root
            #       may require lenition / narrowing (e.g. kitap→kitab before
            #       a vowel-initial suffix).  Loanwords with a single root
            #       form (cumhuriyet, etc.) are intentionally excluded.
            is_affix = tid in self.morph_analyzer.inv_affix_dict
            prev_tid = part_ids[-1] if part_ids else None
            prev_has_allomorphs = (
                prev_tid is not None
                and prev_tid in self.morph_analyzer.root_ids_with_allomorphs
            )
            use_phonology = len(candidates) > 1 or (is_affix and prev_has_allomorphs)

            if use_phonology:
                ctx = self.morph_analyzer.get_vowel_context(parts)
                surface = self.morph_analyzer.apply_phonology(tid, ctx, parts)
            else:
                surface = candidates[0]

            parts.append(self._unwrap_special(surface))
            part_ids.append(tid)
            i += 1

        return "".join(parts)

    # ------------------------------------------------------------------ HELPERS

    def _surface_of(self, tid, parts: list) -> str:
        """
        Resolve a single token ID to its surface form, applying phonology
        when there are multiple allomorph candidates.

        Used internally by :meth:`decode` to handle the token that follows
        an ``<uppercase>`` marker before capitalisation is applied.

        :param tid: Token ID to resolve.
        :param parts: Surface-form parts built so far (used for vowel-harmony
            context).
        :return: Surface string for ``tid``.
        """
        candidates = self.morph_analyzer.reverse_lookup(tid)
        if len(candidates) > 1:
            ctx = self.morph_analyzer.get_vowel_context(parts)
            return self.morph_analyzer.apply_phonology(tid, ctx, parts)
        return self._unwrap_special(candidates[0])

    @classmethod
    def _punct_id(cls, char: str) -> str:
        """
        Wrap a single punctuation character in the ``<P:x>`` token format.

        :param char: Single punctuation character.
        :return: Token string of the form ``"<P:x>"``.
        """
        return cls._PUNCT_PREFIX + char + cls._PUNCT_SUFFIX

    @classmethod
    def _unwrap_special(cls, surface) -> str:
        """
        Convert internal special-token strings to their printable equivalents.

        * ``"<P:x>"``   → ``"x"`` (punctuation character)
        * ``"<space>"`` → ``" "`` (literal space)
        * Anything else is returned unchanged.

        :param surface: Surface form as returned by the morphological analyzer.
        :return: Printable surface string.
        """
        if isinstance(surface, str):
            if surface.startswith(cls._PUNCT_PREFIX) and surface.endswith(cls._PUNCT_SUFFIX):
                return surface[len(cls._PUNCT_PREFIX):-len(cls._PUNCT_SUFFIX)]
            if surface == cls.ID_SPACE:
                return " "
        return surface

    # --------------------------------------------------------- TURKISH CASING

    _TR_LOWER = str.maketrans({"İ": "i", "I": "ı"})
    _TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})

    @classmethod
    def _turkish_lower(cls, word: str) -> str:
        """
        Lowercase a word using Turkish-specific rules.

        Standard Python ``str.lower()`` maps ``I → i`` and ``İ → i``, which
        is incorrect for Turkish. This method maps ``İ → i`` and ``I → ı``
        before delegating to ``str.lower()``.

        :param word: Input word.
        :return: Lowercased word with correct Turkish mappings.
        """
        return word.translate(cls._TR_LOWER).lower()

    @classmethod
    def _turkish_upper_char(cls, ch: str) -> str:
        """
        Uppercase a single character using Turkish-specific rules.

        Maps ``i → İ`` and ``ı → I`` before delegating to ``str.upper()``.

        :param ch: Single input character.
        :return: Uppercased character with correct Turkish mapping.
        """
        return ch.translate(cls._TR_UPPER).upper()

    @classmethod
    def _is_capitalized(cls, word: str) -> bool:
        """
        Return ``True`` if the word starts with an uppercase letter,
        including the Turkish capital ``İ`` which ``str.isupper()`` handles
        correctly but which is explicitly included for clarity.

        :param word: Word to check.
        :return: ``True`` if the first character is uppercase or ``İ``.
        """
        if not word:
            return False
        first = word[0]
        return first.isupper() or first == "İ"

    @classmethod
    def _turkish_capitalize(cls, word: str) -> str:
        """
        Capitalize the first character of a word using Turkish-specific rules.

        :param word: Input word (assumed to be lowercase).
        :return: Word with its first character uppercased according to Turkish
            conventions (``i → İ``, ``ı → I``).
        """
        if not word:
            return word
        return cls._turkish_upper_char(word[0]) + word[1:]
