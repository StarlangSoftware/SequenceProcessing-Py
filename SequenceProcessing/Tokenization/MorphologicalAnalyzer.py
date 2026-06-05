import json
import os


class MorphologicalAnalyzer:
    """
    Morphological analyzer used by TurkishTokenizer.

    Implements the dictionary-driven analyze / reverse-lookup pieces of
    Algorithm 1 (encoder) and Algorithm 2 (decoder) from
    "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
    (Bayram et al., 2025).

    The full lexical resources released with the paper contain
        - 22,231 root tokens mapped to 20,000 canonical root IDs,
        - 177 suffix allomorphs mapped to 72 affix IDs.
    These can be supplied as JSON files via ``roots_path`` / ``affixes_path``.
    If the files are missing, a small dummy dictionary is used so that the
    tokenization pipeline remains testable without external resources.
    """

    VOWELS = "aeıioöuü"
    BACK_VOWELS = "aıou"
    FRONT_VOWELS = "eiöü"
    ROUNDED_VOWELS = "oöuü"
    UNROUNDED_VOWELS = "aeıi"

    # Voiceless consonants ("fıstıkçı şahap") triggering d -> t assimilation.
    VOICELESS_CONSONANTS = "fstkçşhp"

    # Stem-final stops that lenite before a vowel-initial suffix.
    LENITION_MAP = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}

    # Vowel narrowing before progressive -yor.
    NARROWING_MAP = {"a": "ı", "e": "i"}

    def __init__(self, roots_path: str = None, affixes_path: str = None):
        self.root_dict = self._load_dict(roots_path)
        self.affix_dict = self._load_dict(affixes_path)

        if not self.root_dict:
            self.root_dict = {"araba": "R_1", "gel": "R_2", "kitap": "R_3"}
        if not self.affix_dict:
            self.affix_dict = {
                "lar": "A_1",
                "ler": "A_1",
                "da": "A_2",
                "de": "A_2",
            }

        # Keep the FIRST surface form seen for each root ID so that
        # phonological variants (e.g. "kitap"/"kitab"/"kita" → same ID)
        # decode back to the canonical base form.
        self.inv_root_dict = {}
        for surface_form, root_id in self.root_dict.items():
            if root_id not in self.inv_root_dict:
                self.inv_root_dict[root_id] = surface_form

        # Affixes keep ALL allomorphs per ID so apply_phonology can
        # choose the correct surface form during decode.
        self.inv_affix_dict = {}
        for surface_form, affix_id in self.affix_dict.items():
            self.inv_affix_dict.setdefault(affix_id, []).append(surface_form)

        # Root IDs that have more than one surface form (phonological
        # allomorphs, e.g. "kitap"/"kitab"). Only these stems undergo
        # lenition / vowel alternation during decode.
        from collections import Counter
        _counts = Counter(self.root_dict.values())
        self.root_ids_with_allomorphs = frozenset(
            rid for rid, cnt in _counts.items() if cnt > 1
        )

        self.vowels = self.VOWELS
        self.back_vowels = self.BACK_VOWELS
        self.front_vowels = self.FRONT_VOWELS

    # I/O
    @staticmethod
    def _load_dict(path):
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    # Algorithm 1 hook

    def analyze(self, word):
        candidates = [word]
        if not word.startswith(" "):
            candidates.append(" " + word)

        for candidate in candidates:
            for i in range(len(candidate), 0, -1):
                root_candidate = candidate[:i]
                if root_candidate not in self.root_dict:
                    continue
                root_id = self.root_dict[root_candidate]
                remainder = candidate[i:]
                if not remainder:
                    return root_id, []
                suffix_ids = self._extract_suffixes(remainder)
                if suffix_ids is not None:
                    return root_id, suffix_ids
        return None, []

    def _extract_suffixes(self, remainder):
        suffixes = []
        current = remainder
        while current:
            matched = False
            for length in range(len(current), 0, -1):
                piece = current[:length]
                if piece in self.affix_dict:
                    suffixes.append(self.affix_dict[piece])
                    current = current[length:]
                    matched = True
                    break
            if not matched:
                return None
        return suffixes

    # Algorithm 2 hooks

    def lookup_base_string(self, token_id):
        if token_id in self.inv_root_dict:
            return self.inv_root_dict[token_id].lstrip(" ")
        if token_id in self.inv_affix_dict:
            return self.inv_affix_dict[token_id][0]
        return str(token_id) if token_id is not None else ""

    def reverse_lookup(self, token_id):
        if token_id in self.inv_root_dict:
            return [self.inv_root_dict[token_id]]
        if token_id in self.inv_affix_dict:
            return list(self.inv_affix_dict[token_id])
        return [str(token_id)]

    def get_vowel_context(self, parts):
        joined = "".join(parts).lower()
        for char in reversed(joined):
            if char in self.VOWELS:
                return char
        return "a"

    def apply_phonology(self, token_id, ctx, parts):
        """
        Algorithm 2, line 13: pick the correct allomorph for ``token_id``.

        Implements the five rules listed in the paper:
            1. Vowel harmony (front/back, optionally rounded)
            2. Consonant assimilation (d -> t after voiceless consonants)
            3. Stem-final lenition (p/k/t/ç -> b/ğ/d/c before a vowel-initial suffix)
            4. Vowel narrowing (e -> i, a -> ı) before progressive -yor
            5. Buffer-consonant insertion (y / n / s) between vowels
        """
        candidates = self.inv_affix_dict.get(token_id, [])
        if not candidates:
            return str(token_id)

        # (1) Vowel harmony
        harmonized = self._select_by_vowel_harmony(candidates, ctx)
        surface = harmonized if harmonized else candidates[0]
        prev_surface = parts[-1] if parts else ""

        # (2) d -> t assimilation
        if surface and surface[0] == "d" and prev_surface:
            last_char = prev_surface[-1].lower()
            if last_char in self.VOICELESS_CONSONANTS:
                surface = "t" + surface[1:]

        # (3) Lenition of stem-final p/k/t/ç before vowel-initial suffix
        if surface and surface[0] in self.VOWELS and prev_surface:
            last_char = prev_surface[-1].lower()
            if last_char in self.LENITION_MAP:
                parts[-1] = prev_surface[:-1] + self.LENITION_MAP[last_char]
                prev_surface = parts[-1]

        # (4) Vowel narrowing before -yor
        #   (a) suffix already "yor" -- stem-final e/a narrows
        #   (b) suffix is "Iyor" with I in {ı,i,u,ü} -- stem-final vowel
        #       fuses with the suffix-initial vowel; if stem-final is e/a it
        #       also narrows.
        narrowed_via_fusion = False
        if surface.startswith("yor") and prev_surface:
            last_char = prev_surface[-1].lower()
            if last_char in self.NARROWING_MAP:
                parts[-1] = prev_surface[:-1] + self.NARROWING_MAP[last_char]
                prev_surface = parts[-1]
        elif (
            len(surface) >= 4
            and surface[0] in "ıiuü"
            and surface[1:4] == "yor"
            and prev_surface
            and prev_surface[-1].lower() in self.VOWELS
        ):
            last_char = prev_surface[-1].lower()
            if last_char in self.NARROWING_MAP:
                parts[-1] = prev_surface[:-1] + self.NARROWING_MAP[last_char]
                prev_surface = parts[-1]
            surface = surface[1:]
            narrowed_via_fusion = True

        # (5) Buffer-consonant insertion between two vowels
        if (
            not narrowed_via_fusion
            and surface
            and surface[0] in self.VOWELS
            and prev_surface
            and prev_surface[-1].lower() in self.VOWELS
        ):
            buf = self._choose_buffer_consonant(prev_surface, surface)
            surface = buf + surface

        return surface

    # helpers

    def _select_by_vowel_harmony(self, candidates, ctx):
        ctx = ctx.lower()
        is_back = ctx in self.BACK_VOWELS
        is_rounded = ctx in self.ROUNDED_VOWELS
        for require_rounding in (True, False):
            for cand in candidates:
                first = self._first_vowel(cand)
                if first == "":
                    continue
                cand_back = first in self.BACK_VOWELS
                cand_rounded = first in self.ROUNDED_VOWELS
                if cand_back != is_back:
                    continue
                if require_rounding and cand_rounded != is_rounded:
                    continue
                return cand
        return None

    @staticmethod
    def _first_vowel(text):
        for char in text.lower():
            if char in MorphologicalAnalyzer.VOWELS:
                return char
        return ""

    @staticmethod
    def _choose_buffer_consonant(prev_surface, surface):
        """Default buffer consonant is 'y'; use 's' for 3sg poss after vowels."""
        if surface in ("ı", "i", "u", "ü"):
            if prev_surface and prev_surface[-1].lower() in MorphologicalAnalyzer.VOWELS:
                return "s"
        return "y"
