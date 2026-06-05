"""
Downloads the official TurkishTokenizer vocabulary files from HuggingFace.

Source: alibayram/turkish-mft-tokenizer
Paper:  Bayram et al., "Tokens with Meaning: A Hybrid Tokenization
        Approach for Turkish", 2025.

Files downloaded into SequenceProcessing/Tokenization/vocabs/:
    kokler.json       — 22,231 root surface forms → 20,000 canonical IDs
    ekler.json        — 177 suffix allomorphs → 72 affix IDs
    bpe_tokenler.json — BPE fallback subword vocabulary
"""

import os
import urllib.request

REPO_BASE = (
    "https://huggingface.co/alibayram/turkish-mft-tokenizer/resolve/main/vocabs/"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR  = os.path.join(
    SCRIPT_DIR,
    "SequenceProcessing", "Tokenization", "vocabs",
)

FILES = [
    "kokler.json",
    "ekler.json",
    "bpe_tokenler.json",
]


def download(filename: str, dest_dir: str) -> str:
    url  = REPO_BASE + filename
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  {filename:<22}  already present, skipping.")
        return dest
    print(f"  {filename:<22}  downloading … ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = os.path.getsize(dest) / 1024
    print(f"done ({size_kb:.0f} kB)")
    return dest


def main():
    os.makedirs(VOCAB_DIR, exist_ok=True)
    print(f"Saving to: {VOCAB_DIR}\n")
    for name in FILES:
        download(name, VOCAB_DIR)
    print("\nAll done. Vocab files are ready.")
    print(f"\nUsage in code:")
    print(f"    from SequenceProcessing.Tokenization.MorphologicalAnalyzer import MorphologicalAnalyzer")
    print(f"    morph = MorphologicalAnalyzer(")
    print(f"        roots_path  = r'{os.path.join(VOCAB_DIR, 'kokler.json')}',")
    print(f"        affixes_path= r'{os.path.join(VOCAB_DIR, 'ekler.json')}',")
    print(f"    )")


if __name__ == "__main__":
    main()
