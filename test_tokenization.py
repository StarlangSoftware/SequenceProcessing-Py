from SequenceProcessing.Tokenization.MorphologicalAnalyzer import MorphologicalAnalyzer
from SequenceProcessing.Tokenization.TurkishTokenizer import TurkishTokenizer

# Dummy BPE class that simply returns the word itself as a fallback
class DummyBPE:
    def encode(self, text: str):
        # In a real scenario, this will return subword IDs.
        # For testing purposes, we just return the raw text.
        return [text]

def main():
    
    # Initialize the morphological analyzer with dummy paths 
    # (this will trigger the internal dummy data for testing)
    morph_tool = MorphologicalAnalyzer(roots_path="dummy", affixes_path="dummy")
    
    # Initialize the dummy BPE fallback
    bpe_tool = DummyBPE()
    
    # Build the hybrid tokenizer using dependency injection
    tokenizer = TurkishTokenizer(morph_analyzer=morph_tool, bpe_fallback=bpe_tool)
    
    # Define test cases based on our dummy dictionary
    # Keep in mind: roots={araba, gel, kitap}, suffixes={lar, ler, da, de}
    test_cases = [
        "arabalar",          # Dictionary hit: root + suffix
        "Kitap",             # Capitalization + root hit
        "telefon",           # Out-Of-Vocabulary: handled by BPE fallback
        "Araba, kitap!"      # Capitalization, punctuation, and multiple words
    ]
    
    for text in test_cases:
        print("-" * 40)
        print(f"Original: '{text}'")
        
        token_ids = tokenizer.encode(text)
        print(f"Tokens (IDs): {token_ids}")
        
        reconstructed = tokenizer.decode(token_ids)
        print(f"Reconstructed: '{reconstructed}'")
        
        # Validation: remove spaces and convert to lowercase for strict logical token comparison.
        # This is necessary because our current dummy dictionary lacks the special 
        # leading space characters ("_araba") mentioned in the paper.
        clean_original = text.replace(" ", "").lower()
        clean_reconstructed = reconstructed.replace(" ", "").lower()
        
        if clean_original == clean_reconstructed:
            print("Status: SUCCESS")
        else:
            print("Status: ❌ ERROR")

if __name__ == "__main__":
    main()