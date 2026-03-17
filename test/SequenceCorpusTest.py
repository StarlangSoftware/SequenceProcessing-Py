import unittest

from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus


class SequenceCorpusTest(unittest.TestCase):

    def test_corpus_01(self):
        corpus = SequenceCorpus("Datasets\postag-penn.txt")
        self.assertEqual(25957, corpus.sentenceCount())
        self.assertEqual(264930, corpus.numberOfWords())

    def test_corpus_02(self):
        corpus = SequenceCorpus("Datasets\postag-atis-en.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(61879, corpus.numberOfWords())

    def test_corpus_03(self):
        corpus = SequenceCorpus("Datasets\slot-atis-en.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(61879, corpus.numberOfWords())

    def test_corpus_04(self):
        corpus = SequenceCorpus("Datasets\slot-atis-tr.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(45875, corpus.numberOfWords())

    def test_corpus_05(self):
        corpus = SequenceCorpus("Datasets\disambiguation-atis.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(45875, corpus.numberOfWords())

    def test_corpus_06(self):
        corpus = SequenceCorpus("Datasets\metamorpheme-atis.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(45875, corpus.numberOfWords())

    def test_corpus_07(self):
        corpus = SequenceCorpus("Datasets\postag-atis-tr.txt")
        self.assertEqual(5432, corpus.sentenceCount())
        self.assertEqual(45875, corpus.numberOfWords())

    def test_corpus_08(self):
        corpus = SequenceCorpus("Datasets\metamorpheme-penn.txt")
        self.assertEqual(25957, corpus.sentenceCount())
        self.assertEqual(264930, corpus.numberOfWords())

    def test_corpus_09(self):
        corpus = SequenceCorpus("Datasets\ner-penn.txt")
        self.assertEqual(19118, corpus.sentenceCount())
        self.assertEqual(168654, corpus.numberOfWords())

    def test_corpus_10(self):
        corpus = SequenceCorpus("Datasets\postag-penn.txt")
        self.assertEqual(25957, corpus.sentenceCount())
        self.assertEqual(264930, corpus.numberOfWords())

    def test_corpus_11(self):
        corpus = SequenceCorpus("Datasets\semanticrolelabeling-penn.txt")
        self.assertEqual(19118, corpus.sentenceCount())
        self.assertEqual(168654, corpus.numberOfWords())

    def test_corpus_12(self):
        corpus = SequenceCorpus("Datasets\semantics-penn.txt")
        self.assertEqual(19118, corpus.sentenceCount())
        self.assertEqual(168654, corpus.numberOfWords())

    def test_corpus_13(self):
        corpus = SequenceCorpus("Datasets\shallowparse-penn.txt")
        self.assertEqual(9557, corpus.sentenceCount())
        self.assertEqual(87279, corpus.numberOfWords())

    def test_corpus_14(self):
        corpus = SequenceCorpus("Datasets\disambiguation-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())

    def test_corpus_15(self):
        corpus = SequenceCorpus("Datasets\metamorpheme-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())

    def test_corpus_16(self):
        corpus = SequenceCorpus("Datasets\postag-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())

    def test_corpus_17(self):
        corpus = SequenceCorpus("Datasets\semantics-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())

    def test_corpus_18(self):
        corpus = SequenceCorpus("Datasets\shallowparse-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())

    def test_corpus_19(self):
        corpus = SequenceCorpus("Datasets\disambiguation-kenet.txt")
        self.assertEqual(18687, corpus.sentenceCount())
        self.assertEqual(178658, corpus.numberOfWords())

    def test_corpus_20(self):
        corpus = SequenceCorpus("Datasets\metamorpheme-kenet.txt")
        self.assertEqual(18687, corpus.sentenceCount())
        self.assertEqual(178658, corpus.numberOfWords())

    def test_corpus_21(self):
        corpus = SequenceCorpus("Datasets\postag-kenet.txt")
        self.assertEqual(18687, corpus.sentenceCount())
        self.assertEqual(178658, corpus.numberOfWords())

    def test_corpus_22(self):
        corpus = SequenceCorpus("Datasets\disambiguation-framenet.txt")
        self.assertEqual(2704, corpus.sentenceCount())
        self.assertEqual(19286, corpus.numberOfWords())

    def test_corpus_23(self):
        corpus = SequenceCorpus("Datasets\metamorpheme-framenet.txt")
        self.assertEqual(2704, corpus.sentenceCount())
        self.assertEqual(19286, corpus.numberOfWords())

    def test_corpus_24(self):
        corpus = SequenceCorpus("Datasets\postag-framenet.txt")
        self.assertEqual(2704, corpus.sentenceCount())
        self.assertEqual(19286, corpus.numberOfWords())

    def test_corpus_25(self):
        corpus = SequenceCorpus("Datasets\semanticrolelabeling-framenet.txt")
        self.assertEqual(2704, corpus.sentenceCount())
        self.assertEqual(19286, corpus.numberOfWords())

    def test_corpus_26(self):
        corpus = SequenceCorpus("Datasets\sentiment-tourism.txt")
        self.assertEqual(19830, corpus.sentenceCount())
        self.assertEqual(91152, corpus.numberOfWords())


if __name__ == "__main__":
    unittest.main()