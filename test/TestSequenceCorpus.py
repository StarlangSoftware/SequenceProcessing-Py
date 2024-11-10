import unittest
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus


class TestSequenceCorpus(unittest.TestCase):

    def test_sentence_count(self):
        self.corpus = SequenceCorpus("../mock_file.txt")
        self.assertEqual(self.corpus.sentenceCount(), 2)

    def test_class_labels(self):
        self.corpus = SequenceCorpus("../mock_file.txt")
        class_labels = self.corpus.getClassLabels()
        self.assertIn("noun", class_labels)
        self.assertIn("verb", class_labels)
        self.assertIn("adj", class_labels)
        self.assertEqual(len(class_labels), 3)

    def test_corpus01(self):
        self.corpus = SequenceCorpus("../Datasets/disambiguation-penn.txt")
        self.assertEqual(25957, self.corpus.sentenceCount())
        self.assertEqual(264930, self.corpus.numberOfWords())

    def test_corpus26(self):
        self.corpus = SequenceCorpus("../Datasets/sentiment-tourism.txt")
        self.assertEqual(19830, self.corpus.sentenceCount())
        self.assertEqual(91152, self.corpus.numberOfWords())


if __name__ == '__main__':
    unittest.main()
