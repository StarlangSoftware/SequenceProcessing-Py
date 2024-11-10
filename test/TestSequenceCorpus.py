import unittest
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus


class TestSequenceCorpus(unittest.TestCase):

    def setUp(self):
        self.corpus = SequenceCorpus("../mock_file.txt")

    def test_sentence_count(self):
        self.assertEqual(self.corpus.sentenceCount(), 2)

    def test_class_labels(self):
        class_labels = self.corpus.get_class_labels()
        self.assertIn("noun", class_labels)
        self.assertIn("verb", class_labels)
        self.assertIn("adj", class_labels)
        self.assertEqual(len(class_labels), 3)


if __name__ == '__main__':
    unittest.main()
