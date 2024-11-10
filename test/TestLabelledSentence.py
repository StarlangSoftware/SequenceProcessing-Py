import unittest
from Corpus.Sentence import Sentence
from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence


class TestLabelledSentence(unittest.TestCase):

    def test_class_label(self):
        sentence = LabelledSentence("positive")
        self.assertEqual(sentence.getClassLabel, "positive")
        self.assertIsInstance(sentence, Sentence)


if __name__ == '__main__':
    unittest.main()
