import unittest

from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence


class LabelledSentenceTest(unittest.TestCase):

    def testLabel(self):
        sentence = LabelledSentence("positive")

        self.assertEqual("positive", sentence.getClassLabel())


if __name__ == "__main__":
    unittest.main()