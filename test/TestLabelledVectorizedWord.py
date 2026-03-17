import unittest

from Math.Vector import Vector
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord


class LabelledVectorizedWordTest(unittest.TestCase):

    def testConstructorWithEmbedding(self):
        """
        Tests constructor with embedding.
        """
        vec = Vector(3, 1.0)
        word = LabelledVectorizedWord("hello", vec, "positive")

        self.assertEqual("positive", word.getClassLabel())

    def testConstructorWithoutEmbedding(self):
        """
        Tests constructor with default embedding.
        """
        word = LabelledVectorizedWord("hello", "negative")

        self.assertEqual("negative", word.getClassLabel())


if __name__ == "__main__":
    unittest.main()