import unittest
from Math.Vector import Vector
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord


class TestLabelledVectorizedWord(unittest.TestCase):

    def test_initialization_with_embedding(self):
        embedding = Vector(300, 1.0)
        word = LabelledVectorizedWord("example", "noun", embedding)
        self.assertEqual(word.name, "example")
        self.assertEqual(word.getClassLabel, "noun")
        self.assertEqual(word.getVector(), embedding)

    def test_initialization_without_embedding(self):
        word = LabelledVectorizedWord("example", "noun")
        self.assertEqual(word.name, "example")
        self.assertEqual(word.getClassLabel, "noun")
        self.assertEqual(word.getVector().size(), 300)


if __name__ == '__main__':
    unittest.main()
