import unittest
from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord

class TestLabelledVectorizedWord(unittest.TestCase):

    def test_initialization_with_embedding(self):
        embedding = Vector(300, 1.0)
        word = LabelledVectorizedWord("example", "noun", embedding)
        self.assertEqual(word.word, "example")
        self.assertEqual(word.class_label, "noun")
        self.assertEqual(word.embedding, embedding)

    def test_initialization_without_embedding(self):
        word = LabelledVectorizedWord("example", "noun")
        self.assertEqual(word.word, "example")
        self.assertEqual(word.class_label, "noun")
        self.assertEqual(word.embedding.size, 300)
        self.assertTrue(all(value == 0 for value in word.embedding.data))

if __name__ == '__main__':
    unittest.main()
