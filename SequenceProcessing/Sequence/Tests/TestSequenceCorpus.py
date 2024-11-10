import unittest
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus
from Corpus.Sentence import Sentence
from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence

class TestSequenceCorpus(unittest.TestCase):

    def setUp(self):
        # Mocking a file-like object using StringIO for testing
        from io import StringIO
        from Util.FileUtils import FileUtils

        file_content = "<S> positive\nword1 noun\nword2 verb\n</S>\n<S>\nword3 adj\n</S>\n"
        FileUtils.get_input_stream = lambda _: StringIO(file_content)
        
        self.corpus = SequenceCorpus("mock_file.txt")

    def test_sentence_count(self):
        self.assertEqual(self.corpus.sentence_count(), 2)

    def test_class_labels(self):
        class_labels = self.corpus.get_class_labels()
        self.assertIn("positive", class_labels)
        self.assertIn("noun", class_labels)
        self.assertIn("verb", class_labels)
        self.assertIn("adj", class_labels)
        self.assertEqual(len(class_labels), 4)

if __name__ == '__main__':
    unittest.main()
