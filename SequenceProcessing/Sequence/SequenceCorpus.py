from Corpus.Corpus import Corpus
from Corpus.Sentence import Sentence
from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector

from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord


class SequenceCorpus(Corpus):
    """
    Corpus class for sequence processing datasets.
    """

    def __init__(self, file_name: str):
        """
        Reads the given file and constructs the corpus sentence by sentence.

        If a sentence starts with <S> and has a label, a LabelledSentence is created.
        If a word line has two items, a LabelledVectorizedWord is created.
        Otherwise, a VectorizedWord with a zero vector of size 300 is created.

        :param file_name: File which will be read and parsed.
        """
        super().__init__()
        new_sentence: Sentence | None = None

        try:
            with open(file_name, "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()

                    if line == "":
                        continue

                    items = line.split(" ")
                    word = items[0]

                    if word == "<S>":
                        if len(items) == 2:
                            new_sentence = LabelledSentence(items[1])
                        else:
                            new_sentence = Sentence()
                    else:
                        if word == "</S>":
                            if new_sentence is not None:
                                self.addSentence(new_sentence)
                        else:
                            if len(items) == 2:
                                new_word = LabelledVectorizedWord(word, items[1])
                            else:
                                new_word = VectorizedWord(word, Vector(300, 0))

                            if new_sentence is not None:
                                new_sentence.addWord(new_word)
        except OSError:
            pass

    def getClassLabels(self) -> list[str]:
        """
        Returns the distinct class labels in the corpus.

        If sentences are labelled, sentence labels are returned.
        Otherwise, word labels are returned.

        :return: List of distinct class labels.
        """
        sentence_labelled = False
        class_labels = []

        if self.sentenceCount() > 0 and isinstance(self.sentences[0], LabelledSentence):
            sentence_labelled = True

        for i in range(self.sentenceCount()):
            if sentence_labelled:
                sentence = self.sentences[i]
                if sentence.getClassLabel() not in class_labels:
                    class_labels.append(sentence.getClassLabel())
            else:
                sentence = self.sentences[i]
                for j in range(sentence.wordCount()):
                    word = sentence.getWord(j)
                    if isinstance(word, LabelledVectorizedWord):
                        if word.getClassLabel() not in class_labels:
                            class_labels.append(word.getClassLabel())

        return class_labels