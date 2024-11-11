from Corpus.Corpus import Corpus
from Corpus.Sentence import Sentence
from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector
from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord


class SequenceCorpus(Corpus):

    def __init__(self, file_name: str):
        super().__init__()
        new_sentence = None

        try:
            input_file = open(file_name, "r", encoding="utf8")
            lines = input_file.readlines()
            for line in lines:
                items = line.strip().split()
                word = items[0]

                if word == "<S>":
                    if len(items) == 2:
                        new_sentence = LabelledSentence(items[1])
                    else:
                        new_sentence = Sentence()

                elif word == "</S>":
                    self.addSentence(new_sentence)

                else:
                    if len(items) == 2:
                        new_word = LabelledVectorizedWord(word, items[1])
                    else:
                        new_word = VectorizedWord(word, Vector(300, 0))

                    if new_sentence is not None:
                        new_sentence.addWord(new_word)
            input_file.close()

        except IOError:
            pass

    def getClassLabels(self) -> [str]:
        class_labels = []
        sentence_labelled = isinstance(self.sentences[0], LabelledSentence) if self.sentences else False

        for sentence in self.sentences:
            if sentence_labelled:
                if isinstance(sentence, LabelledSentence):
                    if sentence.getClassLabel not in class_labels:
                        class_labels.append(sentence.getClassLabel)
            else:
                for word in sentence.words:
                    if isinstance(word, LabelledVectorizedWord):
                        if word.getClassLabel not in class_labels:
                            class_labels.append(word.getClassLabel)

        return class_labels
