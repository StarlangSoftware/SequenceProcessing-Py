from Corpus.Sentence import Sentence


class LabelledSentence(Sentence):

    __class_label: str

    def __init__(self, class_label):
        super().__init__()
        self.__class_label = class_label

    @property
    def class_label(self)->str:
        return self.__class_label
