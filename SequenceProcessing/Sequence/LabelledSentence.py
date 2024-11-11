from Corpus.Sentence import Sentence


class LabelledSentence(Sentence):

    __class_label: str

    def __init__(self, class_label: str):
        super().__init__()
        self.__class_label = class_label

    @property
    def getClassLabel(self) -> str:
        return self.__class_label
