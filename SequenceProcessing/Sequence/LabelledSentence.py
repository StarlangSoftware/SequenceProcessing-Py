from Corpus.Sentence import Sentence


class LabelledSentence(Sentence):
    """
    Represents a sentence with an associated class label.
    """

    __class_label: str

    def __init__(self, class_label: str):
        """
        Constructor for LabelledSentence.

        :param class_label: The label associated with the sentence.
        """
        super().__init__()
        self.__class_label = class_label

    def getClassLabel(self) -> str:
        """
        Getter for class label.

        :return: Class label of the sentence.
        """
        return self.__class_label

    def toString(self) -> str:
        """
        Returns string representation of the labelled sentence.

        :return: String representation.
        """
        return f"{super().__str__()} -> {self.__class_label}"