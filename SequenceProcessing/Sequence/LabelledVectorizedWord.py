from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector


class LabelledVectorizedWord(VectorizedWord):
    """
    Represents a vectorized word with an associated class label.
    """

    __class_label: str

    def constructor1(self, word: str, embedding: Vector, class_label: str) -> None:
        """
        First constructor: takes word, embedding, and class label.

        :param word: The word string.
        :param embedding: Vector embedding of the word.
        :param class_label: Class label of the word.
        """
        super().__init__(word, embedding)
        self.__class_label = class_label

    def constructor2(self, word: str, class_label: str) -> None:
        """
        Second constructor: creates a default embedding of size 300.

        :param word: The word string.
        :param class_label: Class label of the word.
        """
        super().__init__(word, Vector(300, 0))
        self.__class_label = class_label

    def __init__(self,
                 word: str,
                 embedding_or_label,
                 class_label: str = None):
        """
        Main constructor that dispatches to constructor1 or constructor2.

        :param word: The word string.
        :param embedding_or_label: Either embedding (Vector) or class label (str).
        :param class_label: Class label (if embedding is provided).
        """
        if isinstance(embedding_or_label, Vector):
            # corresponds to: (word, embedding, classLabel)
            self.constructor1(word, embedding_or_label, class_label)
        else:
            # corresponds to: (word, classLabel)
            self.constructor2(word, embedding_or_label)

    def getClassLabel(self) -> str:
        """
        Getter for class label.

        :return: Class label.
        """
        return self.__class_label

    def toString(self) -> str:
        """
        Returns string representation.

        :return: String representation.
        """
        return f"{super().__str__()} -> {self.__class_label}"