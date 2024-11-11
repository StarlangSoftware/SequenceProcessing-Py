from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector


class LabelledVectorizedWord(VectorizedWord):

    __class_label: str

    def __init__(self, word: str, class_label: str, embedding: Vector=None):
        if embedding is None:
            # If no embedding is provided, create a Vector of size 300 initialized to 0
            embedding = Vector(300, 0)
        super().__init__(word, embedding)
        self.__class_label = class_label

    @property
    def getClassLabel(self) -> str:
        return self.__class_label
