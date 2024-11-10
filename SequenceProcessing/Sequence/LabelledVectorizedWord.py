from Dictionary.VectorizedWord import VectorizedWord
from Math.Vector import Vector


class LabelledVectorizedWord(VectorizedWord):

    def __init__(self, word, class_label, embedding=None):
        if embedding is None:
            # If no embedding is provided, create a Vector of size 300 initialized to 0
            embedding = Vector(300, 0)
        super().__init__(word, embedding)
        self._class_label = class_label

    @property
    def class_label(self):
        return self._class_label
