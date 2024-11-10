from Corpus.Sentence import Sentence

class LabelledSentence(Sentence):
    
    def __init__(self, class_label):
        super().__init__()
        self._class_label = class_label

    @property
    def class_label(self):
        return self._class_label
