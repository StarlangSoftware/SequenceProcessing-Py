from Corpus.Corpus import Corpus
from Corpus.Sentence import Sentence
from Dictionary.VectorizedWord import VectorizedWord
from Util.FileUtils import FileUtils
from Math.Vector import Vector
from SequenceProcessing.Sequence.LabelledSentence import LabelledSentence
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord

class SequenceCorpus(Corpus):
    
    def __init__(self, file_name):
        super().__init__()
        new_sentence = None
        
        try:
            with FileUtils.get_input_stream(file_name) as fr:
                for line in fr:
                    items = line.strip().split()
                    word = items[0]
                    
                    if word == "<S>":
                        if len(items) == 2:
                            new_sentence = LabelledSentence(items[1])
                        else:
                            new_sentence = Sentence()
                    
                    elif word == "</S>":
                        self.add_sentence(new_sentence)
                    
                    else:
                        if len(items) == 2:
                            new_word = LabelledVectorizedWord(word, items[1])
                        else:
                            new_word = VectorizedWord(word, Vector(300, 0))
                        
                        if new_sentence is not None:
                            new_sentence.add_word(new_word)
        
        except IOError:
            pass

    def get_class_labels(self):
        class_labels = []
        sentence_labelled = isinstance(self.sentences[0], LabelledSentence) if self.sentences else False

        for sentence in self.sentences:
            if sentence_labelled:
                if sentence.get_class_label() not in class_labels:
                    class_labels.append(sentence.get_class_label())
            else:
                for word in sentence.words:
                    if isinstance(word, LabelledVectorizedWord):
                        if word.class_label not in class_labels:
                            class_labels.append(word.class_label)
        
        return class_labels
