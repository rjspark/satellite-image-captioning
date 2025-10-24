"""
Vocabulary class for mapping between words and indices
This must match the vocabulary structure used during training
"""

class Vocabulary:
    """
    Vocabulary class for word-to-index and index-to-word mapping
    Handles both word2idx/idx2word and stoi/itos naming conventions
    """
    def __init__(self, freq_threshold=5):
        # Initialize with default special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        
    def add_word(self, word):
        """Add a word to the vocabulary"""
        if word not in self.stoi:
            idx = len(self.itos)
            self.stoi[word] = idx
            self.itos[idx] = word
            
    def __call__(self, word):
        """Get index for a word"""
        if word not in self.stoi:
            return self.stoi['<UNK>']
        return self.stoi[word]
    
    def __len__(self):
        """Get vocabulary size"""
        return len(self.itos)
    
    @property
    def word2idx(self):
        """Alias for stoi (string to index)"""
        return self.stoi
    
    @property
    def idx2word(self):
        """Alias for itos (index to string)"""
        return self.itos
