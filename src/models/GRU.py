import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.optimizers import Adam

class GRUModel:
    def __init__(self, vocab_size=500000, max_len=100, embedding_dim=[64,128,256],
                 num_classes=2, recurrent_dropout=0.1, loss=None):
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.recurrent_dropout = recurrent_dropout
        self.loss = loss

    def build(self, hp):
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=hp.Choice('embedding_dim', self.embedding_dim),
            input_length=self.max_len
        ))

        # GRU layers
        num_layers = hp.Int('num_gru_layers', 1, 3)
        
        
        for i in range(num_layers):
            return_seq = True if i < num_layers - 1 else False
            model.add(GRU(
                units=hp.Int(f'gru_units_{i+1}', min_value=32, max_value=128, step=32),
                return_sequences=return_seq,
                dropout=hp.Float(f'dropout_{i+1}', 0.1, 0.5, step=0.1),
                recurrent_dropout=self.recurrent_dropout
            ))

        # Output layer
        activation = 'softmax' if self.num_classes > 2 else 'sigmoid'
        loss = self.loss or ('categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy')
        model.add(Dense(self.num_classes, activation=activation))
        
        # Compile
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=loss,
            metrics=['accuracy']
        )
        return model
