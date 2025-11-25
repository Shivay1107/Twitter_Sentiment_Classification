import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout
from keras.optimizers import Adam

class GRUModel:
    def __init__(self, vocab_size=500000,  max_len=100, embedding_dim = 256, Dropout_l = 0.1, units=256, step=0.1,gru_dropout=0.1,recurrent_dropout=0.1,
                 dense_units=128, dense_dropout=0.1, activation='relu'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.Dropout = Dropout_l
        self.units = units
        
        
    def build(self, hp):
        model = Sequential()

        # Embedding Layer
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len
            )
        )

        model.add(Dropout(self.dropout_rate))

        # GRU Layer
        model.add(
            GRU(
                self.gru_units,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                return_sequences=self.return_sequences
            )
        )

        # Output layer
        activation = 'softmax' if self.num_classes > 2 else 'sigmoid'
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'

        model.add(Dense(self.num_classes, activation=activation))

        model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])

        return model
    
    
    
    def train(self, X_train, y_train, batch_size, epochs, validation_data):
        return self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=1)