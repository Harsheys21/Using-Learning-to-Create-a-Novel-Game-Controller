from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential()
        self.model.add(Rescaling(1./255, input_shape=input_shape))  # Rescale input pixel values to [0, 1]
        self.model.add(layers.Conv2D(categories_count * 16, (3, 3), activation='relu'))  # Convolutional layer with 32 filters and 3x3 kernel
        self.model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer
        self.model.add(layers.Conv2D(categories_count * 8, (3, 3), activation='relu'))  # Another convolutional layer with 64 filters and 3x3 kernel
        self.model.add(layers.MaxPooling2D((2, 2)))  # Another max pooling layer
        self.model.add(layers.Conv2D(categories_count * 4, (3, 3), activation='relu'))  # Another convolutional layer with 128 filters and 3x3 kernel
        self.model.add(layers.Flatten())  # Flatten the convolutional output for fully connected layers
        self.model.add(layers.Dense(categories_count * 2, activation='relu'))  # Fully connected layer with 128 neurons
        self.model.add(layers.Dense(categories_count, activation='softmax'))  # Output layer with softmax activation
    

    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'],)
