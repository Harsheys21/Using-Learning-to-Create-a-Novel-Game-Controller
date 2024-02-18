from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential()
        self.model.add(Rescaling(1./255, input_shape=input_shape))
        
        self.model.add(layers.Conv2D(4, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(16, activation='relu'))
        
        self.model.add(layers.Dense(16, activation='relu'))
        
        self.model.add(layers.Dense(categories_count, activation='softmax'))

    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(optimizer=RMSprop(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
