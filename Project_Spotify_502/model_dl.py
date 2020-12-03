import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def generator(directory_train, directory_val): 
    generator = ImageDataGenerator()
    train = generator.flow_from_directory(directory = directory_train, \
                                        target_size = (128, 2582), color_mode = 'grayscale', batch_size = 8)

    val = generator.flow_from_directory(directory = directory_val, \
                                        target_size = (128, 2582), color_mode = 'grayscale', batch_size = 8)
    return train, val

def initiate_model(): 
    ### First convolution & max-pooling
    model = Sequential()
    model.add(layers.Conv2D(8, (4,4), activation = 'relu', input_shape=(128, 2582, 1)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One fully connected
    model.add(layers.Dense(10, activation = 'relu'))

    ### Last layer (let's say a classification with 10 output)
    model.add(layers.Dense(8, activation='softmax'))
    
    ### Model compilation
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

if __name__ == '__main__': 
    model = initiate_model()
    train, val = generator('../raw_data/generated_spectrograms/train/', '../raw_data/generated_spectrograms/val/')
    model.fit_generator(train, epochs=10, validation_data = val)