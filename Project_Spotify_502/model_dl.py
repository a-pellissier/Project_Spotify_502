import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import joblib
import os

from PIL import ImageFile

from Project_Spotify_502.custom_generator import flow_from_google_storage, GoogleStorageIterator

ImageFile.LOAD_TRUNCATED_IMAGES = True

BUCKET_NAME = 'project_spotify_pellissier'
PROJECT = 'optimal-jigsaw-296709'
TRAIN_DIR = 'generated_spectrograms_small/png/train/'
VAL_DIR = 'generated_spectrograms_small/png/val/'


def generator():
    generator = ImageDataGenerator()
    train = flow_from_google_storage(imageDataGen=generator, project=PROJECT,\
        bucket=BUCKET_NAME, directory=TRAIN_DIR,\
        target_size = (128, 2582), color_mode = 'grayscale', batch_size = 64)

    val = flow_from_google_storage(imageDataGen=generator, project=PROJECT,\
        bucket=BUCKET_NAME, directory=VAL_DIR,\
        target_size = (128, 2582), color_mode = 'grayscale', batch_size = 64)
    return train, val

def initiate_model():
    ### First convolution & max-pooling
    model = Sequential()
    model.add(layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(128, 2582, 1)))
    model.add(layers.MaxPool2D(pool_size=(2,4)))


    model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
    model.add(layers.MaxPool2D(pool_size=(2,4)))

    ### Flattening
    model.add(layers.Flatten())

    ### One fully connected
    model.add(layers.Dense(10, activation = 'relu'))

    ### Last layer (let's say a classification with 10 output)
    model.add(layers.Dense(8, activation='softmax'))

    ### Model compilation
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

def save_model(reg, client, model_name = 'model'):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    local_path = os.path.join('../saved_pipes',f'{model_name}.joblib')

    #local saving
    joblib.dump(reg, local_path)
    print(f"saved {model_name}.joblib locally")

    # gcp saving
    storage_location = f'models/{model_name}.joblib'

    bucket = client.bucket(BUCKET_NAME)
    storage_location = bucket.blob(storage_location)
    storage_location.upload_from_filename(local_path)
    print(f"uploaded model.joblib to gcp cloud storage under \n => {storage_location}")



if __name__ == '__main__':
    model = initiate_model()
    X_train, X_val = generator()
    model.fit(X_train, epochs=1, validation_data=X_val)
    save_model(model, X_train.storage_client)
