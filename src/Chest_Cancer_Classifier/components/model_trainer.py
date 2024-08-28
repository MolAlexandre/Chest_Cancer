import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from keras.preprocessing.image import ImageDataGenerator
from Chest_Cancer_Classifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):

        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            validation_split = 0.2,
                                    
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            #zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        valid_datagen = ImageDataGenerator(rescale = 1./255,
                                        validation_split = 0.2)

        test_datagen  = ImageDataGenerator(rescale = 1./255)

        self.train_dataset  = train_datagen.flow_from_directory(
                    directory = os.path.join(self.config.training_data,'train'),
                    target_size = (224,224),
                    class_mode = 'categorical',
                    batch_size = self.config.params_batch_size
                    )
        self.val_dataset  = valid_datagen.flow_from_directory(
                    directory = os.path.join(self.config.training_data,'valid'),
                    target_size = (224,224),
                    class_mode = 'categorical',
                    batch_size = self.config.params_batch_size
                    )
        self.test_dataset  = test_datagen.flow_from_directory(
                    directory = os.path.join(self.config.training_data,'test'),
                    target_size = (224,224),
                    class_mode = 'categorical',
                    batch_size = self.config.params_batch_size
                    )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def train(self):
        self.steps_per_epoch = self.train_dataset.samples // self.train_dataset.batch_size
        self.validation_steps = self.val_dataset.samples // self.val_dataset.batch_size

        self.model.fit(
            self.train_dataset,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.val_dataset
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )