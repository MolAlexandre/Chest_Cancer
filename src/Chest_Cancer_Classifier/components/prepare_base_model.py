import urllib.request as request
from pathlib import Path
import tensorflow as tf
from Chest_Cancer_Classifier.entity.config_entity import PrepareBaseModelConfig
from keras.layers import Dense, Flatten, Dropout

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    #take the VGG16 model
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    

    #complete model add with 2 layers
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        else :
            model.trainable = False

        flatten_in = Flatten()(model.output)
        x = Dropout(0.5)(flatten_in)
        x = Dense(units=128, activation="relu")(x)
        x = Dense(units=128, activation="relu")(x)
        x = Dropout(0.3)(x)
        prediction = Dense(units=classes, activation="softmax")(x)

        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    