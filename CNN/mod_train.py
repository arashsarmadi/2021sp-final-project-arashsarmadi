import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cnn(img_rows, img_cols):
    num_classes = 4

    model = Sequential(
        [
            # 1st CONV-ReLU Layer
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(img_rows, img_cols, 3),
            ),
            BatchNormalization(),
            # 2nd CONV-ReLU Layer
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            # Max Pooling with Dropout
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            # 3rd set of CONV-ReLU Layers
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            # 4th Set of CONV-ReLU Layers
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            # Max Pooling with Dropout
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            # 5th Set of CONV-ReLU Layers
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            # Global Average Pooling
            GlobalAveragePooling2D(),
            # Final Dense Layer
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def call_backs(model_path):

    checkpoint = ModelCheckpoint(
        os.path.join(model_path, "retinal_cnn1.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    earlystop = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=3,
        verbose=1,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, min_delta=0.00001
    )

    return [earlystop, checkpoint, reduce_lr]


def train_model(
    train_data_dir, validation_data_dir, img_rows, img_cols, model_path, model_output
):

    nb_train_samples = 86
    nb_validation_samples = 32
    epochs = 1
    batch_size = 2

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical",
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical",
    )

    callbacks = call_backs(model_path)

    model = cnn(img_rows, img_cols)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
    )

    scores = model.evaluate(
        validation_generator, steps=nb_validation_samples // batch_size + 1, verbose=1
    )
    print("\nTest result: %.3f loss: %.3f" % (scores[1] * 100, scores[0]))

    model.save(model_output)
