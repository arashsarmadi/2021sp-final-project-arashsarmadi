import os
import random

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model


def load_image(path):
    img = image.load_img(path, target_size=(224, 224, 3))

    img_array = image.img_to_array(img)
    print(img_array.shape)

    img_array.shape = (1,) + img_array.shape
    print(img_array.shape)
    img_preprocessed = img_array / 255

    return img_preprocessed


def test_model(test_data_dir, output_path, model_path, features_name):

    random_class = random.choice(os.listdir(test_data_dir))
    random_image = random.choice(os.listdir(os.path.join(test_data_dir, random_class)))
    test_image_path = os.path.join(test_data_dir, random_class, random_image)

    image = load_image(test_image_path)

    np.save(os.path.join(output_path, "image"), image)

    model = load_model(model_path)

    # same as previous model but with an additional output
    cam_model = Model(
        inputs=model.input, outputs=(model.layers[-3].output, model.layers[-1].output)
    )
    cam_model.summary()

    print(test_data_dir)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir, class_mode="categorical", target_size=(224, 224)
    )

    #
    features, results = cam_model.predict(image)
    np.save(os.path.join(output_path, features_name), features)
    np.save(os.path.join(output_path, "results"), results)

    last_dense_layer = model.layers[-1]
    print(type(last_dense_layer))

    gap_weights_l = np.array(last_dense_layer.get_weights())

    np.save(os.path.join(output_path, "gap_weights_l"), gap_weights_l)

    # print("features shape: ", features.shape)
    # print("results shape", results.shape)
    # print(gap_weights_l)
    # print(type(gap_weights_l))
    # print(type(results))
