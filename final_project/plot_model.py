import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def show_cam(gap_weights_l, results, features, test_image, out_img):
    """
    A generic function that receives model prediction and generates a simple report and plot

    :param gap_weights_l: weights from model training
    :param results: model prediction output for each class
    :param features: model prediction array
    :param test_image: The image that was used as prediction input
    :param out_img: path to the output plot
    :return: saves plot figure
    """

    gap_weights = gap_weights_l[0]

    idx = 0
    features_for_img = features[idx, :, :, :]

    features_for_img_scaled = scipy.ndimage.zoom(
        features_for_img, (512 / 124, 512 / 124, 1), order=2
    )

    prediction_class_id = np.argmax(results[idx])
    gap_weights_for_one_class = gap_weights[:, prediction_class_id]

    cam = np.dot(features_for_img_scaled, gap_weights_for_one_class)

    print(
        "Predicted Class = "
        + str(prediction_class_id)
        + ", Probability = "
        + str(results[idx][prediction_class_id])
    )

    # show the upsampled image
    plt.imshow(np.squeeze(test_image), cmap="gray", alpha=0.2)

    # overlay the cam output
    plt.imshow(cam, cmap="jet", alpha=0.5)

    # display the image
    plt.savefig(out_img)
