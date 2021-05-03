import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def show_cam(gap_weights_l, results, features, test_image, out_img):
    """displays the class activation map of a particular image"""

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

    # strongly classified (95% probability) images will be in green, else red
    # if results[image_index][prediction] > 0.95:
    #     cmap_str = 'Greens'
    # else:
    #     cmap_str = 'Reds'

    # overlay the cam output
    plt.imshow(cam, cmap="jet", alpha=0.5)

    # display the image
    plt.savefig(out_img)
