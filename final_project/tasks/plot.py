import os

from luigi import Task, Parameter
import numpy as np
from csci_utils.luigi.target import SuffixPreservingLocalTarget
from csci_utils.luigi.task import Requirement, TargetOutput, Requires

from .ML_env.ml import ConvNeuralTest
from ..plot_model import show_cam


class PlotResults(Task):
    """Luigi task that uses prediction and saves plots"""

    __version__ = "1.0"

    data_source = Parameter()
    output_pred = Parameter()
    output_model = Parameter()
    train_loc = Parameter()

    requires = Requires()
    req_1 = Requirement(ConvNeuralTest)
    LOCAL_ROOT = os.path.join(os.getcwd(), "data")

    path = os.path.join(LOCAL_ROOT, "{task.__class__.__name__}-{salt}.png")

    output = TargetOutput(
        file_pattern=path, target_class=SuffixPreservingLocalTarget, ext=""
    )

    def run(self):
        features = np.load(self.req_1.output().path)
        test_path = self.req_1.output().path.rstrip("features.npy")
        results = np.load(test_path + "results.npy")
        gap_weights_l = np.load(test_path + "gap_weights_l.npy", allow_pickle=True)
        test_image = np.load(test_path + "image.npy")

        show_cam(gap_weights_l, results, features, test_image, self.output().path)
