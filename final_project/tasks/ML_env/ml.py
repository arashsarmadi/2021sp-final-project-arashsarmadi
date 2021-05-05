import os

from csci_utils.luigi.target import SuffixPreservingLocalTarget
from ..data import LocalImageReduced

from luigi import Parameter
from luigi.contrib.external_program import (
    ExternalPythonProgramTask,
    ExternalProgramTask,
)

from csci_utils.luigi.task import TargetOutput, Requires


class ConvNeural(ExternalPythonProgramTask):
    """
    Luigi Task to run a shell script that builds and activates a new venv
    """

    __version__ = "1.0"

    data_source = Parameter()
    output_pred = Parameter()
    output_model = Parameter()

    requires = Requires()

    LOCAL_ROOT = os.path.join(os.getcwd(), "data")
    LOCAL_IMAGE = os.path.join(LOCAL_ROOT, "OCTReduced")

    extra_pythonpath = os.getcwd()
    virtualenv = os.path.join(extra_pythonpath, "final_project/tasks/ML_env/.venv")

    def program_args(self):
        "executes a shell script where a new venv is created for ML envirionment and the train or test code is run"
        data_path = os.path.join(self.LOCAL_ROOT, self.data_source)
        ml_path = "final_project/tasks/ML_env"
        model_path = self.temp_output_path
        if self.__class__.__name__ == "ConvNeuralTest":
            model_path = self.input().path

        return [
            "./external_script.sh",
            data_path,
            self.temp_output_path,
            ml_path,
            self.virtualenv,
            self.action,
            model_path,
        ]

    def run(self):
        with self.output().temporary_path() as self.temp_output_path:
            super().run()


class ConvNeuralTrain(ConvNeural):
    """Luigi task for training the model"""

    __version__ = "1.0"

    action = "train"

    def requires(self):
        return self.clone(LocalImageReduced)

    path = os.path.join(
        ConvNeural.LOCAL_ROOT, "{task.__class__.__name__}-{salt}/{task.output_model}"
    )

    output = TargetOutput(
        file_pattern=path, target_class=SuffixPreservingLocalTarget, ext=""
    )


class ConvNeuralCluster(ExternalProgramTask):
    """Luigi task for training the model on cluster"""

    __version__ = "1.0"

    data_source = Parameter()
    output_pred = Parameter()
    output_model = Parameter()

    LOCAL_ROOT = os.path.join(os.getcwd(), "data")

    path = os.path.join(
        LOCAL_ROOT, "{task.__class__.__name__}-{salt}/{task.output_model}"
    )

    output = TargetOutput(
        file_pattern=path, target_class=SuffixPreservingLocalTarget, ext=""
    )

    def program_args(self):

        cluster_path = os.getenv("CLUSTER_PATH")
        cluster_pass = os.getenv("CLUSTER_PASS")

        return [
            "sshpass",
            "-p",
            cluster_pass,
            "scp",
            cluster_path,
            self.temp_output_path,
        ]

    def run(self):
        with self.output().temporary_path() as self.temp_output_path:
            super().run()


class ConvNeuralTest(ConvNeural):
    """Luigi task that activates ML venv through its parent class and does the model testing"""

    __version__ = "1.0"
    action = "test"
    train_loc = Parameter()

    def requires(self):
        if self.train_loc == "cluster":
            return self.clone(ConvNeuralCluster)
        else:
            return self.clone(ConvNeuralTrain)

    path = os.path.join(
        ConvNeural.LOCAL_ROOT, "{task.__class__.__name__}-{salt}/{task.output_pred}.npy"
    )
    output = TargetOutput(
        file_pattern=path, target_class=SuffixPreservingLocalTarget, ext=""
    )
