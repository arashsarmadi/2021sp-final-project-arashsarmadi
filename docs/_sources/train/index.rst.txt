===================================
Training The Model
===================================


Local Train
===========================

This path on the graph is used for train code development purposes only. Local Model Training starts with a Luigi
external target to output the OCT images. Because there are around 80,000 images in the train set, another Luigi task
is called to take a small random sample of the images from different classes. Also, composition is used for this task
to define the task requirements and target output

.. code-block::

    class LocalImageReduced(Task):
        """Luigi external task that returns a target for a small subset of train data"""

        __version__ = "1.0"

        requires = Requires()
        req_1 = Requirement(LocalImage)
        LOCAL_ROOT = os.path.join(os.getcwd(), "data")
        LOCAL_IMAGE = os.path.join(LOCAL_ROOT, "OCTReduced")

        output = TargetOutput(
            file_pattern=LOCAL_IMAGE, target_class=SuffixPreservingLocalTarget, ext=""
        )

        def run(self):
            """
            This function goes through the train/test directories and the subdirectories inside for each class and takes
            a small sample of images and copies them into a new directory.
            """
            rootdir = self.req_1.output().path
            newpath = self.output().path
            for src_dir, dirs, files in os.walk(rootdir):
                dst_dir = src_dir.replace(rootdir, newpath, 1)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                counter = 0
                for file_ in files:
                    src_file = os.path.join(src_dir, file_)
                    shutil.copy(src_file, dst_dir)
                    counter += 1
                    if "train" in dst_dir and counter > 50:
                        break
                    elif "test" in dst_dir and counter > 10:
                        break


Next, a class of ExternalPythonProgramTask is built where it takes some parameters and it runs a shell script to create
a new virtual environment in the provided directory. This class will be used for both local training and testing as it
will be shown in the next sections

.. code-block::

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

Below is the shell script that is called above. Note that PIPENV_VENV_IN_PROJECT=1 will result in a venv that is created
in locally in the provided directory where a second pipfile is availbe with ML packages

.. code-block::

    #!/bin/sh
    data=${1}
    out=${2}
    ml_path=${3}
    venv_path=${4}
    action=${5}
    model=${6}
    cd ${ml_path}
    PIPENV_VENV_IN_PROJECT=1 pipenv install
    . ${venv_path}/bin/activate
    python -m CNN -d ${data} -o ${out} -a ${action} -l ${model}


Now for the purpose of training, below child class of ConvNeural is called by Luigi where it triggers the train and
saves a salted trained model to address the data-dependency hell.

.. code-block::

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


Cluster Train
===========================
.. image:: Server-Room.jpg

At this point, the following tasks need to be done on the cluster machine manually:

- Resource allocation request

- Activate the TF environment

- Run CNN package to build the model

Once the model is built, below Luigi tasks will connect to the cluster machine through SSH and brings a salted version
of the model back to the local machine for the next steps

.. code-block::

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