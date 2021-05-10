===================================
Testing The Model
===================================

Model testing is performed on the local machine using test images and the available trained model from the upstream
local or cluster train tasks. This task is also performed in the ML environment so a child of ConvNeural is constructed
as below.

.. code-block::

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

This will create a salted prediction output that will be used in the downstream plotting task
