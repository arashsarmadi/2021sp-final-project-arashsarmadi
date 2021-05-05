import argparse

from final_project.tasks.plot import PlotResults
from luigi import build, WrapperTask


parser = argparse.ArgumentParser("Select Input data")
parser.add_argument("-d", "--data", default="OCTReduced")
parser.add_argument("-p", "--output_pred", default="features")
parser.add_argument("-o", "--output_model", default="retinal_cnn.h5")
parser.add_argument("-l", "--train_loc", default="local")


def main(args=None):
    args = parser.parse_args(args=args)

    class MainTask(WrapperTask):
        """Luigi task that kicks off the program"""

        data_source = args.data
        output_model = args.output_model
        output_pred = args.output_pred
        train_loc = args.train_loc

        def requires(self):
            return [self.clone(PlotResults)]

    build([MainTask()], local_scheduler=True)
