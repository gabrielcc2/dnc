#Based on Pavlo Shevchenko: https://github.com/pshevche/drl-frameworks
import tensorflow as tf

class Tensorboard:
    """
    Custom Tensorboard to post experiment summaries.
    """

    def __init__(self, logdir):
        self.writer = tf.compat.v1.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_summary(self, average_loss, max_edges, max_questions, iteration):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='DNC-OneHop/AverageLoss',
                             simple_value=average_loss),
            tf.Summary.Value(tag='DNC-OneHop/MaxEdges',
                             simple_value=max_edges),
            tf.Summary.Value(tag='DNC-OneHop/MaxQuestions',
                             simple_value=max_questions)])

        self.writer.add_summary(summary, iteration)

        self.writer.flush()