from __future__ import print_function
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import math
import os

import numpy as np
import deepchem as dc

# Only for debug!
np.random.seed(123)

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

#MNIST
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets





PATH_TO_LOCAL_LOGS = os.path.abspath(os.path.expanduser('~/Documents/mnist/logs'))
ROOT_PATH_TO_LOCAL_DATA = os.path.abspath(os.path.expanduser('~/Documents/data/mnist'))

try:
  job_name = os.environ['JOB_NAME']
  task_index = os.environ['TASK_INDEX']
  ps_hosts = os.environ['PS_HOSTS']
  worker_hosts = os.environ['WORKER_HOSTS']
except:
  job_name = None
  task_index = 0
  ps_hosts = None
  worker_hosts = None

flags = tf.app.flags

for param in os.environ.keys():
    print("%s: %s " % (param, os.environ[param]))

# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task the performs the variable "
                     "initialization")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
# Training related flags
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name = "malo/mnist", #all mounted repo
                        local_root = ROOT_PATH_TO_LOCAL_DATA,
                        local_repo = "mnist",
                        path = ''
                        ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                     get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to ClusterOne without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")

flags.DEFINE_integer("hidden1", 128,
                     "Number of units in the 1st hidden layer of the NN")
flags.DEFINE_integer("hidden2", 128,
                     "Number of units in the 2nd hidden layer of the NN")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

FLAGS = flags.FLAGS

print(FLAGS.data_dir)
print(FLAGS.log_dir)

def main(unused_argv):
    # Load Tox21 dataset
    n_features = 1024
    tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
    train_dataset, valid_dataset, test_dataset = tox21_datasets
    
    # Fit models
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    
    model = dc.models.RobustMultitaskClassifier(
        len(tox21_tasks),
        n_features,
        layer_sizes=[1000],
        dropouts=[.25],
        learning_rate=0.001,
        batch_size=50,
        use_queue=False)
    
    # Fit trained model
    model.fit(train_dataset, nb_epoch=1)
    model.save()
    
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)
    
    print("Train scores")
    print(train_scores)
    
    print("Validation scores")
    print(valid_scores)



def device_and_target():
  # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
  # Don't set a device.
  if FLAGS.job_name is None:
    print("Running single-machine training")
    return (None, "")

  # Otherwise we're running distributed TensorFlow.
  print("%s.%d  -- Running distributed training"%(FLAGS.job_name, FLAGS.task_index))
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
    raise ValueError("Must specify an explicit `ps_hosts`")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
    raise ValueError("Must specify an explicit `worker_hosts`")

  cluster_spec = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
  })
  server = tf.train.Server(
      cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  return (
      tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster_spec),
      server.target,
  )


def main_mnist(unused_argv):
  if FLAGS.log_dir is None or FLAGS.log_dir == "":
    raise ValueError("Must specify an explicit `log_dir`")
  if FLAGS.data_dir is None or FLAGS.data_dir == "":
    raise ValueError("Must specify an explicit `data_dir`")
  
  print('Printing Flags')
  print(str(FLAGS))
  
  print(FLAGS.__flags)
  device, target = device_and_target()
  with tf.device(device):
    images = tf.placeholder(tf.float32, [None, 784], name='image_input')
    labels = tf.placeholder(tf.float32, [None], name='label_input')
    data = read_data_sets(FLAGS.data_dir,
            one_hot=False,
            fake_data=False)
    logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels)
    loss = tf.Print(loss, [loss], message="Loss = ")
    train_op = mnist.training(loss, FLAGS.learning_rate)

  with tf.train.MonitoredTrainingSession(
      master=target,
      is_chief=(FLAGS.task_index == 0),
      checkpoint_dir=FLAGS.log_dir) as sess:
    while not sess.should_stop():
      xs, ys = data.train.next_batch(FLAGS.batch_size, fake_data=False)
      sess.run(train_op, feed_dict={images:xs, labels:ys})


if __name__ == "__main__":
  tf.app.run()#Runs the program with an optional 'main' function and 'argv' list.
  
  
  