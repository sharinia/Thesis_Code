# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple transfer learning with Inception v3 or Mobilenet models.

With support for TensorBoard.

This example shows how to take a Inception v3 or Mobilenet model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector (1001-dimensional for
Mobilenet) for each image. We train a softmax layer on top of this
representation. Assuming the softmax layer contains N labels, this corresponds
to learning N + 2048*N (or 1001*N)  model parameters corresponding to the
learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.

By default this script will use the high accuracy, but comparatively large and
slow Inception v3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--architecture` flag with a
Mobilenet model. For example:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos --architecture mobilenet_1.0_224
```

There are 32 different Mobilenet models to choose from, with a variety of file
size and latency options. The first number can be '1.0', '0.75', '0.50', or
'0.25' to control the size, and the second controls the input image size, either
'224', '192', '160', or '128', with smaller sizes running faster. See
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
for more information on Mobilenet.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import io
import sys
import tarfile
import BytesIO

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.python.framework import ops
from tensorflow.python.ops import confusion_matrix


import textwrap
import re
import itertools
import matplotlib


class SaverHook(tf.train.SessionRunHook):
    """
    Saves a confusion matrix as a Summary so that it can be shown in tensorboard
    """
    
    def __init__(self, labels, confusion_matrix_tensor_name, summary_writer):
        """Initializes a `SaveConfusionMatrixHook`.
            
            :param labels: Iterable of String containing the labels to print for each
            row/column in the confusion matrix.
            :param confusion_matrix_tensor_name: The name of the tensor containing the confusion
            matrix
            :param summary_writer: The summary writer that will save the summary
        """
        self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
        self.labels = labels
        self._summary_writer = summary_writer
    
    def end(self, session):
        cm = tf.get_default_graph().get_tensor_by_name(
                self.confusion_matrix_tensor_name + ':0').eval(session=session).astype(int)
        globalStep = tf.train.get_global_step().eval(session=session)
        figure = self._plot_confusion_matrix(cm)
        summary = self._figure_to_summary(figure)
        self._summary_writer.add_summary(summary, globalStep)

    def _figure_to_summary(self, fig):
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
        as a image summary.
        """
            
        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
                
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
                
        # get JPG data from the figure
        jpg_buffer = io.BytesIO()
        fig.canvas.print_jpg(jpg_buffer)
        jpg_encoded = jpg_buffer.getvalue()
        jpg_buffer.close()
                
        summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                        encoded_image_string=jpg_encoded)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.confusion_matrix_tensor_name,image=summary_image)])
                                                     return summary

    def _plot_confusion_matrix(self, cm):
        '''
        :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
        `   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
        '''
        numClasses = len(self.labels)
            
        fig = matplotlib.figure.Figure(figsize=(numClasses, numClasses), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')
            
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]
            
        tick_marks = np.arange(len(classes))
            
        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
            
        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()
            
        for i, j in itertools.product(range(numClasses), range(numClasses)):
            ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center", verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig
