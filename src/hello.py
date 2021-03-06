from __future__ import print_function

import sys

# We'll also import a few standard python libraries
import numpy as np

# These are the droids you are looking for.
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2


def main():
    x = np.random.randn(2, 3).astype(np.float32)
    print("Generated X from numpy:\n{}".format(x))
    workspace.FeedBlob("X", x)

    print("Current blobs in the workspace: {}".format(workspace.Blobs()))
    print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))
    print("Fetched X:\n{}".format(workspace.FetchBlob("X")))

    print('Hello, caffe2!')
    print('Python version: %s' % sys.version)

if __name__ == '__main__':
    main()
