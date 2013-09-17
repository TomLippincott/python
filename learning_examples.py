import numpy
import argparse
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--instances", dest="instances", type=int, default=100)
parser.add_argument("-e", "--error", dest="error", type=float, default=.3)
options = parser.parse_args()

if __name__ == "__main__":
    inputs = numpy.random.random((options.instances))
    targets = numpy.sin(2 * numpy.pi * inputs)
    errors = numpy.random.normal(0.0, options.error, options.instances)
    pyplot.scatter(inputs, targets + errors, c='blue')

    # normA_mean = numpy.random.random((2))
    # normA_stddev = numpy.empty(shape=(2, 2))
    # normA_stddev[0, :] = numpy.random.random((2))
    # normA_stddev[1, 0] = normA_stddev[0, 1]
    # normA_stddev[1, 1] = normA_stddev[0, 0]

    # normB_mean = numpy.random.random((2))
    # normB_stddev = numpy.empty(shape=(2, 2))
    # normB_stddev[0, :] = numpy.random.random((2))
    # normB_stddev[1, 0] = normB_stddev[0, 1]
    # normB_stddev[1, 1] = normB_stddev[0, 0]

    # normC_mean = numpy.random.random((2))
    # normC_stddev = numpy.empty(shape=(2, 2))
    # normC_stddev[0, :] = numpy.random.random((2))
    # normC_stddev[1, 0] = normC_stddev[0, 1]
    # normC_stddev[1, 1] = normC_stddev[0, 0]

    # classA_data = numpy.random.multivariate_normal(normA_mean, normA_stddev, options.instances)
    # classB_data = numpy.random.multivariate_normal(normB_mean, normB_stddev, options.instances) + numpy.random.multivariate_normal(normC_mean, normC_stddev, options.instances)

    # pyplot.scatter(classA_data[:, 0], classA_data[:, 1], c='blue')
    # pyplot.scatter(classB_data[:, 0], classB_data[:, 1], c='red')
    pyplot.show()
