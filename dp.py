import numpy
import argparse
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=10, help="number of sampling iterations")
#parser.add_argument("--min", dest="min", type=int, default=2)
#parser.add_argument("--max", dest="max", type=int, default=20)
parser.add_argument("-c", "--count", dest="count", type=int, default=100, help="number of instances")
parser.add_argument("-l", "--latent", dest="latent", type=int, default=10, help="number of classes")
#parser.add_argument("-v", "--vocab", dest="vocab", type=int, default=3)
#parser.add_argument("-l", "--length", dest="length", type=int, default=3)
parser.add_argument("-a", "--alpha", dest="alpha", type=float, default=.02)
parser.add_argument("-b", "--beta", dest="beta", type=float, default=.02)
#parser.add_argument("-g", "--gamma", dest="gamma", type=float, default=.02)
parser.add_argument("-d", "--debug", dest="debug", type=int, default=20)
options = parser.parse_args()


logging.basicConfig(level = options.debug, format="%(message)s", filemode="w")

# draw mixing parameters
mix_params = numpy.random.mtrand.dirichlet([1 for i in range(options.latent)])

# assign instances a single feature, their true class
values = numpy.asarray(sum([[x for i in range(y)] for x, y in enumerate(numpy.random.multinomial(options.count, mix_params))], []), dtype=int)

# initialize state with a single class (0)
assignments = numpy.asarray([0 for x in values], dtype=int)

# allocate enough space for the maximum number of classes (i.e. the number of instances)
class_by_value = numpy.zeros(shape=(options.count, options.latent), dtype=int)

# fill in table of classes by features
for v, c in zip(values, assignments):
    class_by_value[c, v] += 1

#logging.info("%s", values)
#logging.info("%s", assignments)
#logging.info("%s", class_by_value)


for iteration in range(options.iterations):
    logging.debug("Iteration: %d", iteration)
    for i in range(len(assignments)):
        # how many classes are there currently?
        classes = len(set(assignments))

        logging.debug("Item: %s", i)
        logging.debug("Classes: %s", assignments)
        logging.debug("Values : %s", values)
        logging.debug("%s", class_by_value[0:classes, :])

        # decrement count for old class
        old_class = assignments[i]
        value = values[i]
        logging.debug("Old class: %d\nValue: %d", old_class, value)
        class_by_value[old_class, value] -= 1

        # if the old class now has zero members, remove it and decrement number of classes
        if class_by_value[old_class].sum() == 0:
            if old_class < classes - 1:
                class_by_value[old_class, :] = class_by_value[classes - 1, :]
                class_by_value[classes - 1, :] = 0 
                for j in range(len(assignments)):
                    if assignments[j] == classes - 1:
                        assignments[j] = old_class
            classes -= 1

        logging.debug("after decrementing:\n%s", class_by_value[0:classes, :])

        # pick a random number in [0, 1]
        first = numpy.random.rand()

        # either create a new class, or use an existing class
        if first < options.alpha:
            logging.debug("creating new class %d", classes)
            new_class = classes
        else:
            probs = numpy.asfarray(class_by_value[0 : classes, value])
            probs += options.beta
            probs = probs / probs.sum()
            logging.debug("sampling from %s", probs)
            new_class = [i for i, x in enumerate(numpy.random.multinomial(1, probs)) if x == 1][0]


        logging.debug("New class: %d", new_class)
            
        # assign the new class
        assignments[i] = new_class

        # if this is an unseen class, increment number of classes
        if class_by_value[new_class].sum() == 0:
            classes += 1

        # increment count for the new class
        class_by_value[new_class, value] += 1
        logging.debug("\n")

logging.info("Found: %d", classes)
logging.info("Correct: %d", options.latent)

sys.exit()





class_to_item = numpy.zeros(shape=(options.count, options.vocab))
class_counts = numpy.zeros(shape=(options.count))

data = []
sources = numpy.random.multinomial(options.count, mix_params)
true_assignments = []
for source, count in enumerate(sources):
    for i in range(count):
        true_assignments.append(source)
        data.append(numpy.random.multinomial(options.length, params[source]))
data = numpy.asarray(data)

true_assignments = numpy.asarray(true_assignments)
assignments = numpy.zeros(shape=(options.count))

class_counts[0] = options.count
class_to_item[0, :] = data.sum(0)

unused = set(range(len(assignments)))
unused.remove(0)
used = set([0])
cur_count = 1
for i in range(options.iterations):
    for j in range(len(assignments)):
        first = numpy.random.rand()
        old_class = assignments[j]
        class_counts[old_class] -= 1
        class_to_item[old_class] -= data[j]
        if class_counts[old_class] == 0:
            used.remove(old_class)
            unused.add(old_class)

        if first > options.gamma:
            # choose an existing class
            indices = list(used)
            probs = (class_counts[indices] + options.alpha) / (class_counts[indices].sum() + options.alpha * cur_count)
            for val, count in enumerate(data[j]):
                nprob = (class_to_item[:, val] + options.beta) / (class_to_item[:, val].sum() + options.beta * options.vocab)
                for c in range(count):
                    probs = nprob * nprob
            probs = probs / probs.sum()

            #print probs #.sum()
            new_class = [v for v, x in enumerate(numpy.random.multinomial(1, probs)) if x == 1][0]
            #continue
            pass
        elif len(unused) > 0:
            # create a new class
            new_class = unused.pop()
        assignments[j] = new_class
        class_to_item[new_class, :] = data[j, :]
        class_counts[new_class] += 1
        used.add(new_class)
    cur_count = len(set(assignments))
    print "After iteration #%d, %d classes" % (i + 1, cur_count)
print num
#print 
#print class_counts
