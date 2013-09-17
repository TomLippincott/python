from math import exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rate", dest="rate", type=float)
parser.add_argument("-t", "--time", dest="time", type=int)
parser.add_argument("-c", "--contribution", dest="contribution", type=float)
parser.add_argument("-f", "--frequency", dest="frequency", type=int)
parser.add_argument("-i", "--initial", dest="initial", type=float)
options = parser.parse_args()

#print options.initial * exp(options.rate * options.time)
total = options.initial
for inc in range(options.time * options.frequency):
    total *= exp(options.rate * 1.0 / options.frequency)
    total += options.contribution / float(options.frequency)
print total
