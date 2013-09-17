import pickle
import argparse
import email

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input")
parser.add_argument("-o", "--output", dest="output")
options = parser.parse_args()

emails = [email.message_from_string(x.decode("utf-8")) for x in pickle.load(open(options.input))]

