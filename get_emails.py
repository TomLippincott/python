import imaplib
import getpass
import re
import pickle
import argparse
import email
from email.parser import HeaderParser

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input")
parser.add_argument("-o", "--output", dest="output")
parser.add_argument("-p", "--password", dest="password", default="agm!htf")
options = parser.parse_args()


server = imaplib.IMAP4("imap.hermes.cam.ac.uk", 143)
server.starttls()
server.login("tl318", options.password)
messages = []

server.select("INBOX", readonly=True)
ret, msg_ids = server.search(None, "FROM", '"laura.greig"')
for msg_id in msg_ids[0].split()[0:3]:
    resp, data = server.fetch(msg_id, "(RFC822)")
    messages.append(str(data[0][1]))

server.select("Sent", readonly=True)
ret, msg_ids = server.search(None, "TO", '"greig"')
for msg_id in msg_ids[0].split()[0:3]:
    resp, data = server.fetch(msg_id, "(RFC822)")
    messages.append(str(data[0][1]))

server.logout()

pickle.dump(messages, open(options.output, "wb"))
