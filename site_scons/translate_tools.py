from SCons.Builder import Builder
from SCons.Script import FindFile
import xml.etree.ElementTree as et
import urllib
try:
    import json
except:
    pass
import cPickle
import time
import glob


def google_translate(target, source, env):
    allwords = set([x.text.strip('.,:;') for x in et.parse(source[0].rstr()).getiterator('w')])
    results = []
    times = []
    accum = 0
    #resume = max([re.match('.*?(\d+)', f).group('num') for f in glob.glob("%s[0-9].*" % target[0])])
    #print resume
        
    print "%d words to translate" % len(allwords)
    while len(allwords) > 0:
        print "%d words remaining" % len(allwords)
        words = []
        while len("".join(words)) < 250 and len(allwords) > 0:
            newword = allwords.pop()
            words.append(newword)
        if len("".join(words)) > 250:
            allwords.add(words[-1])
            words = words[0:-1]
        args = urllib.urlencode([('v', '1.0')] + [('q', x) for x in words] + [('langpair', '%s|%s' % (env['L1'], env['L2']))])        
        url = "http://ajax.googleapis.com/ajax/services/language/translate?%s" % args
        start = time.time()
        txt = urllib.urlopen(url).read()
        end = time.time()
        times.append((len(words), end - start))
        try:
            res = json.loads(txt)
        except:
            print txt
        if isinstance(res['responseData'], list):
            for w, x in zip(words, res['responseData']):
                try:
                    results.append((w, x))
                except:
                    print txt
        else:
            results.append((words[0], res))
        accum += len(words)
        if accum > 500:
            accum = 0
            cPickle.dump((results, times), open(target[0].rstr() + str(len(allwords)), 'wb'))
    cPickle.dump((results, times), open(target[0].rstr(), 'wb'))
    print "average of %f words per second" % (float(sum([x[0] for x in times])) / float(sum([x[1] for x in times])))
    return None

def translation_adder(target, source, env):
    tei = et.parse(source[0].rstr())
    dictionary = dict(cPickle.load(open(source[1].rstr(), 'rb'))[0])
    for w in tei.getiterator('w'):
        if w.attrib['type'] == 'token' and w.text.strip('.,:;') in dictionary:
            try:
                w.attrib['trans'] = dictionary[w.text.strip('.,:;')]['responseData']['translatedText']
            except:
                w.attrib['trans'] = '?'
    tei.write(open(target[0].rstr(), 'w'))
    return None


def TOOLS_ADD(env):
    google_t = Builder(action=google_translate)
    translate_a = Builder(action=translation_adder)
    env.Append(BUILDERS = {'Google' : google_t,
                           'Translate' : translate_a,
                           })
