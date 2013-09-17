import re
import sys
sys.path.append('/usr/lib/collatinus')
temp = sys.stdout
sys.stdout = open('/dev/null','w')
import latinus
sys.stdout.close()
sys.stdout = temp
import unicodedata

mutes = 'pbtdcgqPBTDCGQ'
liquids = 'lrLR'
vowels = 'aeiouAEIOU'
diphthongs = []


# try to get the scansion of the line
def scan(line):
    line = line.replace('qu', 'q')
    feet = []
    for x in re.findall('([%(vowels)s]+)\s*([^%(vowels)s\s]+|$)' % {'vowels' : vowels}, line, re.I):
        if len(x[0]) == 4:
            x = (x[0][2:], x[1])
            feet.append('_')
        if len(x[1]) == 2 and x[1][0] in list(mutes) and x[1][1] in list(liquids):
            feet.append('?')
        elif len(x[0]) == 2 or len(x[1]) == 2:
            feet.append('_')        
        else:
            feet.append('u')
    print feet


# some replacements (mostly dipthongs)
replacements = [
    ('LATIN SMALL LETTER AE', 'ae'),
    ('LATIN CAPITAL LETTER AE', 'ae'),
    ('LATIN SMALL LIGATURE OE', 'oe'),
    ('LATIN SMALL LETTER E WITH DIAERESIS', 'e'),
    ('LATIN CAPITAL LETTER E WITH DIAERESIS', 'e')
    ]


# to make collatinus output less continental
french_replacements = {

    # part-of-speech
    'type' : dict([
        ('verbe', 'verb'),
        ('nom', 'noun'),
        ('pron', 'pronoun'),
        ('adj', 'adjective'),
        ('adjectif verbal', 'predicate-adjective'),
        ('inv', 'adj'),
        ('participe', 'participal'),
        ('g\xe9rondif', 'gerund'),
        ]),

    # case
    'case' : dict([
        ('ablatif', 'ablative'),
        ('nominatif', 'nominative'),
        ('datif', 'dative'),
        ('vocatif', 'vocative'),
        ('accusatif', 'accusative'),
        ('g\xe9nitif', 'genitive'),
        ]),
    
    # gender
    'gender' : dict([
        ('f\xe9minin', 'feminine'),
        ('masculin', 'masculine'),
        ('neutre', 'neuter'),
        ]),

   # number
    'number' : dict([
        ('singulier', 'singular'),
        ('pluriel', 'plural'),
        ]),

    # tense
    'tense' : dict([
        ('parfait', 'perfect'),
        ('pr\xe9sent', 'present'),
        ('future ant\xe9rieur', 'future-perfect'),
        ('plus-que-parfait', 'pluperfect'),
        ('futur', 'future'),
        ('imparfait', 'imperfect'),
        ('futur ant\xe9rieur', 'future anterior'),
        ]),
    
    # person
    'person' : dict([
        ('1\xe8re', '1st'),
        ('2\xe8re', '2nd'),
        ('2\xe8me', '2nd'),
        ('3\xe8re', '3rd'),
        ('3\xefme', '3rd'),
        ('3\xe8me', '3rd'),
        ('2\xefme', '2nd'),
        ]),

    # voice
    'voice' : dict([
        ('actif', 'active'),
        ('passif', 'passive'),
        ('futur ant\xe9rieur', 'future anterior'),
        ]),

    # mood
    'mood' : dict([
        ('subjonctif', 'subjunctive'),
        ('indicatif', 'indicative'),
        ('infinitif', 'infinitive'),
        ('participe', 'participial'),
        ('imp\xefratif', 'imperative'),
        ('imp\xe9ratif', 'imperative'),
        ('3\xefme', '3rd'),
        ('3\xe8me', '3rd'),
        ('2\xefme', '2nd'),
        ('futur ant\xe9rieur', 'future anterior'),
        ('adjectif verbal', 'adjectival verb'),
        ('g\xe9rondif', 'gerund'),
        ]),
    }


# general cleaning of roman alphabet text
remove_rx = re.compile('(,|\.|:|\?|~|\)|\(|;|<.*?>|\[|\])')
def clean_text(text):
    for r in replacements:
        text = re.sub(r[0], r[1], text)
    return remove_rx.sub('', text)


# get the inflectional properties of a single form
def inflection(word):
    word = re.sub('que$', '', word)
    for a, b in replacements:
        word = word.replace(unicodedata.lookup(a), b)
    retval = []
    for lemma, morphs in latinus.analysesDe(word.lower()).items.iteritems():        
        for morph in morphs:
            temp = {}
            try:
                for a, b in {#'lemma' : lemma,
                               'type' : morph.lemme.classe_gr(),
                               'number' : latinus.Nombre(morph.des.nombre),
                               'tense' : latinus.Temps(morph.des.temps),
                               'person' : latinus.Personne(morph.des.personne),
                               'voice' : latinus.Voix(morph.des.voix),
                               'gender' : latinus.Genre(morph.des.genre),
                               'case' : latinus.Cas(morph.des.cas),
                               #'model' : latinus.Modele(morph.des.modele),
                               'mood' : latinus.Mode(morph.des.mode),
                             }.iteritems():
                    if not re.match('^\s*$', b):
                        if a == "lemma":
                            temp[a] = b
                        else:
                            try:
                                temp[a] = french_replacements[a][b]
                            except:
                                sys.stderr.write("%s %s\n" % (a, b))
            except:
                continue
            retval.append(temp)
    return retval


import subprocess
words_rx = re.compile("(?P<morphs>((\S+?)\s{10}.*?\n)+)(?P<entry>.*?\n)(?P<definition>.*?\n)", re.S | re.I | re.U)

def words_inflection(word, wpath="/home/tom/Desktop/words-1.97Ed"):
    p = subprocess.Popen(["%s/words" % (wpath), word[0]], stdout=subprocess.PIPE, cwd=wpath)
    retval = []
    for l in words_rx.finditer(p.stdout.read()):
        print l.group('entry')
        for m in l.group('morphs').split("\n"):
            retval.append(
                {'lemma' : '',
                 'type' : '',
                 'number' : '',
                 'tense' : '',
                 'person' : '',
                 'voice' : '',
                 'gender' : '',
                 'case' : '',
                 'mood' : '',
                 }
                )
            #print m
        #print m.group('entry').strip()
        #print "\t" + m.group('morphs').replace("\n", "\n\t").strip()
        ##print
    #print retval
    return retval

if __name__ == "__main__":
    import sys
    words_inflection(sys.argv[1:])
