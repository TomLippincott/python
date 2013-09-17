from subprocess import Popen
import weka
import sqlite3


def generate_weka_action(target, source, env, for_signature):
    lines = [l for l in source[0].get_contents().split("\n") if not re.match('^\s*$', l)]
    try:
        c = [r for l, r in zip(lines, range(len(lines))) if re.match('.*@ATTRIBUTE.*\{.*\}\s*$', l, re.I|re.S)][0]
    except:
        c = 1
    args= dict.fromkeys(['PRIMARY', 'OTHER', 'SEARCH', 'EVALUATION'], '')
    args.update(env.Dictionary())
    args.update({'SOURCE' : source[0].rstr(), 'TARGET' : target[0].rstr(), 'C' : c})
    args['PRIMARY'] = [l for l in env['WEKACLASSES'].split("\n") if l.endswith(".%s.class" % env['PRIMARY']) and ('classifiers' in l or 'filters' in l)][0][0:-6]
    #print args['PRIMARY']
    return 'java -Xmx2048M -cp %(WEKAPATH)s %(PRIMARY)s %(OTHER)s -c %(C)d %(SEARCH)s %(EVALUATION)s > %(TARGET)s' % args

def build_sqlite(target, source, env):
    conn = sqlite3.connect(target[0].rstr())
    c = conn.cursor()
    c.execute("""create table results (filter text, algorithm text, class text, fmeasure real, fp real, precision real, roc real, recall real, tp real)""")
    for f in source:
        s = f.rstr().split('_')
        filt, alg = s[1], s[3][0:-4]
        parsed = regex.parse_output(f.get_contents())
        for class_res in parsed:
            class_res.update({'Filter' : filt, 'Algorithm' : alg})
            if class_res['Class'] not in ['REAL', 'FAKE']:
                c.execute("""insert into results values (:Filter, :Algorithm, :Class, :FMeasure, :FPRate, :Precision, :ROCArea, :Recall, :TPRate)""", class_res)
    conn.commit()
    c.close()
    return None

def TOOLS_ADD_WEKA(env):
    weka_builder = Builder(generator=generate_weka_action)
    sqlite_builder = Builder(action=build_sqlite)
    env.Append(WEKAPATH=FindFile('weka.jar', ['/usr/share/weka/lib', '/usr/share/java']))
    env.Append(WEKACLASSES=Popen(['jar', 'tf', env['WEKAPATH'].rstr()], stdout=PIPE).stdout.read().replace('/', '.'))
    env.Append(BUILDERS = {'Weka' : weka_builder, 'Sqlite' : sqlite_builder})
