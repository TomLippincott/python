from SCons.Builder import Builder
import re
from glob import glob
from functools import partial
import logging
import os.path
import os
from common_tools import meta_open
import cPickle as pickle
import numpy
import math
import xml.etree.ElementTree as et
from rpy2.robjects.packages import importr
from rpy2.robjects import ListVector, IntVector, StrVector
from rpy2.robjects.conversion import ri2py
from rpy2.robjects.numpy2ri import numpy2ri
from common_tools import meta_open, meta_basename, jags_to_numpy, normalized_information_distance as nid, adjusted_rand, harmonic_mean, modified_purity
import xml.sax

stats = importr("stats")
rcluster = importr("cluster")

def strip_extensions(file_name):
    return re.match(r"^(.*?)(\.\w{2,3})*$", file_name).group(1)

def run_hacked_model(target, source, env):
    import graphmod as gm
    args = source[-1].read()
    instances = gm.Instances(source[0].rstr())
    params = gm.Parameters(args)
    bn = gm.BayesianNetwork(instances, params)
    ct = gm.StringCounts()
    bn.compile(ct)
    for i in range(10):
        logging.info("Iteration #%s", i + 1)
        bn.sample(ct)
    print ct
    #print bn.xml()
    #counts = gm.run_hacked_model([str(x) for x in sum([["--%s" % k, v] for k, v in args.iteritems()], [])], instances)
    #data = {}
    #for k in [("verb", "verb_class"), ("verb_class", "syntactic_frame"), ("verb_class", "semantic_frame"), ("semantic_class", "lemma"), ("syntactic_class", "tag")]:
        #print k
    #    if len(k) == 2:
    #        data[k] = numpy.asarray(counts.get(k[0], k[1]))
    #pickle.dump([data, [instances.get_name("verb", x) for x in range(instances.get_size("verb"))]], open(target[0].rstr(), "wb"))
    return None



def cluster_verbs(target, source, env):
    args = source[-1].read()
    return None
    datas, verbs = pickle.load(open(source[0].rstr(), "rb"))
    data = datas[("verb", "verb_class")]
    data = numpy.transpose(data.T / data.sum(1))
    if "clusters" in args:
        res = stats.kmeans(numpy2ri(data), centers=args["clusters"])
    else:
        tres = numpy.asarray(rcluster.clusGap(numpy2ri(data), FUN=stats.kmeans, K_max=30, B=500).rx2("Tab"))
        gaps = tres[:, 2]
        err = tres[:, 3]    
        best = rcluster.maxSE(numpy2ri(gaps), numpy2ri(err), method="globalmax")
        res = stats.kmeans(numpy2ri(data), centers=best)
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")

    return None


def run_model(target, source, env):
    import graphmod as gm

    args = source[-1].read()
    instances = gm.Instances(source[0].rstr())    
    parameters = gm.Parameters({"verb_class" : 10, "syntactic_frame" : 11, "semantic_frame" : 12, "semantic_class" : 13, "syntactic_class" : 14})

    #db = gm.MysqlDatabase("localhost", "graphmod", "", "")
    #prm = gm.PRM(source[1].rstr(), db, True)
    #db.clear_database()
    #db.create_table("test", ["col1", "col2"], ["int", "int"])
    #graph = gm.Graph(source[1].rstr())

    bayesian_network = gm.BayesianNetwork(instances, parameters)
    #print bayesian_network
    
    logging.info("Bayesian Network: %s", bayesian_network)
    #cl = gm.Collapser("data/bayesian_network_conjugates.xml")
    #cl.apply(bayesian_network)
    #cl.apply(graph)
    factor_graph = gm.FactorGraph(bayesian_network)
    compiler = gm.Compiler("data/factor_patterns.xml")
    compiler.compile(factor_graph)
    #dircat = gm.Factor(["dir"], ["cat"])
    #factor_graph.add_factor(dircat)

    factor_graph.forward_sample()
    #print "again"
    #factor_graph.forward_sample()
    
    logging.info("Factor Graph: %s", factor_graph)
    #bayesian_network.expand_plates(parameters, instances)
    #factor_graph = bayesian_network.to_factor_graph()

    #parameters = {}

    #print len([x for x in xml.iter()])
    #compile_plates(xml, instances, parameters)
    #print len([x for x in xml.iter()])
    #print graph
    #print factor_graph
    #print bayesian_network
    open(target[0].rstr(), "w").write(bayesian_network.xml())
    open(target[1].rstr(), "w").write(factor_graph.xml())
    #open(target[2].rstr(), "w").write(graph.xml())
    #open(target[0].rstr(), "w").write(bayesian_network.str())
    logging.info("%s", instances)
    return None


def run_scf(target, source, env):
    import graphmod as gm
    args = source[-1].read()
    parameters = args["parameters"]
    nodes = {}
    factors = {}
    variables = {}


    instances = gm.Instances()
    gm.load_instances(source[0].rstr(), instances)
    if "features" in args:
        instances.transform("argument", args["features"])

    graph = gm.Graph()

    if args.get("model", "") == "multi":
        variables["alpha"] = gm.ContinuousVectorVariable(parameters["scf"], parameters["alpha"])
        variables["beta"] = gm.ContinuousVectorVariable(instances.get_size("argument"), parameters["beta"])

        vxf = []
        for v in range(instances.get_size("verb_lemma")):
            variables["VERB%dxFRAME" % v] = gm.ContinuousVectorVariable([1.0 / parameters["scf"] for x in range(parameters["scf"])])
            vxf.append(variables["VERB%dxFRAME" % v])
            factors["VERB%dxFRAME" % v] = gm.DirichletMultinomial(variables["alpha"], variables["VERB%dxFRAME" % v])
        variables["VERBxFRAME"] = gm.TiledContinuousVectorVariable(gm.ContinuousVectorVariableVector(vxf))

        fxa = []
        for f in range(parameters["scf"]):
            variables["FRAME%dxARG" % f] = gm.ContinuousVectorVariable([1.0 / instances.get_size("argument") for x in range(instances.get_size("argument"))])
            vxf.append(variables["FRAME%dxARG" % f])
            factors["FRAME%dxARG" % f] = gm.DirichletMultinomial(variables["beta"], variables["FRAME%dxARG" % f])
        variables["FRAMExARG"] = gm.TiledContinuousVectorVariable(gm.ContinuousVectorVariableVector(fxa))


        for instance in range(len(instances)):
            verb_id = instances.at(instance)("verb_lemma")[0]
            vname = "instance%d_verb" % (instance)
            variables[vname] = gm.DiscreteScalarVariable(verb_id)
            variables[vname].set_support(instances.get_size("verb_lemma"))
            variables[vname].set_name("verb")

            iname = "instance%d_scf" % (instance)
            variables[iname] = gm.DiscreteScalarVariable()
            variables[iname].set_support(parameters["scf"])
            variables[iname].set_name("scf")

            fnameA = "instance%d_fA" % (instance)
            factors[fnameA] = gm.MultinomialCategorical(variables["VERB0xFRAME"], variables[iname])
            #factors[fnameA] = gm.MultinomialCategorical(variables["VERB0xFRAME"], variables[vname], variables[iname])

            for i, val in enumerate(instances.at(instance)("argument")):
                oname = "instance%d_observation%d" % (instance, i)
                variables[oname] = gm.DiscreteScalarVariable(val)
                variables[oname].set_support(instances.get_size("argument"))
                variables[oname].set_name("argument")                
                fnameB = "instance%d_fB_%d" % (instance, i)
                factors[fnameB] = gm.MultinomialCategorical(variables["FRAME0xARG"], variables[oname])
                #factors[fnameB] = gm.DirichletMultinomialMixture(variables["beta"], variables[iname], variables[oname])

    if args.get("model", "") == "pymulti":
        variables["alpha"] = gm.ContinuousScalarVariable(parameters["alpha"])
        variables["gamma"] = gm.ContinuousScalarVariable(parameters["gamma"])
        variables["beta"] = gm.ContinuousVectorVariable(instances.get_size("argument"), parameters["beta"])
        for instance in range(len(instances)):
            verb_id = instances.at(instance)("verb_lemma")[0]
            vname = "instance%d_verb" % (instance)
            variables[vname] = gm.DiscreteScalarVariable(verb_id)
            variables[vname].set_support(instances.get_size("verb_lemma"))
            variables[vname].set_name("verb")

            iname = "instance%d_scf" % (instance)
            variables[iname] = gm.DiscreteScalarVariable()
            variables[iname].set_support(parameters["scf"])
            variables[iname].set_name("scf")

            fnameA = "instance%d_fA" % (instance)
            factors[fnameA] = gm.DirichletMultinomialMixture(variables["alpha"], variables[vname], variables[iname])

            for i, val in enumerate(instances.at(instance)("argument")):
                oname = "instance%d_observation%d" % (instance, i)
                variables[oname] = gm.DiscreteScalarVariable(val)
                variables[oname].set_support(instances.get_size("argument"))
                variables[oname].set_name("argument")                
                fnameB = "instance%d_fB_%d" % (instance, i)
                factors[fnameB] = gm.DirichletMultinomialMixture(variables["beta"], variables[iname], variables[oname])


    elif args.get("model", "") == "beta":
        variables["alpha"] = gm.ContinuousVectorVariable(parameters["scf"], parameters["alpha"])
        betas = []
        for i in range(instances.get_size("argument")):
            variables["beta%d" % i] = gm.ContinuousVectorVariable(2, parameters["beta"])
            betas.append(variables["beta%d" % i])

        variables["beta"] = gm.TiledContinuousVectorVariable(gm.ContinuousVectorVariableVector(betas))
        for instance in range(len(instances)):
            verb_id = instances.at(instance)("verb_lemma")[0]
            vname = "instance%d_verb" % (instance)
            variables[vname] = gm.DiscreteScalarVariable(verb_id)
            variables[vname].set_support(instances.get_size("verb_lemma"))
            variables[vname].set_name("verb")

            iname = "instance%d_scf" % (instance)
            variables[iname] = gm.DiscreteScalarVariable()
            variables[iname].set_support(parameters["scf"])
            variables[iname].set_name("scf")

            fnameA = "instance%d_fA" % (instance)
            factors[fnameA] = gm.DirichletMultinomialMixture(variables["alpha"], variables[vname], variables[iname])

            oname = "instance%d_observation" % (instance)            
            variables[oname] = gm.DiscreteMapVariable(instances.at(instance)("argument"))
            variables[oname].set_support(instances.get_size("argument"))
            variables[oname].set_name("argument")                
            fnameB = "instance%d_fB" % (instance)
            factors[fnameB] = gm.BetaBinomialMixture(variables["beta"], variables[iname], variables[oname])


    for variable in variables.values():
        graph.add_variable(variable)

    for factor in factors.values():
        graph.add_factor(factor)

    logging.error("Compiling graph...")
    graph.compile()
    logging.error("Creating counter...")
    counts = graph.get_counts_object()
    print len(graph.get_variables()), len(graph.get_factors())
    samples = numpy.empty(shape=(instances.get_size("verb_lemma"), parameters["scf"], args["samples"]))
    logging.error("Starting sampling...")
    return None
    #for v in variables.values():
    #    if not gm.BaseVariable_get_fixed(v):
    #        print v
    #        v.sample(counts)

    #return None


    for i in range(args["burnins"]):
        graph.sample(counts)
        #print nodes["alpha"].get_value()
        logging.error("Burnin #%s", i + 1)
        #print numpy.asarray(variables["alpha"].get_value())
        print numpy.asarray(counts("scf"))
        #print numpy.asarray(counts("scf", "argument"))
        #ll = sampler.log_likelihood(graph)
        #logging.error("LL = %f", ll)
        #print sampler.get_counts("scf")
        #print nodes["alpha"]
        #print nodes["beta"]

    variables["alpha"].set_fixed(False)

    #variables["beta"].set_fixed(False)

    for i in range(args["samples"]):
        graph.sample(counts)
        logging.error("Sample #%s", i + 1)
        #logging.error("alpha=%s, beta=%s", variables["alpha"].get_value()[0], variables["beta"].get_value()[0])
        #ll = sampler.log_likelihood(graph)
        #logging.error("LL = %f", ll)
        #print sampler.get_counts("scf")

        #print nodes["beta"]
        samples[:, :, i] = numpy.asarray(counts("verb", "scf"))

    row_names = [instances.get_name("verb_lemma", x) for x in range(samples.shape[0])]
    pickle.dump((row_names, samples), meta_open(target[0].rstr(), "w"))
    return None



def conll_to_graphmod(target, source, env):
    import graphmod as gm
    args = source[-1].read()
    if args.get("verbs"):
        keep_verbs = [x.split()[0] for x in meta_open(args["verbs"])]
    else:
        keep_verbs = []
    instances = gm.Instances()
    gm.from_conll(args.get("inputs", []), instances, keep_verbs, 0)
    logging.info(instances)
    instances.save(target[0].rstr())
    return None


def evaluate_clustering(target, source, env):
    cluster_names = {}
    gdata = {}
    for verb, cluster_name in [x.strip().split() for x in meta_open(source[1].rstr())]:
        if cluster_name not in cluster_names:
            cluster_names[cluster_name] = len(cluster_names)
        gdata[verb] = cluster_names[cluster_name] + 1

    data = {}
    for i, cluster in enumerate([x.split() for x in meta_open(source[0].rstr())]):
        for verb in cluster:
            data[verb] = i + 1

    kverbs = sorted(set(data.keys()).intersection(gdata.keys()))
    tcand, tgold = [data[x] for x in kverbs], [gdata[x] for x in kverbs]

    gmapping = dict([(x, i + 1) for i, x in enumerate(set(tgold))])
    cmapping = dict([(x, i + 1) for i, x in enumerate(set(tcand))])
    gold = [gmapping[x] for x in tgold]
    cand = [cmapping[x] for x in tcand]
    mpur = harmonic_mean(modified_purity(gold, cand), modified_purity(cand, gold, 2))
    open(target[0].rstr(), "w").write("1 - NID\tARI\tmPur\n%s\t%s\t%s\n" % (1.0 - nid(cand, gold), adjusted_rand(gold, cand), mpur))
    return None


# def cluster_verbs(target, source, env):
#     args = source[-1].read()
#     #verbs, samples = pickle.load(meta_open(source[0].rstr()))
#     #samples = numpy.asarray(samples)
#     #samples = samples.sum(2)
#     feat = args.get("feat", "class")
#     all_data = {}
#     for line in open(source[0].rstr()):
#         toks = line.strip().split()
#         if not toks[0].startswith("_"):
#             verb = toks[0]
#             other = toks[1]
#             vals = [float(x.strip("[],")) for x in toks[2:]]
#             if sum(vals) > 0:
#                 all_data[verb] = all_data.get(verb, {})
#                 all_data[verb][other] = vals



#     data = numpy.zeros(shape=(len(all_data), len(all_data.values()[0]["_%s" % feat])))
#     verbs = sorted(all_data.keys())
#     for i, verb in enumerate(verbs):
#         data[i, :] = all_data[verb]["_%s" % feat]

    


#     data = numpy.transpose(data.T / data.sum(1))
#     if "clusters" in args:
#         res = stats.kmeans(numpy2ri(data), centers=args["clusters"])
#     else:
#         tres = numpy.asarray(rcluster.clusGap(numpy2ri(data), FUN=stats.kmeans, K_max=30, B=500).rx2("Tab"))
#         gaps = tres[:, 2]
#         err = tres[:, 3]    
#         best = rcluster.maxSE(numpy2ri(gaps), numpy2ri(err), method="globalmax")
#         res = stats.kmeans(numpy2ri(data), centers=best)
#     ofd = meta_open(target[0].rstr(), "w")
#     for c in set(res.rx2("cluster")):
#         ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")
#     return None

def old_cluster_verbs(target, source, env):
    args = source[-1].read()
    #verbs, samples = pickle.load(meta_open(source[0].rstr()))
    #samples = numpy.asarray(samples)
    #samples = samples.sum(2)
    feat = args.get("feat", "class")
    all_data = {}
    for line in open(source[0].rstr()):
        toks = line.strip().split()
        if not toks[0].startswith("_"):
            verb = toks[0]
            other = toks[1]
            vals = [float(x.strip("[],")) for x in toks[2:]]
            if sum(vals) > 0:
                all_data[verb] = all_data.get(verb, {})
                all_data[verb][other] = vals



    data = numpy.zeros(shape=(len(all_data), len(all_data.values()[0]["_%s" % feat])))
    verbs = sorted(all_data.keys())
    for i, verb in enumerate(verbs):
        data[i, :] = all_data[verb]["_%s" % feat]

    


    data = numpy.transpose(data.T / data.sum(1))
    if "clusters" in args:
        res = stats.kmeans(numpy2ri(data), centers=args["clusters"])
    else:
        tres = numpy.asarray(rcluster.clusGap(numpy2ri(data), FUN=stats.kmeans, K_max=30, B=500).rx2("Tab"))
        gaps = tres[:, 2]
        err = tres[:, 3]    
        best = rcluster.maxSE(numpy2ri(gaps), numpy2ri(err), method="globalmax")
        res = stats.kmeans(numpy2ri(data), centers=best)
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")
    return None


def cluster_by_grs(target, source, env):
    import graphmod as gm
    args = source[-1].read()
    verb_map = {}
    gr_map = {}
    instances = gm.Instances()
    gm.load_instances(source[0].rstr(), instances)
    for ii in range(len(instances)):
        verb = instances.get_name("verb_lemma", instances.at(ii)["verb_lemma"][0])
        grs = [instances.get_name("gr", x) for x in instances.at(ii)["gr"]]
        verb_map[verb] = verb_map.get(verb, len(verb_map))
        for gr in grs:
            gr_map[gr] = gr_map.get(gr, len(gr_map))
    data = numpy.zeros(shape=(len(verb_map), len(gr_map)))
    for ii in range(len(instances)):
        verb = instances.get_name("verb_lemma", instances.at(ii)["verb_lemma"][0])
        verb_id = verb_map[verb]
        grs = [instances.get_name("gr", x) for x in instances.at(ii)["gr"]]
        gr_ids = [gr_map[x] for x in grs]
        for gr in gr_ids:
            data[verb_id, gr] += 1
    data = numpy.transpose(data.T / data.sum(1))
    tres = numpy.asarray(rcluster.clusGap(numpy2ri(data), FUN=stats.kmeans, K_max=30, B=500).rx2("Tab"))
    gaps = tres[:, 2]
    err = tres[:, 3]    
    best = rcluster.maxSE(numpy2ri(gaps), numpy2ri(err), method="globalmax")
    res = stats.kmeans(numpy2ri(data), centers=best)
    verbs = dict([(v, k) for k, v in verb_map.iteritems()])
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")
    return None


def cluster_by_valex(target, source, env):
    import graphmod as gm
    args = source[-1].read()
    target_verbs = set()
    instances = gm.Instances()
    gm.load_instances(source[0].rstr(), instances)
    for vid in range(instances.get_size("verb_lemma")):
        target_verbs.add(instances.get_name("verb_lemma", vid))
    data = {}
    scfs = {}
    verbs = {}
    for fname in sorted(glob(os.path.join("%s/lex-%s" % (env["VALEX_LEXICON"], args["lexicon"]), "*"))):
        verb = os.path.basename(fname).split(".")[0]
        if verb not in target_verbs:
            continue
        data[verb] = {}
        for m in re.finditer(r":CLASSES \((.*?)\).*\n.*FREQCNT (\d+)", meta_open(fname).read()):
            scf = int(m.group(1).split()[0])
            count = int(m.group(2))
            scfs[scf] = scfs.get(scf, 0) + count
            verbs[verb] = verbs.get(verb, 0) + count
            data[verb][scf] = count
    ddata = numpy.zeros(shape=(len(verbs), len(scfs)))
    verbs = sorted(verbs)
    scfs = sorted(scfs)
    for row, verb in enumerate(verbs):
        for col, scf in enumerate(scfs):
            ddata[row, col] = data[verb].get(scf, 0)

    data = numpy.transpose(ddata.T / ddata.sum(1))
    tres = numpy.asarray(rcluster.clusGap(numpy2ri(data), FUN=stats.kmeans, K_max=30, B=500).rx2("Tab"))
    gaps = tres[:, 2]
    err = tres[:, 3]    
    best = rcluster.maxSE(numpy2ri(gaps), numpy2ri(err), method="globalmax")
    res = stats.kmeans(numpy2ri(data), centers=best)
    ofd = meta_open(target[0].rstr(), "w")
    for c in set(res.rx2("cluster")):
        ofd.write(" ".join([verbs[i] for i, a in enumerate(res.rx2("cluster")) if a == c]) + "\n")
    return None


def graphmod_run(target, source, env, for_signature):
    args = source[-1].read()
    options = ""
    for k, v in args.iteritems():
        options += " --%s %s" % (k, v)
    options += " " + " ".join(glob(os.path.join(env[source[0].read()], "*")))
    return "work/main --output $TARGET %s" % (options)

def graphmod_run_emitter(target, source, env):
    name = "test"
    for k, v in source[-1].read().iteritems():
        name += "_%s=%s" % (k, v)
    target[0] = os.path.join("work", source[0].rstr(), "%s.out" % name.replace("/", "_"))
    source[0] = env.Value(source[0].rstr())
    return target, source

def evaluate_emitter(target, source, env):
    target[0] = source[0].rstr() + ".results"
    return target, source

def xslt_emitter(target, source, env):
    target[0] = strip_extensions(source[0].rstr()) + ".viz"
    return target, source

def dot_emitter(target, source, env):
    target[0] = strip_extensions(source[0].rstr()) + ".png"
    return target, source

def cluster_emitter(target, source, env):
    #target[0] = source[0].rstr() + "%s.cluster" % (source[-1].read()["feat"])
    return target, source

def run_model_emitter(target, source, env):
    target[0] = strip_extensions(source[0].rstr()) + "_bayesian_network.xml"
    target.append(strip_extensions(source[0].rstr()) + "_factor_graph.xml")
    #target.append(strip_extensions(source[0].rstr()) + "_simple_graph.xml")
    return target, source


def TOOLS_ADD(env):
    env.Append(BUILDERS = {
            'RunSCF' : Builder(action=run_scf),
            'RunModel' : Builder(action=run_model, emitter=run_model_emitter),
            'RunHackedModel' : Builder(action=run_hacked_model),
            'ConllToGraphmod' : Builder(action=conll_to_graphmod),
            "ClusterVerbs" : Builder(action=cluster_verbs, emitter=cluster_emitter),
            "EvaluateClustering" : Builder(action=evaluate_clustering, emitter=evaluate_emitter),
            "ClusterByGRs" : Builder(action=cluster_by_grs),
            "ClusterByValex" : Builder(action=cluster_by_valex),
            #"Graphmod" : Builder(action="work/graphmod_test --input ${SOURCES[0]} --output ${TARGET} --data ${SOURCES[1]}"),
            "Graphmod" : Builder(generator=graphmod_run, emitter=graphmod_run_emitter),
            "XSLT" : Builder(action="xsltproc ${SOURCES[1]} ${SOURCES[0]} > $TARGET", emitter=xslt_emitter),
            "DOT" : Builder(action="dot -Tpng -o${TARGET} ${SOURCE}", emitter=dot_emitter),
            })
