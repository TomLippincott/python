from random import gauss
from matplotlib import pyplot, patches
from scipy import stats

def likelihood(parameters, observations, latent_variables):
    prod = []
    for oi, o in zip(range(len(observations)), observations):
        x, y = o
        total = 0
        for pi, p in zip(range(len(parameters)), parameters):
            if latent_variables[oi] == pi:
                total += p[2] * stats.norm(loc=p[0], scale=p[1]).pdf(o[0])
        prod.append(total[0])
    print prod
    return reduce(lambda x, y : x * y, prod)

def expectation():
    """
    Expected value of the log-likelihood function
    """
    pass


def maximization():
    pass


if __name__ == "__main__":
    import optparse
    import random

    
    parser = optparse.OptionParser()
    parser.add_option('-p', '--points', dest='points', default=50, type='int')
    parser.add_option('-c', '--clusters', dest='clusters', default=2, type='int')
    options, remainder = parser.parse_args()

    data = []
    real_clusters = []
    for c in range(options.clusters):
        s1, m1 = random.randint(1, 10), random.randint(-100, 100)
        s2, m2 = random.randint(1, 10), random.randint(-100, 100)
        for p in range(options.points / options.clusters):
            data.append((random.gauss(m1, s1), random.gauss(m2, s1)))            
            
    parameters = [(1, 1, 10) for i in range(options.clusters)]
    latent_variables = [0 for i in range(options.points)]
    #print likelihood(parameters, data, latent_variables)

    
    #actual = [() for c in range(10)]
    #count = 100
    #Xdata = [gauss(1, 1) for i in range(count)] + [gauss(10, 1) for i in range(count)]
    #Ydata = [gauss(1, 1) for i in range(count)] + [gauss(10, 1) for i in range(count)]
    guesses = []


    for i in range(3):
        for g in guesses:
            g.remove()
        guesses = []
        pyplot.scatter([x[0] for x in data], [x[1] for x in data], s=1)
        for x, y, s in parameters:
            c = patches.Circle((x, y), s, fill=False, ec='red', lw=4)
            pyplot.gca().add_patch(c)
            guesses.append(c)
        pyplot.show()

