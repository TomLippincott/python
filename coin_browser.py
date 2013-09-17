import PIL
import optparse
import os.path
from glob import glob
from PIL import Image, ImageChops, ImageOps
import numpy
import scipy
import pylab
import itertools
import matplotlib

from scikits.learn.feature_extraction import image
from scikits.learn.cluster import spectral_clustering
from scikits.learn import mixture

parser = optparse.OptionParser()
parser.add_option("-p", "--path", dest="path")
options, args = parser.parse_args()

prefixes = sorted(set([x.replace("front", "HOLD").replace("back", "HOLD") for x in glob(os.path.join(options.path, "*.jpg"))]))

for p in prefixes[0:1]:
    front = numpy.asarray(ImageOps.crop(Image.open(p.replace("HOLD", "front")))) #, 100))
    
    #graph = image.img_to_graph(ds_front)
    #beta = 5
    #eps  = 1e-6
    #graph.data = np.exp(-beta*graph.data/front.std()) + eps

    #N_REGIONS = 11
    #labels = spectral_clustering(graph, k=N_REGIONS)
    #labels = labels.reshape(front.shape)
    sfront = front.sum(2)
    thresh = sfront.mean() * .7
    mask = numpy.where(sfront > thresh, 0, sfront)
    x, y = mask.nonzero()
    X = numpy.asarray([x, y]).T
    #X = X[0:100000, :]
    clf = mixture.GMM(n_states=21, cvtype="tied")
    clf.fit(X, n_iter=100)
    #splot = pylab.subplot(111, aspect=front.shape[0] / float(front.shape[1]))
    color_iter = itertools.cycle([c for c in "bgrcmykwbgrcmykw"])
    Y_ = clf.predict(X)
    #print Y_[0:10]
    #for zip(Y_, X):
    for i, (mean, covar, color) in enumerate(zip(clf.means, clf.covars, color_iter)):
        #v, w = numpy.linalg.eigh(covar)
        #u = w[0] / numpy.linalg.norm(w[0])
        if len(X[Y_==i, 0] > 0):
            pylab.scatter(X[Y_==i, 1], X[Y_==i, 0], color=color)
        #angle = numpy.arctan(u[1]/u[0])
        #angle = 180 * angle / numpy.pi # convert to degrees
        #ell = matplotlib.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)


    pylab.show()


    #nfront = numpy.empty_like(front)
    #nfront[:, :, 0] = front[:, :, 0] * mask
    #nfront[:, :, 1] = front[:, :, 1] * mask
    #nfront[:, :, 2] = front[:, :, 2] * mask

    #print front.shape, ffront.shape
    #back = numpy.asarray(ImageOps.crop(Image.open(p.replace("HOLD", "back")), 100))
    #print sfront.shape, sfront.mean(), ffront.mean()
#im = Image.fromarray(nfront)
#im.show()


#    print front.getextrema()
#print prefixes
#back.show()
