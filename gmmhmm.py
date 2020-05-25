# import numpy as np
# import matplotlib.pyplot as plt
# from scikits.talkbox.features import mfcc
# from scipy.io import wavfile
# from hmmlearn import hmm
# import numpy as np
# import os
# import warnings
# import scipy.stats as sp

import numpy as np
import matplotlib.pyplot as plt
from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import os
import warnings
import scipy.stats as sp

warnings.filterwarnings("ignore")


fpaths = []
labels = []
spoken = []
features = []

for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):
        print('audio/' + f + '/' + w)
        fpaths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print('Words spoken:', spoken)




for n, file in enumerate(fpaths):
    samplerate, d = wavfile.read(file)
    features.append(mfcc(d, nwin=int(samplerate * 0.03), fs=samplerate, nceps= 6)[0])






c = list(zip(features, labels))
np.random.shuffle(c)
features,labels = zip(*c)

print(len(features))
print(len(labels))

m_trainingsetfeatures = features[0:84]
m_trainingsetlabels = labels[0:84]

print(len(m_trainingsetfeatures))
print(len(m_trainingsetlabels))

m_testingsetfeatures = features[84:105]
m_testingsetlabels = labels[84:105]


print(len(m_testingsetfeatures))
print(len(m_testingsetlabels))


gmmhmmindexdict = {}
index = 0
for word in spoken:
    gmmhmmindexdict[word] = index
    index = index +1


print ('Loading of data completed')

#Parameters needed to train GMMHMM
m_num_of_HMMStates = 3  # number of states
m_num_of_mixtures = 2  # number of mixtures for each hidden state
m_covarianceType = 'diag'  # covariance type
m_n_iter = 10  # number of iterations
m_bakisLevel = 2


def initByBakis(inumstates, ibakisLevel):
    startprobPrior = np.zeros(inumstates)
    startprobPrior[0: ibakisLevel - 1] = 1/float((ibakisLevel - 1))
    transmatPrior = getTransmatPrior(inumstates, ibakisLevel)
    return startprobPrior, transmatPrior


def getTransmatPrior(inumstates, ibakisLevel):
    transmatPrior = (1 / float(ibakisLevel)) * np.eye(inumstates)

    for i in range(inumstates - (ibakisLevel - 1)):
        for j in range(ibakisLevel - 1):
            transmatPrior[i, i + j + 1] = 1. / ibakisLevel

    for i in range(inumstates - ibakisLevel + 1, inumstates):
        for j in range(inumstates - i - j):
            transmatPrior[i, i + j] = 1. / (inumstates - i)

    return transmatPrior

m_startprobPrior ,m_transmatPrior = initByBakis(m_num_of_HMMStates,m_bakisLevel)


print("StartProbPrior=")
print(m_startprobPrior)

print("TransMatPrior=")
print(m_transmatPrior)


class SpeechModel:
    def __init__(self,Class,label):
        self.traindata = np.zeros((0,6))
        self.Class = Class
        self.label = label
        self.model  = hmm.GMMHMM(n_components = m_num_of_HMMStates, n_mix = m_num_of_mixtures, \
                           transmat_prior = m_transmatPrior, startprob_prior = m_startprobPrior, \
                                        covariance_type = m_covarianceType, n_iter = m_n_iter)



#7 GMMHMM Models would be created for 7 words
speechmodels = [None] * 7


for key in gmmhmmindexdict:
    speechmodels[gmmhmmindexdict[key]] = SpeechModel(gmmhmmindexdict[key],key)

for i in range(0,len(m_trainingsetfeatures)):
     for j in range(0,len(speechmodels)):
         if int(speechmodels[j].Class) == int(gmmhmmindexdict[m_trainingsetlabels[i]]):
            speechmodels[j].traindata = np.concatenate((speechmodels[j].traindata , m_trainingsetfeatures[i]))



for speechmodel in speechmodels:
    speechmodel.model.fit(speechmodel.traindata)


print ('Training completed -- 7 GMM-HMM models are built for 7 different types of words')
print (" ")
print(" ")

print("Prediction started")



#Testing
m_PredictionlabelList = []

for i in range(0,len(m_testingsetfeatures)):
    scores = []
    for speechmodel in speechmodels:
         scores.append(speechmodel.model.score(m_testingsetfeatures[i]))
    id  = scores.index(max(scores))
    m_PredictionlabelList.append(speechmodels[id].Class)
    print(str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) +" "+":"+ speechmodels[id].label)

accuracy = 0.0
count = 0


print("")
print("Prediction for Testing DataSet:")

for i in range(0,len(m_testingsetlabels)):
    print( "Label"+str(i+1)+":"+m_testingsetlabels[i])
    if gmmhmmindexdict[m_testingsetlabels[i]] == m_PredictionlabelList[i]:
       count = count+1

accuracy = 100.0*count/float(len(m_testingsetlabels))

print("")
print("accuracy ="+str(accuracy))
print("")
