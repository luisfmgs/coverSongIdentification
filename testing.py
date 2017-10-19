from librosa import load as loadSong
from json import load as jsonLoad
import calculations
#import matplotlib.pyplot as plt
N = 1


songsInfoPath = '/home/luisfmgs/Documents/tcc/songsinfo.json'

with open(songsInfoPath) as data:
    songsInfo = jsonLoad(data)

originalsong, sr1 = loadSong(path=songsInfo[N]['originalPath'], sr=None)
vocalcover, sr2 = loadSong(path=songsInfo[N]['vocalCoverPath'], sr=None)
#instrumentalcover, sr3 = loadSong(path=songsInfo[N]['instrumentalCoverPath'], sr=None)

originalfeatures = calculations.extract_features(originalsong, sr1)
vocalfeatures = calculations.extract_features(vocalcover, sr2)
#instrumentalfeatures = calculations.extract_features(instrumentalcover, sr3)