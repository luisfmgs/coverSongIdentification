from librosa import load as loadSong
from json import load as jsonLoad
from preprocessing import preprocess
from feature_extraction import extract_features,beat_synchronous_mfcc
from compare import compare
from librosa import dtw
import matplotlib.pyplot as plt
from librosa.display import specshow

originalInfoPath = '/home/luisfmgs/Documents/tcc/original.json'
vocalCoverInfoPath = '/home/luisfmgs/Documents/tcc/vocal.json'
instrumentalCoverInfoPath = '/home/luisfmgs/Documents/tcc/instrumental.json'

with open(originalInfoPath) as data:
    originalInfo = jsonLoad(data)

with open(vocalCoverInfoPath) as data:
    vocalInfo = jsonLoad(data)

with open(instrumentalCoverInfoPath) as data:
    instrumentalInfo = jsonLoad(data)

original, sr_original = loadSong(path=originalInfo[1]['path'], sr=None)
vocalCover, sr_vocalCover = loadSong(path=vocalInfo[1]['path'], sr=None)

original_features = extract_features(original, sr_original)
cover_features = extract_features(vocalCover, sr_vocalCover)


x = compare(original_features, cover_features)
plt.matshow(x)
