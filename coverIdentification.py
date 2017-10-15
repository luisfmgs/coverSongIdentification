from librosa import load as loadSong
from librosa.feature import mfcc as getMfccs
from librosa.beat import beat_track
from json import load as jsonLoad
from numpy import concatenate
from scipy.spatial.distance import euclidean
sampleRate = 44100

songsInfoPath = '/home/luisfmgs/Documents/tcc/songsinfo.json'
with open(songsInfoPath) as data:
    songsInfo = jsonLoad(data)

originalFeatures = []
vocalFeatures = []
instrumentalFeatures = []

for song in songsInfo:
    originalWaveForm, sr = loadSong(song['originalPath'], sampleRate)
    vocalCoverWaveForm, sr = loadSong(song['vocalCoverPath'], sampleRate)
    instrumentalCoverWaveForm, sr = loadSong(song['instrumentalCoverPath'], sampleRate)

    originalMfccs = getMfccs(originalWaveForm, sampleRate, None, 13)
    vocalMfccs = getMfccs(vocalCoverWaveForm, sampleRate, None, 13)
    instrumentalMfccs = getMfccs(instrumentalCoverWaveForm, sampleRate, None, 13)

    originalFeatures.append(
        {
            'id': song['id'],
            'features': concatenate([originalMfccs.mean(axis=1), originalMfccs.var(axis=1)])
        }
    )
    vocalFeatures.append(
        {
            'id': song['id'],
            'features': concatenate([vocalMfccs.mean(axis=1), vocalMfccs.var(axis=1)])
        }
    )
    instrumentalFeatures.append(
        {
            'id': song['id'],
            'features': concatenate([instrumentalMfccs.mean(axis=1), instrumentalMfccs.var(axis=1)])
        }
    )

for vocalCover in vocalFeatures:
    minDiff = float('inf')
    minId = None
    for original in originalFeatures:
        if euclidean(original['features'], vocalCover['features']) < minDiff:
            minDiff = euclidean(original['features'], vocalCover['features'])
            minId = original['id']

    print 'para cover id ', vocalCover['id']
    print 'eu achei que fosse ', minId
