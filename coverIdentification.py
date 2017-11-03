from librosa import load as loadSong
from json import load as jsonLoad
from feature_extraction import extract_features
from compare import compare

originalInfoPath = '/home/luisfmgs/Documents/tcc/original.json'
vocalCoverInfoPath = '/home/luisfmgs/Documents/tcc/vocal.json'
instrumentalCoverInfoPath = '/home/luisfmgs/Documents/tcc/instrumental.json'

with open(originalInfoPath) as data:
    originalInfo = jsonLoad(data)

with open(vocalCoverInfoPath) as data:
    vocalInfo = jsonLoad(data)

with open(instrumentalCoverInfoPath) as data:
    instrumentalInfo = jsonLoad(data)

original_features = []
for inforiginal in originalInfo:
    original, sr_original = loadSong(path=inforiginal['path'], sr=None)

    original_features.append(extract_features(original, sr_original))

print('vocal cover')
vocalhits = 0

for infocover in vocalInfo:
    vocalCover, sr_vocalCover = loadSong(path=infocover['path'], sr=None)
    minscore = 0
    originalID = 0

    cover_features = extract_features(vocalCover, sr_vocalCover)

    for idx, original in enumerate(original_features):
        score = compare(original, cover_features)

        if (score > minscore):
            minscore = score
            originalID = idx+1
        elif (score == minscore):
            print('eita empatou ):')

    if (originalID == infocover['id']):
        print("acertou!")
        vocalhits += 1
    else:
        print('errou')

print('hits', vocalhits/25.)
print('instrumental cover')
instrumentalhits = 0

for infoinstrumental in instrumentalInfo:
    instrumentalCover, sr_instrumentalCover = loadSong(path=infoinstrumental['path'], sr=None)
    minscore = 0
    originalID = 0

    cover_features = extract_features(instrumentalCover, sr_instrumentalCover)

    for idx, original in enumerate(original_features):
        score = compare(original, cover_features)

        if (score > minscore):
            minscore = score
            originalID = idx+1
        elif (score == minscore):
            print('empate entre', idx+1, originalID)

    if (originalID == infoinstrumental['id']):
        print("acertou!")
        instrumentalhits += 1
    else:
        print('errou')

print('hits', instrumentalhits/25.)