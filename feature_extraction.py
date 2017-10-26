def extract_features(song,sr):
    return ssm(song,sr)


def beat_synchronous_chroma(song, sr):
    from librosa.beat import beat_track
    from librosa.feature import chroma_cqt
    from librosa.util import sync

    hop_length = 1024

    tempo, beat_frames = beat_track(y=song, sr=sr)

    chromagram = chroma_cqt(y=song, sr=sr, hop_length=hop_length)

    return sync(chromagram, beat_frames)

def beat_synchronous_mfcc(song,sr):
    from librosa.beat import beat_track
    from librosa.feature import mfcc
    from librosa.util import sync

    hop_length = 1024

    tempo, beat_frames = beat_track(y=song, sr=sr)

    mfccs = mfcc(y=song, sr=sr, hop_length=hop_length, n_mfcc=20, n_fft=4096)

    return sync(mfccs, beat_frames)

def both_beat_synchronous(song,sr):
    from numpy import vstack
    beat_chroma = beat_synchronous_chroma(song,sr)
    beat_mfcc = beat_synchronous_mfcc(song,sr)

    return vstack([beat_chroma, beat_mfcc])

def ssm(song,sr,n_beats=9):
    from scipy.spatial.distance import pdist, squareform
    from numpy import asarray, mean
    from numpy.linalg import norm

    mfccs = beat_synchronous_mfcc(song,sr).transpose()
    for i in range(mfccs.shape[0]):
        mfccs[i] = (mfccs[i] - mean(mfccs[i])) / norm(mfccs[i] - mean(mfccs[i]))

    initial_block_index = 0
    final_block_index = n_beats
    result = []

    while final_block_index < mfccs.shape[0]:
        result.append(squareform(pdist(mfccs[initial_block_index:final_block_index, :])))
        initial_block_index += 1
        final_block_index += 1

    return asarray(result)






