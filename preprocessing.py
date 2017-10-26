def preprocess(song):
    from librosa.util import normalize
    return normalize(song)