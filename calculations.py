def extract_features(song, sample_rate, B=14):
    from librosa.beat import beat_track
    from librosa.feature import mfcc
    from numpy import mean
    from numpy.linalg import norm

    # Calculate the number of samples in 23 ms
    beat_tracking_hop = sample_rate*23/1000

    tempo, beat_frames = beat_track(y=song, sr=sample_rate, hop_length=beat_tracking_hop)

    avarage_tempo_length = int(60*sample_rate/tempo)
    mfcc_hop = avarage_tempo_length/200

    initial_block_index = 0
    final_block_index = B*avarage_tempo_length
    result = []

    while final_block_index < song.shape[0]:
        mfccs = mfcc(y=song[initial_block_index:final_block_index], sr=sample_rate,
                     n_fft=avarage_tempo_length, hop_length=mfcc_hop, n_mfcc=20).transpose()
        for i in range(mfccs.shape[0]):
            mfccs[i] = (mfccs[i] - mean(mfccs[i]))/norm(mfccs[i] - mean(mfccs[i]))

        result.append(self_similarity_matrix(mfccs))

        initial_block_index += avarage_tempo_length
        final_block_index += avarage_tempo_length

    return result


def self_similarity_matrix(feature_matrix):
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(feature_matrix))
