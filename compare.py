def compare(X,Y):
    return cross_similarity_score(X,Y)


def dtw_score(X, Y):
    from librosa import dtw

    D, wp = dtw(X, Y)
    minpath = max([X.shape[1],Y.shape[1]])

    return float(D[-1,-1]/minpath)


def cross_similarity_score(X,Y):
    cs = cross_similarity(X,Y)
    bcs = binary_cs(cs)
    score = bcs_score(bcs)

    return score


def cross_similarity(X,Y):
    from numpy import zeros
    from numpy.linalg import norm

    n = len(X)
    m = len(Y)

    csm = zeros((n, m))

    for i in range(n):
        for j in range(m):
            csm[i][j] = norm(X[i] - Y[j])

    return csm


def binary_cs(X):
    from numpy import zeros, copyto

    n = X.shape[0]
    m = X.shape[1]

    bcs = zeros(X.shape)

    for i in range(n):
        for j in range(m):
            line = zeros(X[i, :].shape)
            column = zeros(X[:, j].shape)
            copyto(line, X[i, :])
            line.sort()
            copyto(column, X[:, j])
            column.sort()
            if (X[i,j] < line[20] and X[i,j] < column[20]):
                bcs[i,j] = 1

    return bcs


def bcs_score(bcs):
    from numpy import zeros, amax

    score_matrix = zeros(bcs.shape)

    n = score_matrix.shape[0]
    m = score_matrix.shape[1]

    for i in range(n):
        for j in range(m):
            value = [0]
            if(i>=2 and j>=2):
                value.append(score_matrix[i-1,j-1]+(2*kronecker_delta(bcs[i-1,j-1]))+affine_gap_penalty(bcs[i-2,j-2], bcs[i-1,j-1]))

            if(i>=3 and j>=2):
                value.append(score_matrix[i-2,j-2]+(2*kronecker_delta(bcs[i-1,j-1]))+affine_gap_penalty(bcs[i-3,j-2], bcs[i-1,j-1]))

            if(i>=2 and j>=3):
                value.append(score_matrix[i-1,j-2]+(2*kronecker_delta(bcs[i-1,j-1]))+affine_gap_penalty(bcs[i-2,j-3], bcs[i-1,j-1]))

            score_matrix[i,j] = max(value)

    return amax(score_matrix)


def kronecker_delta(n):
    if n == 1:
        return 1
    else:
        return 0


def affine_gap_penalty(a,b):
    if b == 1:
        return 0
    elif b == 0 and a == 1:
        return -0.5
    elif b == 0 and a == 0:
        return -0.7