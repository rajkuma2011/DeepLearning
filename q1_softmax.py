import numpy as np

def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        maxVect = np.max(x, axis=1)
        x -= maxVect.reshape(len(maxVect), 1)
        t = np.sum(np.exp(x), axis=1)
        return np.exp(x)/t.reshape(len(t),1)
    else:
        # Vector
        maxVal = np.max(x)
        x -= maxVal
        total = np.sum(np.exp(x))
        return np.exp(x)/total

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    scores2D = np.array([[1, 2, 3, 6],
                         [2, 4, 5, 6],
                         [3, 8, 7, 6]])
    scores2D = softmax(scores2D)
    print scores2D
    output = [[ 0.00626879,  0.01704033,  0.04632042,  0.93037047],
              [ 0.01203764,  0.08894682,  0.24178252,  0.65723302],
              [ 0.00446236,  0.66227241,  0.24363641,  0.08962882]]

    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
