# -*- coding: utf-8 -*-
import math
import argparse
import numpy as np

def entropy_kontoyiannis(sequence):
    """
    Estimate the entropy rate of the sequence using Kontoyiannis' algorithm.

    Reference:
        Kontoyiannis, I., Algoet, P. H., Suhov, Y. M., & Wyner, A. J. (1998).
        Nonparametric entropy estimation for stationary processes and random
        fields, with applications to English text. IEEE Transactions on Information
        Theory, 44(3), 1319-1327.

    Equation:
        S_{real} = \left( \frac{1}{n} \sum \Lambda_{i} \right)^{-1}\log_{2}(n)

    Args:
        sequence: the input sequence of symbols.

    Returns:
        A float representing an estimate of the entropy rate of the sequence.
    """
    if not sequence:
        return 0.0
    
    # For the pattern matching below, items in the sequence have to be strings
    sequence = [str(item) for item in sequence]

    lambdas = 0
    n = len(sequence)
    for i in range(n):
        current_sequence = ''.join(sequence[0:i])
        match = True
        k = i
        while match and k < n:
            k += 1
            match = ''.join(sequence[i:k]) in current_sequence
        lambdas += (k - i)
    return (1.0 * len(sequence) / lambdas) * np.log2(len(sequence))

def max_predictability(S, N):
    """
    Estimate the maximum predictability of a sequence with 
    entropy S and alphabet size N.

    Equation:
    $S = - H(\Pi) + (1 - \Pi)\log(N - 1),$
        where $H(\Pi)$ is given by
    $H(\Pi) = \Pi \log_2(\Pi) + (1 - \Pi) \log_2(1 - \Pi)$

    Args:
        S: the entropy of the input sequence of symbols.
        N: the size of the alphabet (number of unique symbols)

    Returns:
        the maximum predictability of the sequence.

    Reference: 
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu, 
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    if S == 0.0 or N <= 1:
        return 1.0
    for p in np.arange(0.0001, 1.0000, 0.0001):
        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        pi_max = h + (1 - p) * math.log2(N - 1) - S
        if pi_max <= 0.001:
            return round(p, 3)
    return 0.0

def main ():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--s", help="Sequence with entropy S", nargs='+')
    parser.add_argument("-a", "--a", help="Alphabet size N", nargs='+')
    args = parser.parse_args()

    ent = entropy_kontoyiannis(args.s)

    print(max_predictability(ent, 3))

if __name__ == '__main__':
    main()