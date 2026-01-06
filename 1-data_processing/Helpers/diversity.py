# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict

def _suffix_array_manber_myers(s):
    """
    Compute the suffix array of a string using the Manber-Meyers algorithm.

    Reference: http://algorithmicalley.com/archive/2013/06/30/suffix-arrays.aspx

    Parameters
    ----------
    s : string
        The input string.

    Returns
    -------
    list
        The suffix array of the input string.
    """
    def sort_bucket(s, bucket, order):
        d = defaultdict(list)
        for i in bucket:
            key = ''.join(s[i + order // 2:i + order])
            d[key].append(i)
        result = []
        for k, v in sorted(d.items()):
            if len(v) > 1:
                result += sort_bucket(s, v, 2 * order)
            else:
                result.append(v[0])
        return result

    return sort_bucket(s, range(len(s)), 1)


def _kasai(s, sa):
    """
    Computes the logest common prefix (LCP) array of a string given its suffix 
    array using Kasai's algorithm.

    References: 
        - https://web.stanford.edu/class/cs166/lectures/03/Small03.pdf
        - https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/

    Parameters
    ----------
    s : string
        The input string.
    sa : list
        The suffix array of the input string.

    Returns
    -------
    list
        The LCP array of the input string.
    """
    n = len(s)
    k = 0
    lcp = [0] * n
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    for i in range(n):
        k = k - 1 if k > 0 else 0
        if rank[i] == n - 1:
            k = 0
            continue
        j = sa[rank[i] + 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
    return lcp


def diversity(sequence):
    """
    Returns the ratio of distinct substrings over the total number of 
    substrings in the sequence. The number of distinct substrings is
    computed using the LCP array. The total number of substrings is
    computed using a closed-formula. A naive implementation, which is
    O(n^2), would be too inefficient for large strings. This implementation
    is O(n log n), where n is the size of the input sequence.

    Parameters
    ----------
    sequence : list
        The input sequence of symbols, where each symbol is seen as a
        character in a string.

    Returns
    -------
    float
        The ratio of distinct substrings over the total number of 
        substrings in the sequence
    """
    n = len(sequence)
    n_unique = len(set(sequence))

    if n <= 1:
        return 0.0

    if n == n_unique:
        return .0

    total_substrs = (n * (n + 1)) // 2

    suffix_array = _suffix_array_manber_myers(sequence)
    lcp = _kasai(sequence, suffix_array)
    distinct_substrs = total_substrs - sum(lcp)

    return distinct_substrs / total_substrs

def main ():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sequence", help="The input sequence of symbols, where each symbol is seen as a character in a string", nargs='+')
    args = parser.parse_args()

    print(diversity(args.sequence))

if __name__ == '__main__':
    main()