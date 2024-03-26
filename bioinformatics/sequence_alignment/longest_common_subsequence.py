import argparse as ap

"""
A longest common subsequence (LCS) is the longest subsequence common to all
sequences in a set of sequences (often just two sequences).

For more information: https://en.wikipedia.org/wiki/Longest_common_subsequence


An alternative defintion of LCS that allows to compute the LENGTH of the LCS is the following: 
def compute_lcs(word1, word2):
    if word1 == "" or word2 == "":
        return 0

    return max(
        compute_lcs(word1, word2[:-1]), # horizontal move
        compute_lcs(word1[:-1], word2), # vertical move
        compute_lcs(word1[:-1], word2[:-1]) + (1 if word1[-1] == word2[-1] else 0)  # diagonal move
    )
"""


def parse_args():
    """Argument parser function. It allows to manage user input through terminal rather
    than at runtime."""

    parser = ap.ArgumentParser(
        description="""Algorithm that calculates the length of the longest
        common subsequence between two strings given as input."""
    )
    parser.add_argument("string1", help="first string")
    parser.add_argument("string2", help="second string")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="option to increase results detail"
    )

    args = parser.parse_args()

    return args


def compute_matrix(word1, word2):
    """Computes a matrix that tracks the longest common subsequence
    between every two substrings."""

    # added words to matrix for an easier output comprehension
    word1 = " " + "-" + word1
    word2 = " " + "-" + word2

    previous = {}
    matrix = [[0 for _ in range(len(word2))] for _ in range(len(word1))]

    matrix[0] = [char for char in word2]  # adds second word as first matrix row
    for index, row in enumerate(matrix):  # adds first as first matrix column
        row[0] = word1[index]

    for row in range(1, len(word1)):
        for col in range(1, len(word2)):
            if row == col == 1:
                matrix[row][col] = 0
                continue

            options = {}
            # for each direction, adds 1 if the characters are equal, 0 otherwise
            if row > 1:
                options[
                    int(matrix[row - 1][col]) + (1 if word1[row] == word2[col] else 0)
                ] = [(row - 1, col)]
            if col > 1:
                val = int(matrix[row][col - 1]) + (1 if word1[row] == word2[col] else 0)
                if val in options:
                    options[val].append((row, col - 1))
                else:
                    options[val] = [(row, col - 1)]
            if row > 1 and col > 1:
                val = int(matrix[row - 1][col - 1]) + (
                    1 if word1[row] == word2[col] else 0
                )
                if val in options:
                    options[val].append((row - 1, col - 1))
                else:
                    options[val] = [(row - 1, col - 1)]

            # we choose the direction from which we have the best case (maximum lcs length)
            max_value = max(options)
            # saving only the diagonal move is present, as saving more is unnecessary
            previous[(row, col)] = (
                (row - 1, col - 1)
                if (row - 1, col - 1) in options[max_value]
                else options[max_value][0]
            )

            matrix[row][col] = max_value  # lcs is defined as "longest" -> max

    return matrix, previous


def build_solution(matrix, previous, row, col):
    """Function that builds the solution given the LCS matrix and the "previous" dictionary as input."""
    if row == col == 1:  # base case
        return ""

    prev = previous[(row, col)]

    # recursive step
    return build_solution(matrix, previous, prev[0], prev[1]) + (
        matrix[0][col]
        if prev[0] == row - 1
        and prev[1] == col - 1
        and matrix[prev[0]][prev[1]] == matrix[row][col] - 1
        else ""
    )


def __main__():
    args = parse_args()  # argument parsing
    word1, word2 = (
        args.string1.upper(),
        args.string2.upper(),
    )  # strings on which to compute algorithm

    matrix, previous = compute_matrix(word1, word2)
    solution = build_solution(matrix, previous, len(word1) + 1, len(word2) + 1)

    # output management
    if args.verbose:
        print("LCS matrix:")
        for row in matrix:
            for element in row:
                if type(element) == str or element >= 0:
                    print(end=" ")
                print(f"{element}\t", end="")
            print()

        print()

    print(f'longest common subsequence: "{solution}"')


__main__()
