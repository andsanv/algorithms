import argparse as ap
import random



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
                options[int(matrix[row - 1][col]) + (1 if word1[row] == word2[col] else 0)] = [(row - 1, col)]
            if col > 1:
                val = int(matrix[row][col - 1]) + (1 if word1[row] == word2[col] else 0)
                if val in options:
                    options[val].append((row, col - 1))
                else:
                    options[val] = [(row, col - 1)]
            if row > 1 and col > 1:
                val = int(matrix[row - 1][col - 1]) + (1 if word1[row] == word2[col] else 0)
                if val in options:
                    options[val].append((row - 1, col - 1))
                else:
                    options[val] = [(row - 1, col - 1)]    

            max_value = max(options)
            prev = options[max_value]

            previous[(row, col)] = prev
            matrix[row][col] = max_value # lcs is defined as "best case" -> max

    return matrix, previous


def compute_lcs(word1, word2):
    if word1 == "" or word2 == "":
        return 0

    return max(
        compute_lcs(word1, word2[:-1]),
        compute_lcs(word1[:-1], word2),
        compute_lcs(word1[:-1], word2[:-1]) + (1 if word1[-1] == word2[-1] else 0)
    )


def build_solution(matrix, previous, row, col):
    if row == col == 1:
        return ""
    
    prev = (row - 1, col - 1) if (row - 1, col - 1) in previous[(row, col)] else previous[(row, col)][0]
    
    return build_solution(matrix, previous, prev[0], prev[1]) + (
        matrix[0][col] if prev[0] == row - 1 and prev[1] == col - 1 and matrix[prev[0]][prev[1]] == matrix[row][col] - 1 else "" 
    )


def __main__():
    args = parse_args()  # argument parsing
    word1, word2 = (
        args.string1.upper(),
        args.string2.upper(),
    )  # strings on which to compute algorithm

    # matrix, lcs = (
    #     compute_matrix(word1, word2)
    #     if args.verbose
    #     else [None, compute_lcs(word1, word2)]
    # )

    matrix, previous = compute_matrix(word1, word2)

    solution = build_solution(matrix, previous, len(word1) + 1, len(word2) + 1)

    print(f"longest commons subsequence: \"{solution}\"")

    # for row in matrix:
    #     print(row)

    # for key, val in previous.items():
    #     print(key, val, sep=": ")


__main__()