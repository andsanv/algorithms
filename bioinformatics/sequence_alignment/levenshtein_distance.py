import argparse as ap


"""
The Levenshtein distance is a string metric for measuring the
difference between two sequences. Informally, the Levenshtein
distance between two words is the minimum number of
single-character edits (insertions, deletions or
substitutions) required to change one word into the other.

For more information: https://en.wikipedia.org/wiki/Levenshtein_distance
"""


def parse_args():
    """Argument parser function. It allows to manage user input through terminal rather
    than at runtime."""

    parser = ap.ArgumentParser(
        description="""Implementation of the Levenshtein algorithm, used to
        calculate the (edit) distance between two strings given as input."""
    )
    parser.add_argument("string1", help="first string")
    parser.add_argument("string2", help="second string")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="option to increase results detail"
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs=3,
        metavar=("GAP", "MISMATCH", "MATCH"),
        default=(1, 1, 0),
        help="""custom values for the weights of the algorithm. If not specified, 
        (1, 1, 0) will be used""",
    )

    args = parser.parse_args()

    return args


def compute_distance(str1: str, str2: str, weights: list[int]) -> int:
    """Computes the levenshtein distance between the given strings."""

    # if one string is empty, distance equals number of gaps times gap penalty
    if len(str1) == 0:
        return len(str2) * weights["gap"]
    if len(str2) == 0:
        return len(str1) * weights["gap"]
    if len(str1) == len(str2) == 1:
        return weights["match"] if str1 == str2 else weights["mismatch"]

    return min(
        weights["gap"] + compute_distance(str1, str2[:-1], weights),
        weights["gap"] + compute_distance(str1[:-1], str2, weights),
        (
            weights["match"] + compute_distance(str1[:-1], str2[:-1], weights)
            if str1[-1] == str2[-1]
            else weights["mismatch"] + compute_distance(str1[:-1], str2[:-1], weights)
        ),
    )


def compute_matrix(
    word1: str, word2: str, weights: list[int]
) -> tuple[list[list], int]:
    """Computes a matrix that tracks Levenshtein distance between every two substrings."""

    # added words to matrix for an easier output comprehension
    word1 = " " + "-" + word1
    word2 = " " + "-" + word2

    matrix = [[0 for _ in range(len(word2))] for _ in range(len(word1))]

    matrix[0] = [char for char in word2]  # adds second word as first matrix row
    for index, row in enumerate(matrix):  # adds first as first matrix column
        row[0] = word1[index]

    for row in range(1, len(word1)):
        for col in range(1, len(word2)):
            if row == col == 1:
                matrix[row][col] = 0
                continue

            options = []
            # adds the cost of coming from top
            if row > 1:
                options.append(matrix[row - 1][col] + weights["gap"])
            # adds the cost of coming from left
            if col > 1:
                options.append(matrix[row][col - 1] + weights["gap"])
            # adds the cost (or reward) of coming from top-left
            if row > 1 and col > 1:
                penalty = (
                    weights["match"]
                    if word1[row] == word2[col]
                    else weights["mismatch"]
                )
                options.append(matrix[row - 1][col - 1] + penalty)

            matrix[row][col] = min(options)  # distance is defined as "best case" -> min

    return matrix, matrix[-1][-1]


def __main__():
    args = parse_args()

    word1, word2 = args.string1, args.string2

    weights = {}  # dictionary to keep track of inputted weights
    weights["gap"], weights["mismatch"], weights["match"] = [
        int(arg) for arg in args.weights
    ]

    matrix, distance = (
        compute_matrix(word1, word2, weights)
        if args.verbose
        else [None, compute_distance(word1, word2, weights)]
    )

    if args.verbose:
        print("distance matrix:")
        for line in matrix:
            for element in line:
                print(f"{element}\t", end="")
            print()

        print()
    else:
        print(
            "note: execute this script with verbose '-v' argument to also show distances matrix.",
            end="\n\n",
        )

    print("levenshtein (edit) distance:", distance)


__main__()
