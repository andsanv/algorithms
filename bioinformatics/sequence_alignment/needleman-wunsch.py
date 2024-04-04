import argparse as ap


"""
The Needleman-Wunsch algorithm is an algorithm used in bioinformatics to (globally) align protein
or nucleotide sequences.

For more information: https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
"""


def parse_args():
    """Argument parser function. It allows to manage user input through terminal rather
    than at runtime."""

    parser = ap.ArgumentParser(
        description="""Implementation of the Needleman-Wunsch algorithm, used to
        calculate global sequence alignment on two strings given as input."""
    )
    parser.add_argument(
        "string1", help="first string on which to calculate sequence alignment"
    )
    parser.add_argument(
        "string2", help="second string on which to calculate sequence alignment"
    )
    parser.add_argument(
        "-a",
        "--align",
        action="store_true",
        default=False,
        help="""if true, gap at the beginning and end of the sequences
        have a penalty of 0.
        """,
    )
    parser.add_argument(
        "-w",
        "--weights",
        nargs=3,
        metavar=("GAP", "MISMATCH", "MATCH"),
        default=(-1, -1, 1),
        help="""custom values for the weights of the algorithm. If not specified, 
        (-1, -1, 1) will be used""",
    )

    args = parser.parse_args()
    invalid_weights = False

    if int(args.weights[0]) > 0:
        invalid_weights = True
        print("error: invalid weight. Gap penalty must be non-positive.")
    if int(args.weights[1]) > 0:
        invalid_weights = True
        print("error: invalid weight. Mismatch penalty must be non-positive.")
    if int(args.weights[2]) < 0:
        invalid_weights = True
        print("error: invalid weight. Match score must be non-negative.")

    if invalid_weights:
        exit(1)

    return args


def find_previous(
    list: list[int], row: int, col: int
) -> tuple[int, list[tuple[int, int]]]:
    """Allows to find relationships between different cells of the matrix, which
    are used for backtracking. That's how solutions can be found after computation.

    Given a list of partial results and the coordinates of a cell, finds the
    maximum of those results and adds the related coordinates to the returned list.
    """

    if list == None or len(list) == 0:
        return None, None

    max_value = max([x for x in list if x is not None])
    indexes = []
    coords = []

    for i, v in enumerate(list):  # finds the indexes of the maximum values in the list
        if v == max_value:
            indexes.append(i)

    for index in indexes:  # based on the index, adds correspondent coords to list
        match index:
            case 0:
                coords.append((row - 1, col))
            case 1:
                coords.append((row, col - 1))
            case 2:
                coords.append((row - 1, col - 1))
            case other:
                exit(1)

    return max_value, coords


def compute(
    word1: str, word2: str, weights: list[int], align: bool
) -> tuple[list[list], dict[tuple[int, int], list[tuple[int, int]]], int]:
    """Core function that builds the Needleman-Wunsch matrix, which tracks the global
    sequence alignments for every substring of the strings given in input."""

    def is_move_on_margin(
        word1: str, word2: str, row1: int, col1: int, row2: int, col2: int
    ) -> bool:
        """Used when 'align' option enabled. Returns true if the move is on the margin
        of the matrix, false otherwise."""
        if row1 == row2:
            if row1 == 1 or row1 == len(word1) - 1:
                return True
        elif col1 == col2:
            if col1 == 1 or col1 == len(word2) - 1:
                return True
        else:
            return False

    word1 = " " + "-" + word1
    word2 = " " + "-" + word2

    matrix = [[0 for _ in range(len(word2))] for _ in range(len(word1))]
    previous = {}

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
            options.append(
                matrix[row - 1][col]
                + (
                    0
                    if align and is_move_on_margin(word1, word2, row, col, row - 1, col)
                    else weights["gap"]
                )
                if row > 1
                else None
            )
            # adds the cost of coming from left
            options.append(
                matrix[row][col - 1]
                + (
                    0
                    if align and is_move_on_margin(word1, word2, row, col, row, col - 1)
                    else weights["gap"]
                )
                if col > 1
                else None
            )
            # adds the cost (or reward) of coming from top-left
            if row > 1 and col > 1:
                penalty = (
                    weights["match"]
                    if word1[row] == word2[col]
                    else weights["mismatch"]
                )
                options.append(matrix[row - 1][col - 1] + penalty)
            else:  # used for the top row and the first column of the matrix
                options.append(None)

            matrix[row][col], prev = find_previous(options, row, col)
            previous[(row, col)] = prev

    return (matrix, previous, matrix[-1][-1])


def build_solutions(
    matrix: list[list],
    previous: dict[tuple[int, int], list[tuple[int, int]]],
    row: int,
    col: int,
) -> list[list[str]]:
    """Creates a visual representation of the distances between two words.
    Solutions may be multiple."""

    if row == col == 1:  # base case
        return [["", ""]]

    solutions = []
    for prev_coords in previous[(row, col)]:  # for every "antecedent"
        for s in build_solutions(matrix, previous, prev_coords[0], prev_coords[1]):
            if prev_coords[0] == row - 1 and prev_coords[1] == col:
                # adding a gap in word2
                s[0] += matrix[row][0]
                s[1] += "-"
            elif prev_coords[0] == row and prev_coords[1] == col - 1:
                # adding a gap in word1
                s[0] += "-"
                s[1] += matrix[0][col]
            else:
                # keeping words as they are
                s[0] += matrix[row][0]
                s[1] += matrix[0][col]

            solutions.append(s)  # solutions may be multiple

    return solutions


def print_results(matrix: list[list], result: int, solutions: list[list[str]]) -> None:
    """Auxiliary function used to format terminal output."""

    print(f"global sequence alignment matrix:")  # computed matrix
    for line in matrix:
        for element in line:
            if type(element) == str or element >= 0:
                print(end=" ")
            print(f"{element}\t", end="")
        print()

    print(f"\n\nalignment cost: {result}", end="\n\n\n")  # calculated "cost"

    print("possible solutions:")  # calculated solutions
    for i, solution in enumerate(solutions):
        print(f"{i + 1} -> \t", end="")
        for c in solution[0]:
            print(f"{c} ", end="")
        print("\n\t", end="")
        for c1, c2 in zip(solution[0], solution[1]):
            if c1 == "-" or c2 == "-":
                print("  ", end="")
            elif c1 == c2:
                print("| ", end="")
            else:
                print(". ", end="")

        print("\n\t", end="")
        for c in solution[1]:
            print(f"{c} ", end="")

        print(end="\n\n")

    return


def __main__() -> None:
    args = parse_args()  # argument parsing
    word1, word2 = (
        args.string1.upper(),
        args.string2.upper(),
    )  # strings on which to compute algorithm

    weights = {}  # dictionary to keep track of inputted weights
    weights["gap"], weights["mismatch"], weights["match"] = [
        int(arg) for arg in args.weights
    ]

    align = args.align

    matrix, previous, result = compute(word1, word2, weights, align)
    solutions = build_solutions(matrix, previous, len(word1) + 1, len(word2) + 1)

    print_results(matrix, result, solutions)

    return


__main__()
