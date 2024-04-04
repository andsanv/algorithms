import argparse as ap


"""
The Smith-Waterman algorithm performs local sequence alignment; that is, for
determining similar regions between two strings of nucleic acid sequences or protein
sequences. Instead of looking at the entire sequence, the Smith-Waterman algorithm
compares segments of all possible lengths and optimizes the similarity measure.

For more information: https://en.wikipedia.org/wiki/Smith-Waterman_algorithm
"""


def parse_args():
    """Argument parser function. It allows to manage user input through terminal rather
    than at runtime."""

    parser = ap.ArgumentParser(
        description="""Implementation of the Smith-Waterman algorithm, used to
        calculate local sequence alignment on two strings given as input."""
    )
    parser.add_argument(
        "string1", help="first string on which to calculate sequence alignment"
    )
    parser.add_argument(
        "string2", help="second string on which to calculate sequence alignment"
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

    return args


def find_previous(list, row, col) -> tuple[int, list[tuple[int, int]]]:
    """Allows to find relationships between different cells of the matrix, which
    are used for backtracking. That's how solutions can be found after computation.

    Given a list of partial results and the coordinates of a cell, finds the
    maximum of those results and adds to the list to return the related coordinates.
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


def compute(word1, word2, weights) -> tuple[list[list], dict, int]:
    """Core function that builds the Smith-Waterman matrix, which tracks the local
    sequence alignments for every substring of the strings given in input."""

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

            options = { 0: None }
            # adds the cost of coming from top
            if row > 1:
                val = int(matrix[row - 1][col]) + weights["gap"]
                if val != 0:
                    options[val] = [(row - 1, col)]
            # adds the cost of coming from left
            if col > 1:
                val = int(matrix[row][col - 1]) + weights["gap"]
                if val != 0:
                    if val in options:
                        options[val].append((row, col - 1))
                    else:
                        options[val] = [(row, col - 1)]
            # adds the cost (or reward) of coming from top-left
            if row > 1 and col > 1:
                val = matrix[row - 1][col - 1] + (weights["match"] if word1[row] == word2[col] else weights["mismatch"])
                if val != 0:
                    if val in options:
                        options[val].append((row - 1, col - 1))
                    else:
                        options[val] = [(row - 1, col - 1)]

            matrix[row][col] = max(options)
            previous[(row, col)] = options[max(options)]

    return matrix, previous
    

    
def build_solutions(matrix: list[list], previous: dict, word1: str, word2: str):
    """Creates a visual representation of the local alignment results.
    Solutions may be multiple."""

    def find_starting_points(matrix: list[list]):
        positions = {}

        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] > 1:
                    if matrix[i][j] in positions:
                        positions[matrix[i][j]].append((i, j))
                    else:
                        positions[matrix[i][j]] = [(i, j)]

        return positions

    def compute_solutions(matrix, previous, row, col):
        if matrix[row][col] == 0:  # base case
            if row > col:
                first = word1[0:row + 1]
                third = ' ' * (row - col) + word2[0:col + 1]
            elif row < col:
                first = ' ' * (col - row) + word1[0:row + 1]
                third = word2[0:col + 1]
            else:
                first = word1[0:row + 1]
                third = word2[0:col + 1]

            second = ' ' * max(row,col)

            return [[first, second, third]]

        solutions = []
        for prev_coords in previous[(row, col)]:  # for every "antecedent"
            for s in compute_solutions(matrix, previous, prev_coords[0], prev_coords[1]):
                if prev_coords[0] == row - 1 and prev_coords[1] == col:
                    # adding a gap in word2
                    s[0] += matrix[row][0]
                    s[1] += ' '
                    s[2] += "-"
                elif prev_coords[0] == row and prev_coords[1] == col - 1:
                    # adding a gap in word1
                    s[0] += "-"
                    s[1] += ' '
                    s[2] += matrix[0][col]
                else:
                    # keeping words as they are
                    s[0] += matrix[row][0]
                    s[1] += '|' if matrix[row][0] == matrix[0][col] else '*'
                    s[2] += matrix[0][col]

                solutions.append(s)  # solutions may be multiple

        print(f"solutions = {solutions}")
        return solutions

    solutions = {}
    positions = find_starting_points(matrix)
    
    visits = [[False for _ in range(len(word2) + 2)] for _ in range(len(word1) + 2)]

    first = ""; second = ""; third = "" # first is word1, third is word2,
                                        # second is a string containing symbols to
                                        # better understand the alignment
    
    for key in positions:
        for (row, col) in positions[key]:
            if visits[row][col] == False:
                print(f"row = {row}, col = {col}")
                for s in compute_solutions(matrix, previous, row, col):
                    if len(word1) - row > len(word2) - col:
                        first = s[0] + word1[row + 1:]
                        third = s[2] + word2[col + 1:] + ' ' * (len(word1) - row)
                    elif len(word1) - row < len(word2) - col:
                        first = s[0] + word1[row + 1:] + ' ' * (len(word2) - col)
                        third = s[2] + word2[col + 1:]
                    else:
                        first = s[0] + word1[row + 1:]
                        third = s[2] + word2[col + 1:]
                    second = s[1] + ' ' * max(len(word1) - row, (len(word2) - col))

                    if key in solutions:
                        solutions[key].append(s)
                    else:
                        solutions[key] = [s]
    
                visits[row][col] = True

    return solutions
            


def print_results(matrix, result, solutions) -> None:
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
        print(f"{i + 1}  --> \t", end="")
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

        print()

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

    matrix, previous = compute(word1, word2, weights)
    solutions = build_solutions(matrix, previous, word1, word2)

    # for key, val in solutions.items():
    #     print(key, ":", sep=" ")
    #     for el in val:
    #         for inner in el:
    #             print("\t", inner)

    # print_results(matrix, result, solutions)

    return


__main__()
