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
        "-v", "--verbose", action="store_true", help="option to increase results detail"
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


def compute(
    word1: str, word2: str, weights: list[int]
) -> tuple[list[list], dict[int, list[list[str]]]]:
    """Core function that builds the Smith-Waterman matrix, which tracks the local
    sequence alignments for every substring of the strings given in input."""

    word1 = " " + "-" + word1
    word2 = " " + "-" + word2

    matrix = [[0 for _ in range(len(word2))] for _ in range(len(word1))]
    previous_coordinates = {}

    matrix[0] = [char for char in word2]  # adds second word as first matrix row
    for index, row in enumerate(matrix):  # adds first as first matrix column
        row[0] = word1[index]

    for row in range(1, len(word1)):
        for col in range(1, len(word2)):
            if row == col == 1:
                matrix[row][col] = 0
                continue

            options = {0: None}
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
                val = matrix[row - 1][col - 1] + (
                    weights["match"]
                    if word1[row] == word2[col]
                    else weights["mismatch"]
                )
                if val != 0:
                    if val in options:
                        options[val].append((row - 1, col - 1))
                    else:
                        options[val] = [(row - 1, col - 1)]

            matrix[row][col] = max(options)
            previous_coordinates[(row, col)] = options[max(options)]

    return matrix, previous_coordinates


def build_solutions(
    matrix: list[list],
    previous_coordinates: dict[tuple[int, int], list[tuple[int, int]]],
    word1: str,
    word2: str,
) -> dict[int, list[tuple[int, int]]]:
    """Builds the whole set of solutions of the local alignment of the two strings."""

    def find_starting_coords(matrix: list[list]) -> dict[int, list[tuple[int, int]]]:
        """Returns a dictionary that maps all matrix values to the their respective
        coordinates."""

        positions = {}  # maps matrix values (alignment scores) to coordinates

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if type(value) == int and value > 1:
                    if value in positions:
                        positions[value].append((i, j))
                    else:
                        positions[value] = [(i, j)]

        return positions

    def compute_solutions(
        matrix: list[list],
        previous_coordinates: dict[tuple[int, int], list[tuple[int, int]]],
        visited: list[list[bool]],
        row: int,
        col: int,
    ) -> list[list[str]]:
        """\"Backtracks\" the solution, or the path that brings to that specific (row, col) position.
        Many code lines may be unclear, but they only help to make output more readable.

        To make things clearer, here's an output example:
            Local alignment:
            ACCCTG      # (solution[0])
                ||      # (solution[1])
               TTGGT    # (solution[2])
        This allows to understand where the common sequence is located in both the inputted words.
        """

        visited[row][col] = True  # setting the current cell as visited

        if matrix[row][col] == 0:  # base case of the recursive function
            if (
                row > col
            ):  # starting index of the common sequence is further in first word
                first = word1[0 : row - 1]
                third = " " * (row - col) + word2[0 : col - 1]  # shifting word2
            elif (
                row < col
            ):  # starting index of the common sequence is further in second word
                first = " " * (col - row) + word1[0 : row - 1]  # shifting word1
                third = word2[0 : col - 1]
            else:  # sequence starts at the same index in both words
                first = word1[0 : row - 1]
                third = word2[0 : col - 1]

            second = " " * (max(row, col) - 1)
            # as the base case only tracks how the initial parts of the words are aligned,
            # only blank spaces are needed

            return [[first, second, third]]

        solutions = []  # solutions may be multiple
        for previous in previous_coordinates[(row, col)]:  # for every "antecedent"
            previous_row, previous_col = previous

            for solution in compute_solutions(
                matrix, previous_coordinates, visited, previous[0], previous[1]
            ):
                if previous_row == row - 1 and previous_col == col:
                    # if we added a gap in word2
                    solution[1] += (
                        " " if len(solution[2].strip()) != len(word2) else "*"
                    )
                    # solution[1] is a string that shows how the words are aligned
                    solution[0] += matrix[row][0]  # solution[0] is the first word
                    solution[2] += "-"  # solution[2] is the second word
                elif previous_row == row and previous_col == col - 1:
                    # if we added a gap in word1
                    solution[1] += (
                        " " if len(solution[0].strip()) != len(word1) else "*"
                    )
                    solution[0] += "-"
                    solution[2] += matrix[0][col]
                else:
                    # if we kept words as they were
                    solution[0] += matrix[row][0]
                    solution[1] += "|" if matrix[row][0] == matrix[0][col] else "*"
                    solution[2] += matrix[0][col]

                solutions.append(solution)

        return solutions

    solutions = {}  # dictionary that will contain solutions for every alignment score
    coordinates = find_starting_coords(matrix)
    visited = [[False for _ in range(len(word2) + 2)] for _ in range(len(word1) + 2)]
    # matrix to keep track of the visited cells

    # adds the final part to the strings
    for matrix_value in sorted(coordinates.keys(), reverse=True):   
        for row, col in coordinates[matrix_value]:
            if not visited[row][col]:
                for solution in compute_solutions(
                    matrix, previous_coordinates, visited, row, col
                ):
                    if len(word1) - row > len(word2) - col:
                        solution[0] += word1[row - 1 :]
                        solution[2] += word2[col - 1 :] + " " * (len(word1) - row + 1)
                    elif len(word1) - row < len(word2) - col:
                        solution[0] += word1[row - 1 :] + " " * (len(word2) - col + 1)
                        solution[2] += word2[col - 1 :]
                    else:
                        solution[0] += word1[row - 1 :]
                        solution[2] += word2[col - 1 :]
                    solution[1] += " " * (max(len(word1) - row, (len(word2) - col)) + 1)

                    if matrix_value in solutions:
                        solutions[matrix_value].append(solution)
                    else:
                        solutions[matrix_value] = [solution]

                visited[row][col] = True

    return solutions


def print_results(
    matrix: list[list], results: dict[int, list[list[str]]], verbose: bool
) -> None:
    """Auxiliary function used to format terminal output."""

    if verbose:
        print(f"local sequence alignment matrix:")  # prints computed matrix
        for line in matrix:
            for element in line:
                if type(element) == str or element >= 0:
                    print(end=" ")
                print(f"{element}\t", end="")
            print()

        i = 0

    if (
        len(results.keys()) == 0
    ):  # case where alignment has found no significant results
        if verbose:
            print()  # printing a newline for output formatting
        print("no significant solutions were found.")
    else:
        if (
            verbose
        ):  # prints all alignment scores found and the related possible solutions
            for alignment_score, solutions in results.items():
                print(f"\n\nalignment score: {alignment_score}")

                for solution in solutions:
                    i += 1
                    print(f"\n{i}  -->", end="")
                    for string in solution:
                        print(f"\t{string}")
        else:  # only prints the highest alignment score with its solutions
            print(f"highest alignment score: {max(results.keys())}")

            print("solutions:")
            for i, solution in enumerate(results[max(results.keys())]):
                print(f"\n{i + 1}  -->", end="")
                for string in solution:
                    print(f"\t{string}")

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

    matrix, previous_coordinates = compute(word1, word2, weights)
    solutions = build_solutions(matrix, previous_coordinates, word1, word2)

    print_results(matrix, solutions, args.verbose)

    return


__main__()