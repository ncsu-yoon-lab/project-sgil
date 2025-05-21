from itertools import product

lists = [
    [(1.2, 4.5), (4.3, 2.3), (3.4, 9.9)],
    [(1.2, 4.5), (4.1, 2.3), (4.1, 0.3)],
    [(4.1, 2.3), (4.0, 2.3)],
]

# Generate all combinations taking one element from each sublist
all_combinations = product(*lists)

# Filter out combinations with duplicate elements
unique_combinations = [combo for combo in all_combinations if len(set(combo)) == len(combo)]

# Print the result
for combo in unique_combinations:
    print(combo)
