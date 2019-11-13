import numpy as np


# rotate a tour to when the most ordered pairs occurs
def reorder(tour):
    n = tour.shape[0]
    count = sum((tour < tour[i])[:i].sum() for i in range(n))

    # find the split
    total = n * (n - 1) / 2
    best_count = count
    best_split = 0
    rev = False
    for i in range(n):
        if best_count < count:
            best_count = count
            best_split = i
            rev = False
        if best_count < total - count:
            best_count = total - count
            best_split = i
            rev = True
        count += 2 * tour[i] + 1 - n

    # reorder the tour
    tour = np.concatenate((tour[best_split:], tour[:best_split]))
    if rev:
        tour = tour[::-1]
    return tour
