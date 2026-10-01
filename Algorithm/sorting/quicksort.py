import numpy as np


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = _partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)
    return arr


def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


if __name__ == "__main__":
    arr = [10, 7, 8, 9, 1, 5, 3, 2, 6, 4]

    print("Original:", arr)
    print("Sorted (simple):", quicksort(arr.copy()))
    print("Sorted (in-place):", quicksort_inplace(arr.copy()))

    np.random.seed(42)
    big_arr = np.random.randint(0, 1000, 100).tolist()
    print("\nBig array sorted:", quicksort(big_arr) == sorted(big_arr))

