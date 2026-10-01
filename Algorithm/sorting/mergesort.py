import numpy as np


def mergesort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])

    return _merge(left, right)


def _merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def mergesort_inplace(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left < right:
        mid = (left + right) // 2
        mergesort_inplace(arr, left, mid)
        mergesort_inplace(arr, mid + 1, right)
        _merge_inplace(arr, left, mid, right)
    return arr


def _merge_inplace(arr, left, mid, right):
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1


if __name__ == "__main__":
    arr = [10, 7, 8, 9, 1, 5, 3, 2, 6, 4]

    print("Original:", arr)
    print("Sorted (simple):", mergesort(arr.copy()))
    print("Sorted (in-place):", mergesort_inplace(arr.copy()))

    np.random.seed(42)
    big_arr = np.random.randint(0, 1000, 100).tolist()
    print("\nBig array sorted:", mergesort(big_arr) == sorted(big_arr))
