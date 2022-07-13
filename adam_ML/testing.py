import numpy as np

array1 = np.random.rand(5, 3)
array2 = np.random.rand(5, 3)
final_array = np.array([])
print(array1, array2)

for row1, row2 in zip(array1, array2):
    print(row1[0], row2[0])
    if row1[0] > row1[1] and row2[0] > row2[2]:
        np.append(final_array, 0)
    elif row1[0] > row1[1] and row2[0] < row2[2]:
        np.append(final_array, 2)
    elif row1[0] < row1[1] and row2[0] > row2[2]:
        np.append(final_array, 1)
    elif row1[0] < row1[1] and row2[0] < row2[2]:
        if row1[1] > row2[2]:
            np.append(final_array, 1)
        else:
            np.append(final_array, 2)