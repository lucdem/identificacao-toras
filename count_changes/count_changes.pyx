#cython: language_level=3

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)


def count_changes(matrix: np.ndarray, int number_of_lines):
	cdef int step = matrix.shape[0] // number_of_lines
	chosen_lines = np.arange(0, matrix.shape[0], step)
	cdef int n = chosen_lines.size
	cdef int i
	changes = np.zeros(n)
	for i in range(n):
		changes[i] = _count_changes(matrix[i, :])
	return changes


cdef _count_changes(int[:] line):
	cdef int i
	cdef int last_value = 0
	cdef int changes = 0

	for i in range(line.size):
		if line[i] != last_value:
			changes += 1
			last_value = line[i]
	return changes