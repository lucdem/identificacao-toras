import math

import numpy as np
import cv2
import skimage
import scipy

from count_changes.count_changes import count_changes


def load_img(path):
	img = cv2.imread(path)
	return np.flip(img, 2) # opencv loads in BGR, order is flipped to convert to RGB


def hough_circle(img, min_radius, max_radius, radius_step, relative_threshold, normalize):
	radii = np.arange(min_radius, max_radius, radius_step)
	hough_space = skimage.transform.hough_circle(img, radii)

	# (accums, x, y, radii)
	return skimage.transform.hough_circle_peaks(hough_space, radii,
		threshold = relative_threshold * np.max(hough_space),
		min_xdistance = min_radius, min_ydistance = min_radius,
		normalize = normalize)


def draw_hough_circles(shape, hough_circle_res, max_circles = None):
	circle_canvas = np.zeros(shape)
	if max_circles is None:
		iter = zip(hough_circle_res[1], hough_circle_res[2], hough_circle_res[3])
	else:
		indexes = np.argsort(hough_circle_res[0])[-max_circles:]
		iter = zip(hough_circle_res[1][indexes], hough_circle_res[2][indexes], hough_circle_res[3][indexes])
	for center_x, center_y, radius in iter:
		circle_canvas[skimage.draw.circle_perimeter(center_y, center_x, radius, shape = circle_canvas.shape)] = 1
	# só pra ressaltar os contornos
	circle_canvas = skimage.morphology.dilation(circle_canvas)
	circle_canvas = skimage.morphology.dilation(circle_canvas)
	return circle_canvas


def draw_circles(shape, circles):
	circle_canvas = np.zeros(shape)
	for center_x, center_y, radius in circles:
		circle_canvas[skimage.draw.circle_perimeter(center_y, center_x, radius, shape = circle_canvas.shape)] = 1
	# só pra ressaltar os contornos
	circle_canvas = skimage.morphology.dilation(circle_canvas)
	circle_canvas = skimage.morphology.dilation(circle_canvas)
	return circle_canvas


def draw_hough_disks(shape, hough_circle_res, mask = None, mask_percentage = None, max_disks = None):
	disk_canvas = np.zeros(shape)

	if max_disks is None:
		indexes = np.flip(np.argsort(hough_circle_res[3])) # sort by highest radius
	else:
		indexes = np.argsort(hough_circle_res[0])[-max_disks:] # sort by highest accumulator value
	iter = zip(hough_circle_res[1][indexes], hough_circle_res[2][indexes], hough_circle_res[3][indexes])

	disks = []

	for center_x, center_y, radius in iter:
		if disk_canvas[center_y, center_x] == 1:
			continue	# ignora circulos cujo centro já está em outro circulo
		disk = skimage.draw.disk((center_y, center_x), radius, shape = disk_canvas.shape)

		if (mask is not None) and (mask_percentage is not None):
			if np.average(mask[disk]) < mask_percentage:
				continue

		disks.append((center_x, center_y, radius))
		disk_canvas[disk] = 1

	return disk_canvas, disks


def find_empty_square_sizes(matrix: np.ndarray):
	shape = matrix.shape
	res_matrix = np.ones(shape, dtype=np.int32)
	res_matrix[matrix != 0] = 0
	for i in range(1, shape[0]):
		for j in range(1, shape[1]):
			if matrix[i][j] == 1:
				res_matrix[i][j] = 0
			else:
				res_matrix[i][j] = np.min(np.ma.masked_array(res_matrix[i - 1:i + 1, j - 1:j + 1], mask=[[0, 0], [0, 1]])) + 1
	return res_matrix


def get_square_range(bottom_right_x, bottom_right_y, size):
	upper_left_x = bottom_right_x - (size - 1)
	upper_left_y = bottom_right_y - (size - 1)
	return slice(upper_left_x, bottom_right_x + 1), slice(upper_left_y, bottom_right_y + 1)


def mahalanobis_distance_from_mean(points, mean, covariance_matrix):
	covariance_inverse = np.linalg.inv(covariance_matrix)
	diff = points - mean
	return np.sum(diff @ covariance_inverse * diff, 1)


def normalize_0_to_1(arr):
	max = np.max(arr)
	min = np.min(arr)
	return (arr - min) / (max - min)


def full_identification_process(img):

	median_kernel_size = 21

	canny_low_threshold = 40
	canny_high_threshold = 70

	edges = __pre_processing(img, median_kernel_size, False, 0, 0, canny_low_threshold, canny_high_threshold)

	min_radius = 50
	max_radius = 150
	radius_step = 2
	hough_relative_threshold = 0.3
	hough_normalize = False

	hough_res = hough_circle(edges, min_radius, max_radius, radius_step, hough_relative_threshold, hough_normalize)
	res_canvas, disks = draw_hough_disks(edges.shape, hough_res, max_disks=2)

	distances_img = __img_mahalanobis(img, res_canvas)
	distance_threshold_mask = __distance_threshold_mask(distances_img)

	median_kernel_size = 11

	gauss_kernel_size = 11
	sigma = 1.8

	canny_low_threshold = 40
	canny_high_threshold = 70

	edges = __pre_processing(img, median_kernel_size, True, sigma, gauss_kernel_size, canny_low_threshold, canny_high_threshold)

	min_radius = 30
	max_radius = 180
	radius_step = 2
	hough_relative_threshold = 0.4
	hough_normalize = False

	hough_res = hough_circle(edges, min_radius, max_radius, radius_step, hough_relative_threshold, hough_normalize)
	res_canvas, disks = draw_hough_disks(edges.shape, hough_res, distance_threshold_mask, 0.7)

	return res_canvas, disks


def __pre_processing(img, median_kernel_size,
	gauss, sigma, gauss_kernel_size,
	canny_low_threshold, canny_high_threshold):

	blurred = np.copy(img)

	blurred = cv2.medianBlur(blurred, median_kernel_size)
	if gauss:
		blurred = cv2.GaussianBlur(blurred, (gauss_kernel_size, gauss_kernel_size), sigma)

	gray_img = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

	edges = cv2.Canny(gray_img, threshold1 = canny_low_threshold, threshold2 = canny_high_threshold, L2gradient = True)

	return edges


def __img_mahalanobis(img, selected_area):
	disks_on_img = np.copy(img)
	disks_on_img[selected_area == 0] = (0, 0, 0)

	img_HSV = np.copy(img)
	img_HSV = cv2.cvtColor(img_HSV, cv2.COLOR_RGB2HSV)
	disks_on_img_HSV = cv2.cvtColor(disks_on_img, cv2.COLOR_RGB2HSV)

	pixels = np.copy(disks_on_img_HSV[selected_area != 0])
	covar_matrix = np.cov(img_HSV.reshape((-1, 3)).T)
	distances = mahalanobis_distance_from_mean(img_HSV.reshape((-1, 3)), np.mean(pixels, 0), covar_matrix)
	distances = normalize_0_to_1(distances)
	return distances.reshape(selected_area.shape)


def __distance_threshold_mask(distances_img):
	threshold_mask = np.zeros(distances_img.shape)
	threshold_mask[distances_img < 0.15] = 1

	kernel_size = 15
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel)
	kernel_size = 15
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	return cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)


def full_age_counting_process(img, disk):
	polar = cv2.linearPolar(img, (int(disk[0]), int(disk[1])), disk[2], cv2.WARP_FILL_OUTLIERS)
	distances_img = __img_mahalanobis(polar, polar[:, 0:500, :])

	threshhold_mask = np.zeros(distances_img.shape)
	threshhold_mask[distances_img < 0.05] = 1

	kernel_size = 15
	threshhold_mask = img_open(threshhold_mask, kernel_size)
	threshhold_mask = img_close(threshhold_mask, kernel_size)

	threshhold_img = np.copy(polar)
	threshhold_img[threshhold_mask != 1] = [0, 0, 0]

	gray_polar = cv2.cvtColor(threshhold_img, cv2.COLOR_RGB2GRAY)

	equalized = cv2.equalizeHist(gray_polar)
	equalized = normalize_0_to_1(equalized)

	equalized_threshold = np.copy(equalized)
	equalized_threshold[equalized < 0.5] = 0
	equalized_threshold[0.5 < equalized] = 1

	kernel_size = 5
	final = img_close(equalized_threshold)
	final = img_open(final, kernel_size)

	n = 30
	changes = count_changes(final.astype(np.int32), n)
	# std_dev = np.std(changes)
	mean = np.mean(changes)
	# median = np.median(changes)
	# mode = np.mean(scipy.stats.mode(changes)[0]) # em caso de mais de uma moda, toma media das modas
	return math.floor((mean - 1) / 2)


def rows_moving_avg(matrix, n):
	return scipy.signal.convolve2d(matrix, np.ones((n, 1)), 'same') / n


def img_open(img, kernel_size):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def img_close(img, kernel_size):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)