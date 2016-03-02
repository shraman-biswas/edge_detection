import cv2
import numpy as np
import sys


def colour_norm(img):
	sum_img = np.int0(img[:,:,0]) + np.int0(img[:,:,1]) + np.int0(img[:,:,2])
	sum_img = np.dstack([sum_img, sum_img, sum_img])
	img = ((255 * img.astype("int64")) / sum_img).astype("uint8")
	return img


def get_bbox(cnt, img):
	rect = cv2.minAreaRect(cnt)
	bbox = cv2.boxPoints(rect)
	bbox = np.int0(bbox)
	#cv2.drawContours(img, [bbox], 0, (0,255,0), 2)
	# rotate bbox
	M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
	pts = np.ones((4, 3))
	pts[:,:-1] = bbox
	bbox_rot = np.int0(np.dot(pts, M.T))
	# resize bbox
	bbox_rot[0][0] -= 15
	bbox_rot[0][1] += 120
	bbox_rot[1][0] -= 15
	bbox_rot[2][0] += 5
	bbox_rot[3][0] += 5
	bbox_rot[3][1] += 120
	# rotate back bbox
	p = (bbox_rot[1][0], bbox_rot[1][1])
	M = cv2.getRotationMatrix2D(p, -rect[2], 1)
	pts = np.ones((4, 3))
	pts[:,:-1] = bbox_rot
	bbox = np.int0(np.dot(pts, M.T))
	return bbox


def main():
	print "[ opencv edge detection ]"

	# get image filename
	img_filename = sys.argv[1] if len(sys.argv) > 1 else "image.png"

	# load image
	img = cv2.imread(img_filename)
	if img is None:
		print "image (%s) could not be loaded!" % img_filename
		sys.exit()

	# crop out logo
	img_h, img_w, img_d = img.shape
	img = img[:img_h - 180,:]

	# image processing
	norm_img = colour_norm(img)
	hsv_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
	low_thresh = (150, 100, 0)
	upp_thresh = (200, 255, 255)
	detect_mask = cv2.inRange(hsv_img, low_thresh, upp_thresh)
	detect_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], \
						dtype=np.uint8)
	detect_mask = cv2.morphologyEx(detect_mask, \
									cv2.MORPH_OPEN, \
									detect_kernel)
	detect_img = cv2.bitwise_and(img, img, mask=detect_mask)

	# bbox
	bbox_kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 1, 1, 1, 0, 0],
							[0, 1, 1, 1, 1, 1, 0], 
							[1, 1, 1, 1, 1, 1, 1],
							[0, 1, 1, 1, 1, 1, 0], 
							[0, 0, 1, 1, 1, 0, 0], 
							[0, 0, 0, 1, 0, 0, 0]], \
							dtype=np.uint8)
	bbox_mask = cv2.morphologyEx(detect_mask, \
							cv2.MORPH_DILATE, \
							bbox_kernel,
							iterations=2)
	#bbox_img = cv2.bitwise_and(img, img, mask=bbox_mask)

	# contours
	res_img = img.copy()
	cnt_img, contours, hier = cv2.findContours(bbox_mask, \
										cv2.RETR_TREE, \
										cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		cnt = contours[0]
		#cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
		bbox = get_bbox(cnt, res_img)
		# create bbox mask image
		res_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
		cv2.drawContours(res_mask, [bbox], 0, 255, 5)
		# draw bbox
		cv2.drawContours(res_img, [bbox], 0, (255,0,0), 2)

	# normal edge detection
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edge_img = cv2.Canny(gray_img, 20, 50)
	ext_img = cv2.bitwise_and(edge_img, edge_img, mask=res_mask)

	'''	
	p1, p2, p3, p4 = bbox
	margin = 5
	for r in range(p3[1] - margin, p1[1] + margin):
		for c in range(p2[0] - margin, p4[0] + margin):
			if ext_img[r, c]
	'''

	# display result
	cv2.imshow("result", res_img)
	cv2.imwrite("result.png", res_img)
	cv2.waitKey(0)


if __name__ == "__main__":
	main()
