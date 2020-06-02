import numpy as np


def intersection_area(box1, box2):
	"""
	intersection area of two rectangular boxes
	:param box1: (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values
	:param box2: (b2x1, b2y1, b2x2, b2y2) - tuple/list with numeric values
	:return: numeric value of intersection area
	"""

	b1x1, b1y1, b1x2, b1y2 = box1
	b2x1, b2y1, b2x2, b2y2 = box2

	left   = max(b1x1, b2x1)
	top    = max(b1y1, b2y1)
	right  = min(b1x2, b2x2)
	bottom = min(b1y2, b2y2)

	if left < right and top < bottom:
		return (right - left) * (bottom - top)
	else:
		return 0

def union_area(box1, box2):
	"""
	union area of two rectangular boxes
	:param box1: (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param box2: (b2x1, b2y1, b2x2, b2y2) - tuple/list with numeric values, or a 1D np.ndarray
	:return: numeric value of union area
	"""

	b1x1, b1y1, b1x2, b1y2 = box1
	b2x1, b2y1, b2x2, b2y2 = box2

	area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
	area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
	inter_area = intersection_area(box1, box2)

	return area1 + area2 - inter_area

def IoU(box1, box2):
	"""
	intersection over union of two rectangular boxes
	:param box1: (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param box2: (b2x1, b2y1, b2x2, b2y2) - tuple/list with numeric values, or a 1D np.ndarray
	:return: numeric value of intersection over union
	"""
	i = intersection_area(box1, box2)
	u = union_area(box1, box2)

	return i/u

def AP(boxes_gt, boxes_pred, iou_threshold=0.5, scores=None, resulting_score_threshold=0.):
	"""
	Average precision (AP)
	:param boxes_gt: GROUND TRUTH (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param boxes_pred: PREDICTED (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param iou_threshold: minimal value for iou of two boxes to count them as match
	:param scores: scores for predited boxes in the same order
	:param resulting_score_threshold: minimal value for iou*score of two boxes to count them as match
	:return: numeric value of average precision
	"""
	L_gt = len(boxes_gt)
	L_pr = len(boxes_pred)
	iou_mat = np.zeros(shape=(L_gt, L_pr), dtype=np.float)
	for b1, box1 in enumerate(boxes_gt):
		for b2, box2 in enumerate(boxes_pred):
			iou_mat[b1, b2] = IoU(box1, box2)
	iou_mat[iou_mat < iou_threshold] = 0.

	if scores:
		iou_mat *= np.array(scores, dtype=np.float)
		iou_mat[iou_mat < resulting_score_threshold] = 0.

	nb_nonzero = np.count_nonzero(iou_mat)

	# get list of matches sorted by (iou*score) descending (Thanks to Ashwini Chaudhary: https://stackoverflow.com/a/30577520/11304602)
	sorted_matches = np.argsort(-iou_mat, axis=None) # because we need DESCENDING order, we use -iou_mat
	sorted_matches = np.unravel_index(sorted_matches, shape=(L_gt, L_pr))
	sorted_matches = np.dstack(sorted_matches)
	sorted_matches = sorted_matches[:nb_nonzero]

	# distribute boxes between TP, FP, FN
	TP = 0
	available_gt = list(range(L_gt))
	available_pr = list(range(L_pr))
	for gt_i, pr_i in sorted_matches:
		if (gt_i in available_gt) and (pr_i in available_pr):
			TP += 1
			available_gt.remove(gt_i)
			available_pr.remove(pr_i)
	FP = len(available_pr)
	FN = len(available_gt)

	AP = TP / (TP + FP + FN)
	return AP

def mAP(boxes_gt, boxes_pred, iou_thresholds=[0.5]):
	"""
	Average precision (AP)
	:param boxes_gt: GROUND TRUTH (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param boxes_pred: PREDICTED (b1x1, b1y1, b1x2, b1y2) - tuple/list with numeric values, or a 1D np.ndarray
	:param iou_thresholds: list of iou_threshold values (iou_threshold is a minimal value for iou of two boxes to count them as match)
	:return: numeric value of mean average precision
	"""
	L_gt = len(boxes_gt)
	L_pr = len(boxes_pred)
	iou_mat = np.zeros(shape=(L_gt, L_pr), dtype=np.float)
	for b1, box1 in enumerate(boxes_gt):
		for b2, box2 in enumerate(boxes_pred):
			iou_mat[b1, b2] = IoU(box1, box2)

	# get list of matches sorted by iou descending (Thanks to Ashwini Chaudhary: https://stackoverflow.com/a/30577520/11304602)
	sorted_matches = np.argsort(-iou_mat, axis=None) # because we need DESCENDING order, we use -iou_mat
	sorted_matches = np.unravel_index(sorted_matches, shape=(L_gt, L_pr))
	sorted_matches = np.dstack(sorted_matches)[0]

	AP_sum = 0

	for t in iou_thresholds:
		nb_nonzero = np.count_nonzero(np.greater_equal(iou_mat, t))
		sorted_matches = sorted_matches[:nb_nonzero]

		# distribute boxes between TP, FP, FN
		TP = 0
		available_gt = list(range(L_gt))
		available_pr = list(range(L_pr))
		for gt_i, pr_i in sorted_matches:
			if (gt_i in available_gt) and (pr_i in available_pr):
				TP += 1
				available_gt.remove(gt_i)
				available_pr.remove(pr_i)
		FP = len(available_pr)
		FN = len(available_gt)

		AP = TP / (TP + FP + FN)
		AP_sum += AP

	mAP = AP_sum / len(iou_thresholds)
	return mAP

'''
Example:
mAP(boxes_gt=[[10, 11, 25, 18], [110, 111, 125, 118]],
   boxes_pred=[[11, 10, 23, 17], [8, 8, 20, 15], [111, 110, 122, 117]],
   iou_thresholds=[0.1, 0.3, 0.5, 0.9])


'''

# boxes in form: (xmin, ymin, w, h)
gts = [[954, 391,  70,  90],
       [660, 220,  95, 102],
       [ 64, 209,  76,  57],
       [896,  99, 102,  69],
       [747, 460,  72,  77],
       [885, 163, 103,  69],
       [514, 399,  90,  97],
       [702, 794,  97,  99],
       [721, 624,  98, 108],
       [826, 512,  82,  94],
       [883, 944,  79,  74],
       [247, 594, 123,  92],
       [673, 514,  95, 113],
       [829, 847, 102, 110],
       [ 94, 737,  92, 107],
       [588, 568,  75, 107],
       [158, 890, 103,  64],
       [744, 906,  75,  79],
       [826,  33,  72,  74],
       [601,  69,  67,  87]]
preds = [[956, 409, 68, 85],
		 [883, 945, 85, 77],
		 [745, 468, 81, 87],
		 [658, 239, 103, 105],
		 [518, 419, 91, 100],
		 [711, 805, 92, 106],
		 [62, 213, 72, 64],
		 [884, 175, 109, 68],
		 [721, 626, 96, 104],
		 [878, 619, 121, 81],
		 [887, 107, 111, 71],
		 [827, 525, 88, 83],
		 [816, 868, 102, 86],
		 [166, 882, 78, 75],
		 [603, 563, 78, 97],
		 [744, 916, 68, 52],
		 [582, 86, 86, 72],
		 [79, 715, 91, 101],
		 [246, 586, 95, 80],
		 [181, 512, 93, 89],
		 [655, 527, 99, 90],
		 [568, 363, 61, 76],
		 [9, 717, 152, 110],
		 [576, 698, 75, 78],
		 [805, 974, 75, 50],
		 [10, 15, 78, 64],
		 [826, 40, 69, 74],
		 [32, 983, 106, 40]]
# convert to form: (xmin, ymin, xmax, ymax)
for one in gts:
	one[2] += one[0]
	one[3] += one[1]
for one in preds:
	one[2] += one[0]
	one[3] += one[1]

# TODO: doesn't match with last result there: https://www.kaggle.com/pestipeti/competition-metric-details-script
print(mAP(boxes_gt=gts, boxes_pred=preds, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75]))







