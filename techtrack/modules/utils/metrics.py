import numpy as np
from collections import defaultdict


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    The IoU is a metric used to evaluate how much two bounding boxes overlap. It is computed
    as the ratio between the area of overlap (intersection) and the total area covered by the
    two boxes (union). This metric is widely used in object detection tasks to determine the
    quality of predicted bounding boxes with respect to the ground truth.

    The computation is performed as follows:
      1. Extract the x and y coordinates along with width and height for both bounding boxes.
      2. Determine the coordinates of the intersection rectangle
      3. Compute the width and height of the intersection region as the difference between these coordinates.
      4. If the computed width or height is negative, it indicates no overlap; in such cases, the intersection area is set to zero.
      5. Calculate the area of the intersection region by multiplying the width and height.
      6. Compute the area of each bounding box individually.
      7. Calculate the union area as the sum of the two bounding box areas minus the intersection area.
      8. Finally, compute the IoU by dividing the intersection area by the union area. If the union area is zero,
         the function returns 0 to avoid division by zero.

    Parameters
    ----------
    boxA : tuple
        A tuple of four numbers representing the first bounding box in the format (x, y, w, h),
        where (x, y) represents the top-left corner, and w and h represent the width and height.
    boxB : tuple
        A tuple of four numbers representing the second bounding box in the format (x, y, w, h).

    Returns
    -------
    float
        The IoU value, which is the ratio of the intersection area over the union area.
        The value ranges from 0 to 1, where 0 indicates no overlap and 1 indicates perfect overlap.
    """
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    
    # Convert to (x_min, y_min, x_max, y_max) format
    # In OpenCV, the origin (0,0) is at the top-left corner of the image
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1 
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return float(inter_area) / union_area if union_area > 0 else 0.0


def evaluate_detections(boxes, classes, scores, cls_scores, gt_boxes, gt_classes, map_iou_threshold, num_classes=20, eval_type="class_scores"):
    """
    Evaluate detections by matching predicted bounding boxes with ground truth boxes and generate
    corresponding true labels and prediction scores for further evaluation (e.g., computing mAP).

    This function processes multiple images and performs the following steps for each image:
      1. Extract detection and ground truth data.
      2. Compute an IoU (Intersection over Union) matrix between each detected box and each ground truth box.
      3. For each detection, determine the maximum IoU with any ground truth box.
      4. Identify valid matches.
      5. Assign prediction scores and true labels for:
         - True Positives (TP)
         - False Negatives (FN)
         - False Positives (FP)

    The function supports three evaluation modes specified by the `eval_type` parameter:
      - "objectness": Use objectness scores deri ved from the `scores` parameter.
      - "class_scores": Use classification scores provided in the `cls_scores` parameter.
      - "combined": Use the element-wise product of the objectness and classification scores.

    Parameters
    ----------
    boxes : list
        A list of detected bounding boxes for each image. Each element of the list corresponds to one image
        and is itself a list of tuples, with each tuple representing a detection box in the format (x, y, w, h) 
        as (top-left x, top-left y, width, height).
    classes : list
        A list of detected class labels for each image. Each element is a list of class labels corresponding
        to the detection boxes in the same image.
    scores : list
        A list of detection confidence scores for each image. Each element is a list of confidence scores
        corresponding to the detection boxes in that image.
    cls_scores : list
        A list of classification scores for detected objects for each image. Each element is a list (or array)
        of classification scores (or score vectors) associated with the detections.
    gt_boxes : list
        A list of ground truth bounding boxes for each image. Each element is a list of tuples, with each tuple
        representing a ground truth box in the format (x, y, w, h) as (top-left x, top-left y, width, height).
    gt_classes : list
        A list of ground truth class labels for each image. Each element is a list of labels corresponding to the
        ground truth boxes in that image.
    map_iou_threshold : float
        The IoU threshold used to determine whether a detection matches a ground truth box.
    eval_type : str, optional
        The type of evaluation to perform, which determines which scores are used for predictions.
        Options are:
          - "objectness": Use the objectness scores derived from the `scores` parameter.
          - "class_scores": Use the classification scores provided in `cls_scores`.
          - "combined": Use the element-wise product of the objectness and classification scores. (See example below)
        Default is "class_scores".

        Example
        -------
        >>> if eval_type == "objectness":
        >>>     y_pred = score
        >>> elif eval_type == "class_scores":
        >>>     y_pred = cls_score
        >>> elif eval_type == "combined":
        >>>     y_pred = np.multiply(score, cls_score)

    Returns
    -------
    y_true : list
        A list of true labels corresponding to each detection or ground truth match. For detections that match
        a ground truth box, the true label is taken from the ground truth. For false negatives (missed detections),
        the corresponding ground truth label is added.
    pred_scores : list
        A list of predicted scores corresponding to the labels in `y_true`. The scores are derived based on the
        selected evaluation type ("objectness", "class_scores", or "combined"). Dummy scores are assigned for false negatives.
    
    Notes
    -----
    - The function uses IoU matching to determine whether a detection sufficiently overlaps with a ground truth box.
    - The specific handling of scores (e.g., weighting by objectness and/or classification) is determined by the eval_type.

    Example
    -------
    >>> boxes = [[(50, 50, 100, 100), (200, 200, 80, 80)]]
    >>> classes = [[0, 1]]
    >>> scores = [[0.9, 0.85]]
    >>> cls_scores = [[0.95, 0.88]]
    >>> gt_boxes = [[(48, 48, 100, 100), (205, 205, 75, 75)]]
    >>> gt_classes = [[0, 1]]
    >>> map_iou_threshold = 0.5
    >>> y_true, pred_scores = evaluate_detections(boxes, classes, scores, cls_scores,
    ...                                           gt_boxes, gt_classes, map_iou_threshold, eval_type="class_scores")
    >>> print(y_true)
    [0, 1, ...]
    >>> print(pred_scores)
    [0.95, 0.88, ...]
    """

    ### Task 1: Evaluate detections by matching predicted bounding boxes 
    #           with ground truth boxes and generate corresponding true 
    #           labels and prediction scores for further evaluation (e.g., 
    #           computing mAP).
    #   Notes
    #   -----
    #   - The function uses IoU matching to determine whether a detection 
    #     sufficiently overlaps with a ground truth box.
    #   - The specific handling of scores (e.g., weighting by objectness 
    #     and/or classification) is determined by the eval_type.


    # Inner helper function to get the prediction score based on evaluation criteria
    def _get_pred_score(dt_idx, img_dt_scores, img_dt_cls_scores, eval_type):
        cls_score = np.max(img_dt_cls_scores[dt_idx]) # get class score at given detection index from one-hot encoded data
        obj_score = img_dt_scores[dt_idx]

        if eval_type == "objectness":
            return obj_score
        elif eval_type == "class_scores":
            return cls_score
        elif eval_type == "combined":
            return obj_score * cls_score
        else:
            raise ValueError("Invalid eval_type. Choose from 'objectness', 'class_scores', or 'combined'.")
        
    # Initiate pred_scores and y_true
    # n_classes = max(max(classes[i]) for i in range(len(classes))) + 1  # Determine number of classes
    # n_samples = sum(len(img_dt_boxes) for img_dt_boxes in boxes)  # Total number of detections

    n_images = len(boxes)
    n_dt_samples = [len(img_dt_boxes) for img_dt_boxes in boxes]  # number of detections per image
    n_gt_samples = [len(img_gt_boxes) for img_gt_boxes in gt_boxes] # number of ground truth per image
    n_max_samples = sum(n_dt_samples) + sum(n_gt_samples)
    # pred_scores = np.zeros((n_max_samples, num_classes + 1))  # Add 1 extra column for dummy class (-1)
    pred_scores = np.full((n_max_samples, num_classes + 1), np.nan)
    y_true = []
    
    dummy_cls_id = -1  # Dummy class ID for FP and FN
    sample_idx = 0  # Track the current sample index in pred_scores

    # Iterate each image
    for i in range(n_images):
        img_dt_boxes, img_dt_classes, img_dt_scores, img_dt_cls_scores = boxes[i], classes[i], scores[i], cls_scores[i]
        img_gt_boxes, img_gt_classes = gt_boxes[i], gt_classes[i]

        # number of detections and ground per image
        num_dts, num_gts = n_dt_samples[i], n_gt_samples[i]

        # Raise alerts for invalid inputs
        if num_dts == num_gts == 0:
            raise ValueError(f"The detection and ground truth data cannot be empty simultaneously. Check the input for image {i}")

        if num_gts == 0:
            # No ground truth, all predictions are False Positives
            for j in range(num_dts):
                pred_score = _get_pred_score(j, img_dt_scores, img_dt_cls_scores, eval_type)
                pred_class_id = img_dt_classes[j]
                pred_scores[sample_idx, pred_class_id] = pred_score # assign prediction scores
                y_true.append(dummy_cls_id) # assign true labels as -1 for FP
                sample_idx += 1
            continue

        if num_dts == 0:
            # No detections, all ground truths are False Negatives
            for j in range(num_gts):
                pred_scores[sample_idx, dummy_cls_id] = 0  # Assign prediction score as 0 for FN (last column)
                y_true.append(img_gt_classes[j])  # Assign true labels
                sample_idx += 1
            continue

        # Iterate each detection and match to ground truth
        for j in range(num_dts):
            dt_box = img_dt_boxes[j]
            ious = np.array([calculate_iou(dt_box, gt_box) for gt_box in img_gt_boxes])
            best_match_gt = np.argmax(ious)
            best_iou = ious[best_match_gt]

            # True Positive
            if best_iou >= map_iou_threshold and img_dt_classes[j] == img_gt_classes[best_match_gt]: 
                pred_score = _get_pred_score(j, img_dt_scores, img_dt_cls_scores, eval_type)
                pred_class_id = img_dt_classes[j]
                pred_scores[sample_idx, pred_class_id] = pred_score  # assign prediction scores
                y_true.append(img_gt_classes[best_match_gt]) # assign true labels
                sample_idx += 1

            # False Positive
            else: 
                pred_score = _get_pred_score(j, img_dt_scores, img_dt_cls_scores, eval_type)
                pred_class_id = img_dt_classes[j]
                pred_scores[sample_idx, pred_class_id] = pred_score # assign prediction scores
                y_true.append(dummy_cls_id) # assign true labels as -1 for FP
                sample_idx += 1

        # Iterate each ground truth and match to detection
        for j in range(num_gts):
            gt_box = img_gt_boxes[j] 
            ious = np.array([calculate_iou(dt_box, gt_box) for dt_box in img_dt_boxes])
            best_match_dt = np.argmax(ious)
            best_iou = ious[best_match_dt]

            # False Negative
            if best_iou < map_iou_threshold:
                pred_scores[sample_idx, dummy_cls_id] = 0  # Assign prediction score as 0 for FN (last column)  
                y_true.append(img_gt_classes[j]) # assign true labels
                sample_idx += 1

    
    # Remove empty rows in pred_scores
    y_true, pred_scores =  np.array(y_true), np.array(pred_scores)
    rows_with_all_nan = np.all(np.isnan(pred_scores), axis=1)
    pred_scores = pred_scores[~rows_with_all_nan] 

    # Replace existing empty values with 0
    pred_scores = np.nan_to_num(pred_scores, nan=0)

    # Double check the length matches
    assert len(y_true) == len(pred_scores)

    return y_true, pred_scores


def calculate_precision_recall_curve(y_true, pred_scores, num_classes=20):
    """
    Compute the precision-recall curve for each class in a multi-class classification task.

    This function takes the true labels and the predicted confidence scores for each class,
    then calculates the precision and recall values at various threshold levels for each class.
    The thresholds are determined by sorting the predicted scores in descending order. For every
    unique threshold (each predicted score in the sorted order), the function computes the number
    of true positives (TP), false positives (FP), and false negatives (FN) to derive the precision
    (TP / (TP + FP)) and recall (TP / (TP + FN)). 

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels for each sample. Each element should be an integer representing the correct class.
    pred_scores : array-like of shape (n_samples, n_classes)
        Predicted scores or confidence values for each class, for every sample. Each row should correspond
        to a sample, and each column corresponds to one of the classes. Higher scores indicate a higher
        confidence in the prediction for that class.
    num_classes : int, optional
        The total number of classes. This parameter is used to binarize the true labels and to iterate
        over each class when computing the precision and recall curves. Default is 20.

    Returns
    -------
    precision : dict
        A dictionary where each key is a class index (from 0 to num_classes-1) and the corresponding value
        is a list of precision values computed at various score thresholds. The precision is calculated as
        TP / (TP + FP) at each threshold.
    recall : dict
        A dictionary where each key is a class index (from 0 to num_classes-1) and the corresponding value
        is a list of recall values computed at various score thresholds. The recall is calculated as
        TP / (TP + FN) at each threshold.
    thresholds : dict
        A dictionary where each key is a class index (from 0 to num_classes-1) and the corresponding value
        is a numpy array of threshold values (sorted in descending order, with an extra 0 appended) used to
        compute the precision and recall for that class.

    Notes
    -----
    - The true labels are first binarized using `label_binarize` from scikit-learn to facilitate
      per-class evaluation.
    - For each class, predicted scores are sorted in descending order, and the true binary labels are
      rearranged accordingly.
    - The precision and recall are computed iteratively: for each threshold, the counts of true positives,
      false positives, and false negatives are updated, and the corresponding precision and recall are computed.
    - This function assumes that higher predicted scores correspond to a higher likelihood that the sample
      belongs to the class.

    Examples
    --------
    >>> y_true = [0, 1, 2, 1, 0]
    >>> pred_scores = [[0.9, 0.05, 0.05],
    ...                [0.1, 0.8, 0.1],
    ...                [0.05, 0.1, 0.85],
    ...                [0.2, 0.7, 0.1],
    ...                [0.8, 0.1, 0.1]]
    >>> precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes=3)
    >>> print(precision[0])
    [ ... ]
    >>> print(recall[0])
    [ ... ]
    
    Returns the precision, recall values, and thresholds for each class based on the provided predictions.
    """
    ### Task 2: Compute the precision-recall curve for each class 
    #           in a multi-class classification task. 
    #           Notes
    #           -----
    #           - The true labels are first binarized using `label_binarize` from scikit-learn to facilitate
    #             per-class evaluation.
    #           - The precision and recall are computed iteratively: for each threshold, the counts of true positives,
    #             false positives, and false negatives are updated, and the corresponding precision and recall are computed.
    #           - This function assumes that higher predicted scores correspond to a higher likelihood that the sample
    #             belongs to the class.

    # Internal Helper function to get the the precision and recall from given class
    def _get_precision_recall(y_true_binary, y_pred):
        # Sort scores and corresponding truth values in descending order
        desc_score_indices = np.argsort(y_pred)[::-1]
        y_pred = y_pred[desc_score_indices]
        y_true_binary = y_true_binary[desc_score_indices]

        # Initialize variables
        precision, recall = [], []
        thresholds = np.unique(y_pred)

        # Iterate each threshold to get precision and recall
        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

            # Compute precision and recall
            
            if tp == 0 and fp == 0 and fn == 0: # corner case: https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
                precision.append(1.0)
                recall.append(1.0)

            elif tp + fn == 0: # no true positives
                precision.append(tp / (tp + fp))
                recall.append(1.0) 

            elif tp + fp == 0: # no predicted positives
                precision.append(1.0)
                recall.append(tp / (tp + fn))

            else:
                precision.append(tp / (tp + fp))
                recall.append(tp / (tp + fn))
        
        # The last point (1.0, 0.0) is appended to maintain a well-formed precision-recall curve
        precision.append(1.0)
        recall.append(0.0)
        return precision, recall, thresholds


    # Initialize variables
    precision = {}
    recall = {}
    thresholds = {}

    # Iterate over each valid class (excluding the dummy class)
    for class_id in range(num_classes):
        # Get the ground truth binary labels for the current class
        y_true_binary = (y_true == class_id).astype(int) 

        # Get the predicted scores for the current class
        y_pred = pred_scores[:, class_id]

        # Get the precision and recall for each class
        precision[class_id], recall[class_id], thresholds[class_id] = _get_precision_recall(y_true_binary, y_pred)

    return precision, recall, thresholds


def calculate_map_x_point_interpolated(precision_recall_points, num_classes, num_interpolated_points=11):
    """
    Calculate the Mean Average Precision (mAP) using x-point interpolation for multi-class object detection tasks.

    This function computes the average precision for each class by interpolating the precision values at a fixed
    number of recall thresholds (default is 11, corresponding to recall levels from 0.0 to 1.0 in increments of 0.1).
    For each class, the precision-recall curve is first sorted in descending order by precision. Then, for each
    recall threshold, the maximum precision for all recall values greater than or equal to the threshold is selected.
    The average precision for a class is the mean of these interpolated precision values, and the mAP is the average
    of the average precisions across all classes.

    Parameters
    ----------
    precision_recall_points : dict
        A dictionary where:
          - Keys are class indices (e.g., 0, 1, 2, ...).
          - Values are lists of tuples (recall, precision) that represent points on the precision-recall curve for the class.
            It is assumed that these points are generated from detection evaluations and that the list is not necessarily sorted.
    num_classes : int
        The total number of classes for which the mAP should be computed.
    num_interpolated_points : int, optional
        The number of equally spaced recall thresholds at which to interpolate the precision values.
        Default is 11, which corresponds to thresholds [0.0, 0.1, 0.2, ..., 1.0].

    Returns
    -------
    float
        The overall mean average precision (mAP) value averaged over all classes.

    Process
    -------
    For each class:
      1. Retrieve the list of (recall, precision) points and sort them in descending order of precision.
      2. For each of the specified recall thresholds (e.g., 0.0, 0.1, ..., 1.0):
         - Find all precision values corresponding to recall values that are greater than or equal to the threshold.
         - If any such precision values exist, take the maximum as the interpolated precision for that threshold.
         - If no points exist for a given threshold, assign a precision of 0 for that threshold.
      3. Compute the average precision for the class as the mean of these interpolated precision values.
    Finally, compute the overall mAP as the mean of the average precisions across all classes.

    Example
    -------
    >>> # Assume precision_recall_points for 3 classes are given as follows:
    >>> precision_recall_points = {
    ...     0: [(0.0, 1.0), (0.5, 0.8), (1.0, 0.6)],
    ...     1: [(0.0, 0.9), (0.3, 0.7), (0.7, 0.5), (1.0, 0.4)],
    ...     2: [(0.0, 0.95), (0.6, 0.85), (1.0, 0.75)]
    ... }
    >>> num_classes = 3
    >>> map_value = calculate_map_x_point_interpolated(precision_recall_points, num_classes)
    >>> print(map_value)
    0.7  # (for example, actual value depends on interpolation)

    Returns
    -------
    float
        The computed mean average precision (mAP) value.
    """
    mean_average_precisions = []

    for i in range(num_classes):
        # Retrieve the precision-recall points for the current class.
        points = precision_recall_points[i]
        # Sort the points in descending order based on the precision value.
        points = sorted(points, key=lambda x: x[1], reverse=True)
        
        interpolated_precisions = []
        # Generate the list of recall thresholds at which to interpolate.
        for recall_threshold in [j * 0.1 for j in range(num_interpolated_points)]:
            # For the current recall threshold, gather all precision values
            # for which the recall is greater than or equal to the threshold.
            possible_precisions = [p for r, p in points if r >= recall_threshold]
            
            # Interpolate the precision: if any precision values are found,
            # select the maximum one; otherwise, assign 0.
            if possible_precisions:
                interpolated_precisions.append(max(possible_precisions))
            else:
                interpolated_precisions.append(0)
        
        # Calculate the average precision for the current class as the mean of the interpolated precisions.
        mean_average_precision = sum(interpolated_precisions) / len(interpolated_precisions)
        mean_average_precisions.append(mean_average_precision)
    
    # Calculate the overall mAP as the mean of the average precisions across all classes.
    overall_map = sum(mean_average_precisions) / num_classes
    
    return mean_average_precisions, overall_map




if __name__ == "__main__":
    # -------------------------
    # Configuration Parameters
    # -------------------------
    
    # Number of classes in the dataset (e.g., classes: 0, 1, 2)
    num_classes = 3

    # IoU threshold for considering a detection as a valid match with a ground truth box.
    map_iou_threshold = 0.5

    # ---------------------------
    # Ground Truth Initialization
    # ---------------------------
    
    # Define ground truth bounding boxes for each image.
    # Each inner list corresponds to one image, with each bounding box defined as [x, y, width, height].
    gt_boxes = [
        [[33, 117, 259, 396], [362, 161, 259, 362]],  # Ground truth boxes for image 1
        [[163, 29, 301, 553]]                          # Ground truth boxes for image 2
    ]
    
    # Define ground truth class labels for each image.
    # For image 1, both boxes are labeled as class 0; for image 2, the box is labeled as class 2.
    gt_classes = [
        [0, 0],  # Classes for image 1
        [2]      # Class for image 2
    ]
    
    # -------------------------------
    # Detection (Prediction) Setup
    # -------------------------------
    
    # Define detection bounding boxes for each image.
    # These boxes are designed to approximately match the ground truth boxes.
    # Note: The third detection in image 1 is extra and may be considered a false positive.
    boxes = [
        [[30, 187, 253, 276], [363, 194, 266, 291], [460, 371, 52, 23]],  # Detections for image 1
        [[147, 26, 322, 578]]                                               # Detections for image 2
    ]
    
    # Define the predicted class labels for each detection.
    # These are dummy values indicating which class is predicted for each detection.
    classes = [
        [0, 0, 1],  # Detected classes for image 1 (note: the third detection is labeled as class 1)
        [2]         # Detected class for image 2
    ]
    
    # Define detection confidence scores for each detection.
    # These scores indicate the confidence level for each detection.
    scores = [
        [0.95, 0.92, 0.30],  # Confidence scores for detections in image 1
        [0.91]               # Confidence score for the detection in image 2
    ]
    
    # -------------------------------
    # Classification Scores Generation
    # -------------------------------
    
    # Generate dummy classification scores for each detection.
    # Instead of using the same numbers as detection confidence scores,
    # we now use different numbers for demonstration.
    # For image 1, we use [0.85, 0.75, 0.65] and for image 2, we use [0.80].
    dummy_max_cls_scores = [
        [0.85, 0.75, 0.65],  # Dummy classification scores for image 1
        [0.80]               # Dummy classification score for image 2
    ]
    # For each image, create a one-hot encoded matrix for the detected classes and multiply
    # element-wise by the corresponding dummy classification scores to generate a score vector.
    cls_scores = [
        np.eye(num_classes)[np.array(class_list)] * np.array(score_list)
        for class_list, score_list in zip(classes, dummy_max_cls_scores)
    ]

    # ---------------------------
    # Evaluation of Detections
    # ---------------------------
    
    # Evaluate detections by matching them with ground truth boxes.
    # This function compares predicted boxes with ground truth boxes using IoU,
    # assigns true labels and prediction scores based on the matches,
    # and handles false positives and false negatives.
    y_true, pred_scores = evaluate_detections(
        boxes,         # Detected bounding boxes per image
        classes,       # Detected class labels per image
        scores,        # Detection confidence scores per image
        cls_scores,    # Classification score vectors per image
        gt_boxes,      # Ground truth bounding boxes per image
        gt_classes,    # Ground truth class labels per image
        map_iou_threshold,  # IoU threshold for a valid match
        num_classes # Number of classes
    )
    
    # Print the evaluation results: true labels and corresponding prediction scores.
    print("True labels:", y_true) # -1 for false positive
    print("Prediction scores:", pred_scores)
    
   
    # ---------------------------
    # Precision-Recall Curve Calculation
    # ---------------------------
    
    # Calculate the precision-recall curve based on the true labels and prediction scores.
    # This function returns dictionaries containing precision, recall, and threshold values for each class.
    precision, recall, thresholds = calculate_precision_recall_curve( 
        y_true,         # True labels from the evaluation
        pred_scores,    # Predicted scores from the evaluation
        num_classes=num_classes  # Number of classes
    )

    # For each class, print the precision, recall, and threshold values.
    for cls in range(num_classes):
        print(f"\nClass {cls}:")
        print("Precision:", precision[cls])
        print("Recall:", recall[cls])
        print("Thresholds:", thresholds[cls])

    # ---------------------------
    # Creating Precision-Recall Pairs
    # ---------------------------
    
    # Combine the precision and recall values into (recall, precision) pairs for each class.
    precision_recall_points = {
        class_index: list(zip(recall[class_index], precision[class_index]))
        for class_index in range(num_classes)
    }

    # ---------------------------
    # Compute Mean Average Precision (mAP)
    # ---------------------------
    
    # Compute the Mean Average Precision (mAP) using 11-point interpolation.
    # The function calculates the average precision for each class by interpolating precision
    # at 11 recall levels, and then averages these values to obtain the mAP.
    map_value = calculate_map_x_point_interpolated(precision_recall_points, num_classes)

    # Output the calculated mAP value formatted to four decimal places.
    print(f"Mean Average Precision (mAP): {map_value:.4f}")
