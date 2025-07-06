import numpy as np
from typing import List, Tuple
from ..utils.metrics import calculate_iou


class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.

        :ivar self.score_threshold: The threshold below which detections are discarded.
        :ivar self.nms_iou_threshold: The IoU threshold that determines whether two boxes 
                                      are considered redundant.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
    
    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[float],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.

        :return: A tuple containing:
            - **filtered_bboxes (List[List[int]])**: The final bounding boxes after NMS.
            - **filtered_class_ids (List[int])**: The class IDs of retained bounding boxes.
            - **filtered_scores (List[float])**: The confidence scores of retained bounding boxes.
            - **filtered_class_scores (List[float])**: The class-specific scores of retained boxes.

        **How NMS Works:**
        - The function selects the bounding box with the highest confidence.
        - It suppresses any boxes that have a high IoU (overlapping area) with this selected box.
        - This process is repeated until all valid boxes are retained.

        **Example Usage:**
        ```python
        nms_processor = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        final_bboxes, final_class_ids, final_scores, final_class_scores = nms_processor.filter(
            bboxes, class_ids, scores, class_scores
        )
        ```
        """

        # TASK 4: Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        #         DO NOT USE **cv2.dnn.NMSBoxes()** for this Assignment. For Assignment 5, you will be
        #         permitted to use this function.
        
        # Convert inputs to numpy arrays for easier manipulation
        bboxes = np.array(bboxes)
        class_ids = np.array(class_ids)
        scores = np.array(scores)
        class_scores = np.array(class_scores)

        # Filter out boxes with scores below the threshold
        keep_indices = [i for i, score in enumerate(scores) if score >= self.score_threshold]
        filtered_bboxes = np.copy(bboxes[keep_indices])
        filtered_class_ids = np.copy(class_ids[keep_indices])
        filtered_scores = np.copy(scores[keep_indices])
        filtered_class_scores = np.copy(class_scores[keep_indices])


        # Sort indices by scores in descending order
        sorted_indices = np.argsort(filtered_scores)[::-1]
        filtered_bboxes = filtered_bboxes[sorted_indices]
        filtered_class_ids = filtered_class_ids[sorted_indices]
        filtered_scores = filtered_scores[sorted_indices]
        filtered_class_scores = filtered_class_scores[sorted_indices]
        sorted_indices = np.array([i for i in range(len(filtered_bboxes))]) # reset sorted_indices

        # Initialize a list to store the indices of boxes to keep
        keep = []
        while len(sorted_indices) > 0:
            # Select the box with the highest score
            i = sorted_indices[0]
            sorted_indices = sorted_indices[1:] # pop the first element
            keep.append(i)

            # Compute IoU of this box with all remaining boxes
            iou = np.array([calculate_iou(filtered_bboxes[i], bbox) for bbox in filtered_bboxes[sorted_indices[1:]]])

            # Find indices of boxes with IoU less than the threshold
            non_overlapping_indices = np.where(iou <= self.nms_iou_threshold)[0]

            # Keep only non-overlapping boxes
            sorted_indices = sorted_indices[non_overlapping_indices]

        # Get the finalized results
        filtered_bboxes = filtered_bboxes[keep].tolist()
        filtered_class_ids = filtered_class_ids[keep].tolist()
        filtered_scores = filtered_scores[keep].tolist()
        filtered_class_scores = filtered_class_scores[keep].tolist()

        # Return these variables in order as described in Line 46-50:
        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores
   