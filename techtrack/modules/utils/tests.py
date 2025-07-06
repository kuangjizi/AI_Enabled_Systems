# Test cases and functions for validation

import numpy as np
from .metrics import calculate_precision_recall_curve
from ..inference.nms import NMS

# Helper function to test out metrics.calculate_precision_recall_curve()
def test_metrics_calculate_precision_recall_curve():
    from sklearn.metrics import precision_recall_curve
    
    print("========= Test starts for utils.metrics.calculate_precision_recall_curve() =========")
    # Test Case 1: Basic multi-class classification with perfect predictions
    y_true = np.array([0, 1, 2, 1, 0, 2])
    pred_scores = np.array([
        [1.0, 0.0, 0.0],  # Perfect confidence in class 0
        [0.0, 1.0, 0.0],  # Perfect confidence in class 1
        [0.0, 0.0, 1.0],  # Perfect confidence in class 2
        [0.0, 1.0, 0.0],  # Perfect confidence in class 1
        [1.0, 0.0, 0.0],  # Perfect confidence in class 0
        [0.0, 0.0, 1.0]   # Perfect confidence in class 2
    ])
    num_classes = 3

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)
        assert np.allclose(recall[class_id], sk_recall, atol=1e-6)
        assert np.allclose(thresholds[class_id], sk_thresholds, atol=1e-6)

    print("Test Case 1 Passed: Perfect Predictions.")

    # Test Case 2: Single class present
    y_true = np.array([0, 0, 0, 0, 0])
    pred_scores = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.6, 0.4, 0.0],
        [0.5, 0.5, 0.0]
    ])
    num_classes = 3

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        # print(f"class_id: {class_id}, {sk_precision}, {sk_recall}, {sk_thresholds}")
        # print(f"class_id: {class_id}, {precision[class_id]}, {recall[class_id]}, {thresholds[class_id]}")
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)
        assert np.allclose(recall[class_id], sk_recall, atol=1e-6)
        assert np.allclose(thresholds[class_id], sk_thresholds, atol=1e-6)
    print("Test Case 2 Passed: Single Class Present.")

    # Test Case 3: All classes missing from predictions
    y_true = np.array([0, 1, 2])
    pred_scores = np.zeros((3, 3))  # All scores are zero
    num_classes = 3

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)

    print("Test Case 3 Passed: All Zero Predictions.")

     # Test Case 4: Imbalanced class distribution
    y_true = np.array([0] * 95 + [1] * 5)  # 95% class 0, 5% class 1
    pred_scores = np.random.rand(100, 2)  # Random scores
    num_classes = 2

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)
        assert np.allclose(recall[class_id], sk_recall, atol=1e-6)
        assert np.allclose(thresholds[class_id], sk_thresholds, atol=1e-6)

    print("Test Case 4 Passed: Imbalanced Classes.")

    # Test Case 5: Edge Case with Single Sample
    y_true = np.array([0])
    pred_scores = np.array([[0.9, 0.1, 0.0]])
    num_classes = 3

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)
        assert np.allclose(recall[class_id], sk_recall, atol=1e-6)
        assert np.allclose(thresholds[class_id], sk_thresholds, atol=1e-6)
    print("Test Case 5 Passed: Single Sample.")

    # Test Case 6: Given test case from assignment
    y_true = np.array([0, 0, -1, 2])
    pred_scores = np.array([
        [0.85, 0.0, 0.0, 0.0],
        [0.85, 0.0, 0.0, 0.0],
        [0.0, 0.75, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0],
    ])
    num_classes = 3

    precision, recall, thresholds = calculate_precision_recall_curve(y_true, pred_scores, num_classes)

    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true_binary, pred_scores[:, class_id])
        assert np.allclose(precision[class_id], sk_precision, atol=1e-6)
        assert np.allclose(recall[class_id], sk_recall, atol=1e-6)
        assert np.allclose(thresholds[class_id], sk_thresholds, atol=1e-6)
    print("Test Case 6 Passed: Given test case from assignment.")


    print("========= All test cases passed for utils.metrics.calculate_precision_recall_curve() =========")
    print("\n")



def test_nms_class():
    import cv2  

    print("========= Test starts for inference.nms.NMS =========")

    # Test case 1
    bboxes_1 = [
        [10, 10, 100, 100],  # Large box
        [20, 20, 80, 80],    # Smaller box inside the large box
        [15, 15, 90, 90],    # Overlapping box with the large box
        [200, 200, 50, 50],  # Separate box (should not be suppressed)
    ]
    class_ids_1 = [0, 1, 0, 2] #[0, 0, 0, 2]
    scores_1 = [0.9, 0.75, 0.85, 0.95] #[0.9, 0.3, 0.85, 0.95]
    class_scores_1 = [0.8, 0.6, 0.7, 0.9] # [0.8, 0.6, 0.7, 0.9]

    nms = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
    filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores = nms.filter(bboxes_1, class_ids_1, scores_1, class_scores_1)
    # print("Filtered bboxes:", filtered_bboxes)

    nms_expected_indices = cv2.dnn.NMSBoxes(bboxes_1, scores_1, 0.5, 0.4)
    # print("Expected bboxes:", [bboxes_1[i] for i in nms_expected_indices])

    assert filtered_bboxes == [bboxes_1[i] for i in nms_expected_indices]
    print("Test case 1 passed.")

    # Test case 2
    bboxes_2 = [
        [100, 100, 50, 50],   # Box A
        [102, 102, 50, 50]    # Box B, nearly identical to Box A
    ]
    class_ids_2 = [1, 1] # [0, 1]
    scores_2 = [0.95, 0.90] # Box A has a higher confidence score than Box B
    class_scores_2 = [0.92, 0.88] # [0.8, 0.6]

    nms = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
    filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores = nms.filter(bboxes_2, class_ids_2, scores_2, class_scores_2)
    # print("Filtered bboxes:", filtered_bboxes)

    nms_expected_indices = cv2.dnn.NMSBoxes(bboxes_2, scores_2, 0.5, 0.4)
    # print("Expected bboxes:", [bboxes_2[i] for i in nms_expected_indices])

    assert filtered_bboxes == [bboxes_2[i] for i in nms_expected_indices]
    print("Test case 2 passed.")
    # print("\n")

    # Test case 3
    bboxes_3 = [[100,10,20,20], [12,20,50,80], [30,23,45,45]]
    class_ids_3 = [0, 1, 2]
    scores_3 = [0.9, 0.7, 0.5]
    class_scores_3 = [0.8, 0.6, 0.4]

    nms = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
    filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores = nms.filter(bboxes_3, class_ids_3, scores_3, class_scores_3)
    # print("Filtered bboxes:", filtered_bboxes)

    nms_expected_indices = cv2.dnn.NMSBoxes(bboxes_3, scores_3, 0.5, 0.4)
    # print("Expected bboxes:", [bboxes_3[i] for i in nms_expected_indices])
    
    assert filtered_bboxes == [bboxes_3[i] for i in nms_expected_indices]
    print("Test case 3 passed.")
    
    print("========= All test cases passed for inference.nms.NMS =========")
    print("\n")


if __name__=="__main__":
    test_metrics_calculate_precision_recall_curve()
    test_nms_class()