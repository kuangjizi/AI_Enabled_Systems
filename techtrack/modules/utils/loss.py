import itertools
import numpy as np
from .metrics import calculate_iou

class Loss:
    """
    *Modified* YOLO Loss for Hard Negative Mining.

    Attributes:
        num_classes (int): Number of classes.
        iou_threshold (float): Intersection over Union (IoU) threshold.
        lambda_coord (float): Weighting factor for localization loss.
        lambda_noobj (float): Weighting factor for no object confidence loss.
    """

    def __init__(self, iou_threshold=0.5, lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.5, lambda_cls=0.5, num_classes=20):
        """
        Initialize the Loss object with the given parameters.

        Internal Process:
        1. Stores the provided hyperparameters as instance attributes.
        2. Defines the column names for loss components to track them in results.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weighting factor for localization loss.
            lambda_obj (float): Weighting factor for objectness loss.
            lambda_noobj (float): Weighting factor for no object confidence loss.
            lambda_cls (float): Weighting factor for classification loss.
        """
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.columns = [
            'total_loss', 
            f'loc_loss (lambda={self.lambda_coord})', 
            'conf_loss_obj', 
            f'conf_loss_noobj (lambda={self.lambda_noobj})', 
            'class_loss'
        ]
        self.iou_threshold = iou_threshold

    def cross_entropy_loss(self, y_true, y_pred, epsilon=1e-12):
        """
        Compute the cross entropy loss between true labels and predicted probabilities.

        Args:
            y_true (numpy array): True labels, one-hot encoded or binary labels.
            y_pred (numpy array): Predicted probabilities, same shape as y_true.
            epsilon (float): Small value to avoid log(0). Default is 1e-12.

        Returns:
            float: Cross entropy loss.
        """
        # Clip y_pred to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

        # Binary classification case
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Multi-class classification case
        else:
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def bbox_loss(self, true_bbox, pred_bbox):
        """
        Compute the bbox_loss with mean squared error between true and predicted bounding boxes.

        Args:
            true_bbox (list): True bounding box coordinates as [x_center, y_center, width, height].
            pred_bbox (list): Predicted bounding box coordinates as [x_center, y_center, width, height].

        Returns:
            float: Bounding box loss.
        """
        return np.sum((true_bbox - pred_bbox) ** 2)

    def confidence_loss(self, true_confidence_score, pred_confidence_score):
        """
        Compute the objectness_loss

        Args:
            true_confidence_score (float): True objectness score.
            pred_confidence_score (float): Predicted objectness score.

        Returns:
            float: Objectness loss.
        """
        return (true_confidence_score - pred_confidence_score) ** 2
    
    def get_predictions(self, predictions):
        """
        Extracts bounding box coordinates, objectness scores, and class scores from predictions.

        Internal Process:
        1. Iterates over predictions to extract bounding box coordinates.
        2. Extracts objectness scores.
        3. Extracts class scores.

        Args:
            predictions (list): List of predicted bounding boxes and associated scores.
        
        Returns:
            tuple: (bounding boxes, class_ids, objectness scores, class scores)
        """
        bboxes, class_ids, objectness_scores, class_scores = [], [], [], []
        for output in predictions:  
            for detection in output:
                box = detection[:4]  # (cx, cy, w, h)
                objectness = detection[4]  # Objectness score
                
                scores = detection[5:]  # Class scores
                class_id = np.argmax(scores)  # Most likely class
                
                # Store values
                bboxes.append(box)
                objectness_scores.append(float(objectness))
                class_ids.append(class_id)
                class_scores.append(scores)

        return bboxes, class_ids, objectness_scores, class_scores
        
    def get_annotations(self, annotations):
        """
        Extract ground truth bounding boxes and class IDs from annotations.
        
        Internal Process:
        1. Iterates over annotations to extract bounding box coordinates.
        2. Extracts the corresponding class labels.
        
        Args:
            annotations (list): List of ground truth annotations.
        
        Returns:
            tuple: (ground truth bounding boxes, class labels)
        """
        
        bboxes = []
        class_ids = []
        for annotation in annotations:
            class_label, x_center, y_center, width, height = annotation
            class_ids.append(class_label)
            box = [x_center, y_center, width, height]
            bboxes.append(box)

        return bboxes, class_ids

    def compute(self, predictions, annotations):
        """
        Compute the YOLO loss components.

        Internal Process:
        1. Extracts predictions and annotations of a single image/frame.
        2. Iterates through annotations to compute localization, confidence, and class loss.
        3. Computes total loss using predefined weighting factors.

        Args:
            predictions (list): List of predictions of a single image.
            annotations (list): List of ground truth annotations of a single image.

        Returns:
            dict: Dictionary containing the computed loss components.
        """
        loc_loss = 0 # localization loss
        class_loss = 0 # classification loss
        conf_loss_obj = 0 # objectness (or confidence) loss
        total_loss = 0 # aggregate loss including loc_loss, class_loss, conf_loss_obj

        # TASK 2: Complete this method to compute the Loss function.
        #         This method calculates the localization, objectness 
        #         (or confidence) and classification loss.
        #         This method will be called in the HardNegativeMiner class.
        #         ----------------------------------------------------------
        #         HINT: For simplicity complete use get_predictions(), get_annotations().
        #         You may add class methods to improve the readability of this code. 
        #         For your convenience, cross_entropy_loss() is already implemented for you.

        # Extract predictions and annotations
        pred_bboxes, pred_class_ids, pred_confidence_scores, pred_class_scores = self.get_predictions(predictions)
        true_bboxes, true_class_ids = self.get_annotations(annotations)
        
        # Convert to numpy arrays for easier computation
        pred_bboxes, pred_class_ids, pred_confidence_scores, pred_class_scores = np.array(pred_bboxes, dtype=np.float32), np.array(pred_class_ids, dtype=np.int32), np.array(pred_confidence_scores, dtype=np.float32), np.array(pred_class_scores, dtype=np.float64)
        true_bboxes, true_class_ids = np.array(true_bboxes, dtype=np.float32), np.array(true_class_ids, dtype=np.int32)

        # Iterate over ground truth 
        for i in range(len(true_bboxes)):
            true_bbox = true_bboxes[i]
            true_class_id = true_class_ids[i]

            # Compute IoU between ground truth and all predictions and get the best match
            for j, pred_bbox in enumerate(pred_bboxes):
                if calculate_iou(true_bbox, pred_bbox) > self.iou_threshold:
                    # Localization loss
                    loc_loss += self.lambda_coord * self.bbox_loss(true_bbox, pred_bboxes[j])

                    # Confidence loss (object)
                    conf_loss_obj += self.lambda_obj * self.confidence_loss(1, pred_confidence_scores[j])

                    # Classification loss
                    y_true = np.zeros(self.num_classes, dtype=np.float64)
                    y_true[true_class_id] = 1.0  # One-hot encode true class
                    y_pred = pred_class_scores[j]  # Predicted class probabilities
                    class_loss += self.lambda_cls * self.cross_entropy_loss(y_true, y_pred)
                
                # Confidence loss (no object) for non-matching predictions
                else:
                     conf_loss_obj += self.lambda_noobj * self.confidence_loss(0, pred_confidence_scores[j])

        # Compute total loss
        total_loss = loc_loss + conf_loss_obj + class_loss

        return {
            "loc_loss": loc_loss, 
            "conf_loss_obj": conf_loss_obj, 
            "class_loss": class_loss,
            "total_loss": total_loss, 
        }

if __name__ == "__main__":
    test_cases = {
        "Perfect Prediction": ([[[0.5, 0.5, 1.0, 1.0, 1.0, 1, 0]]],  # (cx, cy, w, h, conf, class_scores)
                               [[0, 0.5, 0.5, 1.0, 1.0]],            # (class_label, cx, cy, w, h)
                               0.0),
        
        "Localization Error": ([[[0.5, 0.5, 1.0, 1.0, 1.0, 1, 0]]], 
                               [[0, 0.6, 0.5, 1.0, 1.0]], 
                               0.005),
        
        "Confidence Mismatch": ([[[0.5, 0.5, 1.0, 1.0, 0.5, 1, 0]]], 
                                [[0, 0.5, 0.5, 1.0, 1.0]], 
                                0.125),

        "Classification Error": ([[[0.5, 0.5, 1.0, 1.0, 0.5, 1, 0]]], 
                                [[1, 0.5, 0.5, 1.0, 1.0]], 
                                "greater than 0"),

        "False Positive Error": ([[[5, 5, 5, 5, 0.5, 1, 0]]], 
                                [[0, 0, 0, 0, 0]], 
                                0.1),
    }

    for test_name, (pred, target, expected) in test_cases.items():
        loss = Loss(iou_threshold=0.5, lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.4, lambda_cls=0.5, num_classes=2)
        total_loss = float(loss.compute(pred, target)["total_loss"])
        print(f"{test_name} Loss: {round(total_loss, 4)}")

        if expected == 'greater than 0':
            assert total_loss > 0, f"{test_name} failed! Expected loss > 0, got {total_loss}"
        else:
            expected = float(expected)
            assert np.isclose(total_loss, expected, atol=1e-12), f"{test_name} failed! Expected {expected}, got {total_loss}"
            
        print(f"{test_name} passed!")
        print("\n")