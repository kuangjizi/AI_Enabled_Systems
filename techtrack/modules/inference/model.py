import cv2
import numpy as np
from typing import List, Tuple


class Detector:
    """
    A class that represents an object detection model using OpenCV's DNN module
    with a YOLO-based architecture.
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str, score_threshold: float=.5) -> None:
        """
        Initializes the YOLO model by loading the pre-trained network and class labels.

        :param weights_path: Path to the pre-trained YOLO weights file.
        :param config_path: Path to the YOLO configuration file.
        :param class_path: Path to the file containing class labels.

        :ivar self.net: The neural network model loaded from weights and config files.
        :ivar self.classes: A list of class labels loaded from the class_path file.
        :ivar self.img_height: Height of the input image/frame.
        :ivar self.img_width: Width of the input image/frame.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load class labels
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.img_height: int = 0
        self.img_width: int = 0

        self.score_threshold = score_threshold

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """
        Runs the YOLO model on a single input frame and returns raw predictions.

        :param preprocessed_frame: A single image frame that has been preprocessed 
                                   for YOLO model inference (e.g., resized and normalized).

        :return: A list of NumPy arrays containing the raw output from the YOLO model.
                 Each output consists of multiple detections with bounding boxes, 
                 confidence scores, and class probabilities.

        :ivar self.img_height: The height of the input image/frame.
        :ivar self.img_width: The width of the input image/frame.


        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        self.img_height, self.img_width = preprocessed_frame.shape[:2]

        # TASK 2: Use the YOLO model to return all raw outputs
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(preprocessed_frame, 1/255.0, (self.img_width, self.img_height), swapRB=True, crop=False)

        # Set the input to the network and perform a forward pass
        self.net.setInput(blob)
        
        # Get the output layer names (filtering out the connected output layers)
        output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Get the raw outputs from the model
        outputs = self.net.forward(output_layers)

        # Return model outputs:
        return outputs

    def post_process(
        self, predict_output: List[np.ndarray]
    ) -> Tuple[List[List[int]], List[int], List[float], List[np.ndarray]]:
        """
        Processes the raw YOLO model predictions and filters out low-confidence detections.

        :param predict_output: A list of NumPy arrays containing raw predictions 
                               from the YOLO model.
        :param score_threshold: Minimum confidence score required for a detection 
                                to be considered valid.

        :return: A tuple containing:
            - **bboxes (List[List[int]])**: List of bounding boxes as `[x, y, width, height]`, 
              where (x, y) represents the top-left corner.
            - **class_ids (List[int])**: List of detected object class indices.
            - **confidence_scores (List[float])**: List of confidence scores for each detection.
            - **class_scores (List[np.ndarray])**: List of all class-specific confidence scores.

        **YOLO Output Format:**
        Each detection in the output contains:
        - First 4 values: Bounding box center x, center y, width, height.
        - 5th value: Confidence score.
        - Remaining values: Class probabilities for each detected object.

        **Post-processing steps:**
        1. Extract bounding box coordinates from YOLO output.
        2. Compute class probabilities and determine the most likely class.
        3. Filter out detections below the confidence threshold.
        4. Convert bounding box coordinates from center-based format to 
           top-left corner format.

        **Bounding Box Conversion:**
        YOLO outputs bounding box coordinates in the format:
        ```
        center_x, center_y, width, height
        ```
        This function converts them to:
        ```
        x, y, width, height
        ```
        where (x, y) is the top-left corner.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        
        # TASK 3: Use the YOLO model to return list of NumPy arrays filtered
        #         by processing the raw YOLO model predictions and filters out 
        #         low-confidence detections (i.e., < score_threshold). Use the logic
        #         in Line 83-88.

        bboxes, class_ids, confidence_scores, class_scores = [], [], [], []
        for output in predict_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores) # Get most likely class
                confidence = scores[class_id]  # Confidence of class
                
                if confidence > self.score_threshold:
                    box = detection[:4] * np.array([self.img_width, self.img_height, self.img_width, self.img_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    bboxes.append(box)
                    confidence_scores.append(float(confidence))
                    class_ids.append(class_id)
                    class_scores.append(scores)

        # Return these variables in order:
        return bboxes, class_ids, confidence_scores, class_scores



# Test the Model module
if __name__ == "__main__":
    from preprocessing import Preprocessing

    # Initialize the Preprocessing and Detector classes with test_video and yolo_model_1
    test_video_path = "techtrack/storage/test_videos/worker-zone-detection.mp4"
    weights_path="techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights"
    config_path="techtrack/storage/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg"
    class_path="techtrack/storage/yolo_model_1/logistics.names"

    video_processor = Preprocessing(test_video_path, drop_rate=100)
    detector_model = Detector(weights_path, config_path, class_path)

    # Perform object detection on the first 5 yielded frames
    frame_count = 0
    for frame in video_processor.capture_video():
        if frame_count < 5:
            predictions = detector_model.predict(frame)
            bboxes, class_ids, confidence_scores, class_scores = detector_model.post_process(predictions)
            print(f"Frame {frame_count+1}: {len(predictions)} raw outputs; {len(bboxes)} detections.")
            if len(bboxes) > 0:
                print(f"\tBounding Boxes: {bboxes}, class IDs: {class_ids}, confidence scores: {confidence_scores}, class_scores: {class_scores}")
        else:
            break
        frame_count += 1
    
