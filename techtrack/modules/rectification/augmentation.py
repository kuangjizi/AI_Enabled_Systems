import cv2
import numpy as np
import random


class Augmenter:
    """
    A collection of dataset augmentation methods including transformations, 
    blurring, resizing, and brightness adjustments. 

    NOTE: This class is used to transform data necessary for training TechTrack's models.
          Imagine that the output of `self.transform()` is fed directly to train the model.
    
    The following transformations are included:
    - Horizontal flipping: i.e., def horizontal_flip(**kwargs)
    - Gaussian blurring: i.e., def gaussian_blur_image(**kwargs)
    - Resizing: i.e., def resize(**kwargs)
    - Brightness and contrast adjustments: i.e., def change_brightness(**kwargs)
        - HINT: you may use cv2.addWeighted()

    NOTE: These methods uses **kwargs to accept arbitrary keyword arguments,
    but explicit parameter definitions improve clarity and usability.
    - "**kwargs" reference: https://www.geeksforgeeks.org/args-kwargs-python/

    Finally, Provide a demonstration and visualizations of these methods in `notebooks/augmentation.ipynb`.
    You will define your own keywords for "**kwargs".
    """

    ## TASK 1: Complete the five augmenter class methods. 
    #          - This class is used to transform data necessary for training TechTrack's models.
    #          - Imagine that the output of `self.transform()` is fed directly to train the model.
    #          - You will define your own keywords for "**kwargs".
    #          --------------------------------------------------------------------------------
    #          Create your own augmentation method. Use the same structure as the format used below.
    #          For example,
    #
    #          def your_custom_transformation(**kwargs):
    #              # your process
    #              return ...
    #
    #          Name this method appropriately based on its capability. And add docstrings to 
    #          describe its process.
    #          --------------------------------------------------------------------------------
    #          Provide a demonstration and visualizations of these methods in 
    #          `techtrack/notebooks/augmentation.ipynb`.
    
    @staticmethod
    def horizontal_flip(**kwargs):
        """
        Horizontally flip the image and annotated bbox
        
        """
        img = kwargs.get("image")
        anno = kwargs.get("original_annotation")

        new_anno = np.copy(np.array(anno))
        new_anno[:, 1] = 1 - new_anno[:, 1] # Flip the x_center
        return cv2.flip(img, 1), new_anno.tolist()

    @staticmethod
    def gaussian_blur(**kwargs):
        """
        Apply Gaussian blur to the image.
        
        """
        img = kwargs.get("image")
        ksize = kwargs.get("gaussian_ksize") 
        sigmaX = kwargs.get("gaussian_sigmaX") 
        sigmaY = kwargs.get("gaussian_sigmaY") 
        anno = kwargs.get("original_annotation")

        new_anno = np.copy(np.array(anno))
        return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY), new_anno.tolist()


    @staticmethod
    def resize(**kwargs):
        """
        Resize the image.
        
        """
        img = kwargs.get("image")
        dsize = kwargs.get("resize_dsize") 
        fx = kwargs.get("resize_fx")
        fy = kwargs.get("resize_fy")
        interpolation = kwargs.get("resize_interpolation", cv2.INTER_LINEAR) # default interpolation method
        anno = kwargs.get("original_annotation")

        new_anno = np.copy(np.array(anno))
        return cv2.resize(img, dsize, fx, fy, interpolation), new_anno.tolist()

    @staticmethod
    def change_brightness(**kwargs):
        """
        Adjust brightness and contrast of the image.
        
        """
        img = kwargs.get("image")
        value = kwargs.get("brightness_value", 0) # default value as 0 (no change in brightness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        anno = kwargs.get("original_annotation")

        new_anno = np.copy(np.array(anno))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR), new_anno.tolist()

    @staticmethod
    def transform(image, **kwargs):
        """
        Apply random augmentations from the available methods.
        
        Internal Process:
        1. A list of available augmentation functions is created.
        2. The list is shuffled to introduce randomness.
        3. A random number of augmentations is selected.
        4. The selected augmentations are applied sequentially to the image.
        
        :param image: Input image (numpy array)
        :param kwargs: Additional parameters for transformations (if any)
        :return: Augmented image, Augmented annotation
        """
        # Sanity check for input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")
        
        # Apply random augmentations
        augmentations = [Augmenter.horizontal_flip, Augmenter.gaussian_blur, Augmenter.resize, Augmenter.change_brightness]
        random.shuffle(augmentations)
        num_augmentations = random.randint(1, len(augmentations))

        for i in range(num_augmentations):
            augmented_image, augmented_annotation = augmentations[i](image=image, **kwargs)

        return augmented_image, augmented_annotation
        

"""
EXAMPLE RUNNER:

# Create an instance of Augmenter
augmenter = Augmenter()

kwargs = {"image": your_image, # Numpy type
            ... # Add more...
        }

# Apply random transformations
augmented_image = augmenter.transform(**kwargs)

# Display the original and transformed images
cv2.imshow("Original Image", image)
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

