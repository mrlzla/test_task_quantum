from abc import ABC, abstractmethod

import torch
import torchvision
import torch.functional as F
import numpy as np
from torch import nn
from sklearn.ensemble import RandomForestClassifier


class MNISTCNN(nn.Module):
    """
    Simple CNN for MNIST.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class DigitClassificationInterface(ABC):
    """
    Interface for Digit Classification models.

    This abstract class defines the interface for models used in
    digit classification tasks.Subclasses must implement
    the abstract methods defined here.

    Attributes:
        model_type:
            A string representing the type of the digit classification model.
        num_classes:
            The number of classes for digit classification,
            which is typically 10 for digits 0-9.
    """
    model_type = None

    def __init__(self):
        super().__init__()
        self.num_classes = 10

    def normalize_image(self, image:np.array):
        """
        Normalize an input image.

        This method normalizes the pixel values of an input image
        to a range of [-1, 1].

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            A normalized numpy array of the input image.
        """
        return ((image/255 - 0.5) / 0.5).astype(np.float32)

    @abstractmethod
    def init_model(self):
        """
        Initialize the digit classification model.

        This method should be implemented in subclasses to initialize
        the specific model.
        """
        raise NotImplementedError()

    @abstractmethod
    def preprocess_image(self, image:np.array):
        """
        Preprocess an input image for digit classification.

        Subclasses should implement this method to perform any
        necessary preprocessing on the input image before classification.

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            Preprocessed image data ready for classification.
        """
        raise NotImplementedError()

    def train(self):
        """
        Train the digit classification model (not implemented).

        This method should not be implemented in subclasses,
        as it's not used in the interface. Subclasses may raise a
        "NotImplementedError" to indicate that training is not supported.
        """
        raise NotImplementedError("No implementation for train")

    @abstractmethod
    def predict(self, image:np.array) -> int:
        """
        Predict the digit in the input image.

        Subclasses should implement this method to make predictions
        based on the input image.

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            An integer representing the predicted digit.
        """
        raise NotImplementedError()


class CNNClassification(DigitClassificationInterface):
    """
    CNN Implementation of DigitClassificationInterface.

    This class implements the DigitClassificationInterface using a simple
    PyTorch model for inference.It's designed for digit classification and
    uses a Convolutional Neural Network (CNN) model. Note that there is no
    'train' method implemented, as training is not supported in this class.

    Attributes:
        model_type (str):
            A string representing the type of the digit classification model ('cnn').
        device:
            The device (CPU or CUDA) used for model inference.
        model:
            The CNN model used for digit classification.

    Methods:
        init_model: Initialize the CNN model.
        preprocess_image: Preprocess an input image for digit classification.
        predict: Predict the digit in the input image.
    """
    model_type = 'cnn'

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_model()

    def init_model(self):
        self.model = MNISTCNN().to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        """
        Preprocess an input image for digit classification.

        This method normalizes and formats the input image
        for use with the CNN model.

        Args:
            image: A numpy array representing the input image.

        Returns:
            Preprocessed image data in PyTorch format ready for classification.
        """
        inp = self.normalize_image(image)
        inp_torch = torch.tensor(inp).to(self.device)
        inp_torch = inp_torch.unsqueeze(0).unsqueeze(0)
        return inp_torch

    def predict(self, image:np.array) -> int:
        """
        Predict the digit in the input image.

        This method takes a preprocessed image and uses the CNN
        model to make a prediction.

        Args:
            image: A numpy array representing the input image.

        Returns:
            An integer representing the predicted digit.
        """
        inp_torch = self.preprocess_image(image)
        prediction = self.model(inp_torch).squeeze(0)
        class_id = prediction.argmax().item()
        return class_id


class RFClassification(DigitClassificationInterface):
    """
    Random Forest (RF) Implementation of DigitClassificationInterface.

    This class implements the DigitClassificationInterface using
    a Random Forest model for digit classification. It's designed for
    digit classification tasks using a scikit-learn RandomForestClassifier.

    Attributes:
        model_type (str):
            A string representing the type of the digit classification model ('rf').
        model:
            The Random Forest model used for digit classification.

    Methods:
        init_model: Initialize the Random Forest model.
        train: Train the Random Forest model with random data (for demonstration purposes).
        preprocess_image: Preprocess an input image for digit classification.
        predict: Predict the digit in the input image.
    """
    model_type = 'rf'

    def __init__(self):
        super().__init__()
        self.init_model()

    def init_model(self):
        self.model = RandomForestClassifier()
        self.train()

    def train(self):
        """
        Train the Random Forest model with random data (for demonstration purposes).

        This method generates random training data and labels, normalizes the data,
        and fits the Random Forest model to the training data.
        """
        X = np.random.randint(0, 255, size=(1000, 28*28))
        X = self.normalize_image(X)
        y = np.random.randint(0, 9, size=(1000))
        self.model.fit(X, y)

    def preprocess_image(self, image):
        """
        Preprocess an input image for digit classification.

        This method normalizes and flattens the input image for use
        with the Random Forest model.

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            Preprocessed image data in flattened format ready for classification.
        """
        inp = self.normalize_image(image)
        return inp.flatten()

    def predict(self, image:np.array) -> int:
        """
        Predict the digit in the input image.

        This method takes a preprocessed image and uses the Random Forest
        model to make a prediction.

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            An integer representing the predicted digit.
        """
        inp = self.preprocess_image(image)[None]
        prediction = self.model.predict(inp)[0]
        return prediction


class RandomClassification(DigitClassificationInterface):
    """
    Random Model Implementation of DigitClassificationInterface.

    This class implements the DigitClassificationInterface with
    a random model for digit classification. It generates random predictions
    for digit classification tasks.

    Attributes:
        model_type: A string representing the type of the digit classification model ('rand').

    Methods:
        init_model:
            Initialize the random model (not a real model).
        preprocess_image:
            Preprocess an input image for digit classification (no actual preprocessing).
        predict:
            Generate a random prediction for the input image.
    """
    model_type = 'rand'

    def __init__(self):
        super().__init__()
        self.init_model()

    def init_model(self):
        """
        Initialize the random model (not a real model).

        This method does not set up a real model but is provided
        for demonstration purposes.
        """
        self.model = np.random.randint

    def preprocess_image(self, image):
        return image

    def predict(self, image:np.array) -> int:
        """
        Generate a random prediction for the input image.

        This method generates a random integer between 0 and 9 as
        a prediction for digit classification.

        Args:
            image (np.array):
                A numpy array representing the input image (not used in prediction).

        Returns:
            An integer representing a random prediction between 0 and 9.
        """
        prediction = self.model(0, 9)
        return prediction


class DigitClassifier:
    """
    Digit Classifier using various model types.

    This class allows you to create a digit classifier instance with a
    specified model type. It dynamically selects the model based on
    the provided 'model_type' and delegates the prediction to the chosen model.

    Attributes:
        model: An instance of the selected digit classification model.

    Methods:
        predict: Predict the digit in the input image using the selected model.
    """
    def __init__(self, model_type:str = 'cnn'):
        for _, cand in globals().items():
            if not isinstance(cand, type):
                continue
            if not issubclass(cand, DigitClassificationInterface):
                continue
            if cand.model_type == model_type:
                self.model = cand()
                break
        else:
            raise ValueError(f"There is no model with model_type {model_type}")

    def predict(self, image: np.array) -> int:
        """
        Predict the digit in the input image using the selected model.

        Delegates the prediction to the selected digit classification model.

        Args:
            image (np.array): A numpy array representing the input image.

        Returns:
            An integer representing the predicted digit.
        """
        return self.model.predict(image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="DigitClassifier")
    parser.add_argument("--model_type",
                        type=str,
                        help="A folder where MNIST will be stored",
                        default="cnn")
    parser.add_argument("--output_folder",
                        type=str,
                        help="A folder where MNIST will be stored",
                        default="./output_data")
    args = parser.parse_args()
    data = torchvision.datasets.MNIST(args.output_folder, download=True)
    digit_classifier = DigitClassifier(model_type=args.model_type)
    image = data.data[0].numpy()
    print(digit_classifier.predict(image))
