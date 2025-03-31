import unittest
import numpy as np
from bci_essentials.classification.generic_classifier import GenericClassifier
from bci_essentials.utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
logger = Logger(name="test_ready_for_fit")


# Create a minimal implementation of GenericClassifier for testing
class TestableClassifier(GenericClassifier):
    """Minimal implementation of GenericClassifier for testing."""

    def fit(self):
        """Implement required abstract method."""
        pass

    def predict(self, X):
        """Implement required abstract method."""
        pass


class TestClassifierReadiness(unittest.TestCase):
    def setUp(self):
        """Set up a fresh classifier instance for each test."""
        self.classifier = TestableClassifier()

    def test_FALSE_no_data_available(self):
        """Test that classifier reports not ready when no data is available."""
        logger.info("Testing no data available case.")
        # Default initialization has empty X, so no setup needed
        self.assertFalse(self.classifier._check_ready_for_fit())

    def test_FALSE_only_one_class(self):
        """Test that classifier reports not ready when only one class is present."""
        logger.info("Testing only one class case.")
        # Create data with only one class
        self.classifier.X = np.random.rand(
            10, 5, 128
        )  # 10 epochs, 5 channels, 128 samples
        self.classifier.y = np.ones(10)  # All labels are 1

        self.assertFalse(self.classifier._check_ready_for_fit())

    def test_FALSE_insufficient_samples_per_class(self):
        """Test that classifier reports not ready when any class has fewer samples than n_splits."""
        logger.info("Testing insufficient samples per class case.")
        # Create data with two classes but one has too few samples
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(
            8, 5, 128
        )  # 8 epochs, 5 channels, 128 samples
        self.classifier.y = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1]
        )  # 4 samples of each class, less than n_splits=5

        self.assertFalse(self.classifier._check_ready_for_fit())

    def test_FALSE_imbalanced_insufficient_samples(self):
        """Test rejection when one class has enough samples but another doesn't."""
        logger.info("Testing imbalanced insufficient samples case.")
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(10, 5, 128)
        self.classifier.y = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        )  # 6 of class 0, 4 of class 1

        self.assertFalse(self.classifier._check_ready_for_fit())

    def test_FALSE_multiple_classes_some_insufficient(self):
        """Test rejection with multiple classes where some have insufficient samples."""
        logger.info("Testing multiple classes with some insufficient samples case.")
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(15, 5, 128)
        self.classifier.y = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3]
        )  # Classes 0,1: 5 samples (sufficient), Class 2: 4 samples, Class 3: 1 sample (both insufficient)

        self.assertFalse(self.classifier._check_ready_for_fit())

    def test_TRUE_exact_minimum_sufficient_samples(self):
        """Test acceptance when classes have exactly n_splits samples."""
        logger.info("Testing exact minimum samples case.")
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(10, 5, 128)
        self.classifier.y = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )  # Exactly 5 of each class

        self.assertTrue(self.classifier._check_ready_for_fit())

    def test_TRUE_more_than_sufficient_samples(self):
        """Test acceptance that classifier reports ready when we have more than sufficient samples."""
        logger.info("Testing more than sufficient samples case.")
        # Create data with enough samples for all classes
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(
            12, 5, 128
        )  # 12 epochs, 5 channels, 128 samples
        self.classifier.y = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        )  # 6 samples of each class, greater than n_splits=5

        self.assertTrue(self.classifier._check_ready_for_fit())

    def test_TRUE_multiple_classes_sufficient(self):
        """Test acceptance with multiple classes where all have sufficient samples"""
        logger.info("Testing multiple classes with sufficient samples case.")
        self.classifier.n_splits = 5
        self.classifier.X = np.random.rand(15, 5, 128)
        self.classifier.y = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        )

        self.assertTrue(self.classifier._check_ready_for_fit())


if __name__ == "__main__":
    unittest.main()
