"""
This module contains functions for saving and loading sessions.
"""

import os
import pickle

from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

# Define a module-level variable for the session save path
# This is the local path to the session_saves directory in the package root
session_save_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "session_saves"
)


def save_classifier(classifier, filename):
    """
    Saves a classifier to a file.

    Parameters
    ----------
    classifier : object
        The classifier object to be saved.
    filename : str
        The name of the file to save the classifier to.
    """

    # Join filename to session_saves directory in package root
    filepath = os.path.join(session_save_path, filename)

    with open(filepath, "wb") as f:
        pickle.dump(classifier, f)
    logger.debug("Saved classifier to %s", filepath)


def load_classifier(filename):
    """
    Loads a classifier from a file.

    Parameters
    ----------
    filename : str
        The name of the file to load the classifier from.

    Returns
    -------
    classifier : object
        The classifier object that was loaded.
    """

    # Join filename to session_saves directory in package root
    filepath = os.path.join(session_save_path, filename)

    with open(filepath, "rb") as f:
        classifier = pickle.load(f)
    logger.debug("Loaded classifier from %s", filepath)
    return classifier
