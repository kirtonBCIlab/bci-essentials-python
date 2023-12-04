"""Utility for logging within BCI-Essentials.

This module provides a `Logger` class that allows for easy configuration of
logging settings for the 'bci_essentials' package. It uses the `logging` module
from Python's standard library to handle logging functionality.

The `Logger` class provides methods to initialize and configure the logger,
set the logging level, and format log messages.

Example usage:
    from logger import Logger

    # Create a logger instance with default settings (INFO level)
    logger = Logger()

    # Create a logger instance with modified logging level e.g. DEBUG
    import logging
    logger = Logger(level=logging.DEBUG)

    # Change the logging level of an existing logger instance
    import logging
    logger.setLevel(logging.DEBUG)

    # Start saving logs to a file
    # NOTE: This will save all logs AFTER this function is called
    logger.start_saving()

    # Log an informational message
    logger.logger.info("This is an informational message")

    # Log a warning message
    logger.logger.warning("This is a warning message")

    # Log an error message
    logger.logger.error("This is an error message")

    # Log a critical message
    logger.logger.critical("This is a critical message")

"""

import logging
import datetime


class Logger():
    def __init__(self, level=logging.INFO, name='bci_essentials'):
        """Initializes and configures the logger.

        Parameters
        ----------
        level : logging.Level
            Logging level as per python `logging` module.
            - Default is `logging.INFO`.
        name : str, *optional*
            Name of the logger. **IMPORTANT**: When Logger() is instantiated
            outside of 'bci_essentials' package (e.g. in a user script),
            **DO NOT** change the default value of this parameter. It is set to
            'bci_essentials' by default, and is used by other modules to set
            set module-specific loggers. Changing the default value will break
            logging behaviour in other modules, which will effectively cause
            log messages to be lost.
            - Default is "bci_essentials".

        """
        self.logger = logging.getLogger(name)
        self.__configure(level)

    def __configure(self, level):
        """Configures the logger with the specified level and format.

        Parameters
        ----------
        level : logging.Level
            Logging level to be set for the logger.

        """
        # Set the format for log messages
        logging_format = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Check if this logger already has handlers set up
        if not self.logger.hasHandlers():
            # The logger does not have any handlers set up yet
            # Set the logging level
            self.setLevel(level)

            # Create a console handler that logs to standard output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging_format)

            # Add the console handler to the logger
            self.logger.addHandler(console_handler)
        else:
            # The logger already has handlers set up
            # Only update the formatter of existing handlers
            for handler in self.logger.handlers:
                handler.setFormatter(logging_format)

    def setLevel(self, level):
        """Sets the logging level for the logger.

        Parameters
        ----------
        level : logging.Level
            Logging level to be set for the logger.

        """
        # Set the level for the logger
        self.logger.setLevel(level)

        # Also set the level for all handlers of the logger
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def start_saving(self, filename=None):
        """Enables saving logs to a file.

        Log messages AFTER this function has been called will be saved
        to a text file using either the specified or default filename.

        **NOTE**: Messages before this function is called will not be saved.

        Parameters
        ----------
        filename : str, *optional*
            Name of the file to log messages to
            - Default is is "YYYYMMDD-HMS-bci_essentials.log".
        """
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-bci_essentials.log'
        self.info(f'Logging to file: {filename}')

        file_handler = logging.FileHandler(filename)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s : %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        """Logs a DEBUG message.

        Parameters
        ----------
        message : str
            Message to be logged at the DEBUG level.

        """
        self.logger.debug(message)

    def info(self, message):
        """Logs an INFO message.

        This is the default logging level.

        Parameters
        ----------
        message : str
            Message to be logged at the INFO level.

        """
        self.logger.info(message)

    def warning(self, message):
        """Logs a WARNING message.

        Parameters
        ----------
        message : str
            Message to be logged at the WARNING level.

        """
        self.logger.warning(message)

    def error(self, message):
        """Logs an ERROR message.

        Parameters
        ----------
        message : str
            Message to be logged at the ERROR level.

        """
        self.logger.error(message)

    def critical(self, message):
        """Logs a CRITICAL message.

        Parameters
        ----------
        message : str
            Message to be logged at the CRITICAL level.

        """
        self.logger.critical(message)