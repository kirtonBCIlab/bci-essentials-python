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
    logger = Logger(Logger.DEBUG)

    # Change the logging level of an existing logger instance e.g. DEBUG
    logger.set_level(Logger.DEBUG)

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
    """
    A custom logger for the 'bci_essentials' package.

    This class provides an easy-to-use interface for logging within the 'bci_essentials'
    package. It supports standard logging levels and allows saving log messages to a
    file.

    Class Properties
    ----------------
    DEBUG : logging.Level
        Debug level logging. This level outputs detailed information, typically 
        of interest only when diagnosing problems.
    INFO : logging.Level
        Info level logging. Confirmation that things are working as expected.
    WARNING : logging.Level
        Warning level logging. An indication that something unexpected happened,
        or indicative of some problem in the near future (e.g., 'disk space low').
        The software is still working as expected.
    ERROR : logging.Level
        Error level logging. Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL : logging.Level
        Critical level logging. A serious error, indicating that the program itself may be unable to continue running.

    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Setting the default logging level to INFO
    __default_level = INFO

    def __init__(self, level=__default_level, name='bci_essentials'):
        """Initializes and configures the logger.

        Parameters
        ----------
        level : logging.Level
            Logging level as per python `logging` module.
            - Default is `INFO`.
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
            self.set_level(level)

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

    def set_level(self, level=__default_level):
        """Sets the logging level for the logger.

        Parameters
        ----------
        level : logging.Level
            Logging level to be set for the logger.
            – Default is `INFO`.

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