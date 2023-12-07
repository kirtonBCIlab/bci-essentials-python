"""Utility for logging within BCI-Essentials.

This module provides a `Logger` wrapper class that allows for easy
configuration of logging settings for the 'bci_essentials' package.
The `Logger` class provides methods to initialize and configure the logger.

It uses the `logging` module from Python's standard library to handle
the logging functionality. The logging levels are the same as the ones
in the `logging` module. They are given here in order of increasing
severity. When the logging level is set to a particular level, all
messages of that level and higher will be logged.

The logging levels can be accessed as class properties of the `Logger`
class. For example, the `Logger.INFO` property can be used to set the
logging level to INFO.

Logging Levels
--------------
DEBUG :
    Debug level logging. This level outputs detailed information,
    typically of interest only when diagnosing problems.
INFO :
    Info level logging. Confirmation that things are working as expected.
    - Default logging level for the 'bci_essentials' package
WARNING :
    Warning level logging. An indication that something unexpected happened,
    or indicative of some problem in the near future (e.g., 'disk space low').
    The software is still working as expected.
ERROR :
    Error level logging. Due to a more serious problem, the software has not
    been able to perform some function.
CRITICAL :
    Critical level logging. A serious error, indicating that the program itself
    may be unable to continue running.


Examples
--------

Example 1A: Creating a Logger object within the 'bci_essentials' package
---------------------------------------------------------------------
The following example shows how to use the `Logger` class within the
'bci_essentials' package, i.e. within package modules. After importing
the `Logger` class, a logger instance is created at the package default
log level of `Logger.INFO`. The name of the logger is set to the name
the module it is being used in (e.g. 'bci_essentials.utils.logger') so
that log messages can be traced back to the module they originated from.
    >>> from bci_essentials.utils.logger import Logger
    >>> logger = Logger(name=__name__)

Example 1B: Creating the Logger object outside of the 'bci_essentials' package
------------------------------------------------------------------------------
The following example shows how to use the `Logger` class to log messages
from bci_essentials in a **User** script. After importing the `Logger` class,
the User can create a logger instance with the default logging level of INFO.

**NOTE**: It is important to **NOT** change the default name of the
logger instance in User scripts. The default name is set to 'bci_essentials'
and is used by other modules to set module-specific loggers.

Here is an example of creating a logger instance at the default logging
level of INFO:
    >>> from bci_essentials.utils.logger import Logger
    >>> logger = Logger()

Here is an example of creating a logger instance at the DEBUG logging level:
    >>> from bci_essentials.utils.logger import Logger
    >>> logger = Logger(Logger.DEBUG)

Example 2: Logging messages
---------------------------
After creating the logger instance, log messages can be recorded in the
same way as the `logging` module, calling methods for debug(), info(),
warning(), error(), and critical() messages. The logger instance will
automatically log the name of the module it is being used in, along with
the logging level and the message.
    >>> logger.debug("This is a DEBUG message")
    >>> logger.info("This is an INFO message")
    >>> logger.logger.warning("This is a warning message")
    >>> logger.logger.error("This is an error message")
    >>> logger.logger.critical("This is a critical message")

This is an example of including a variable in the output at the INFO level:
    >>> logger.info("The number of channels is %s", num_channels)

Example 3: Changing the logging level after Logger instantiation
----------------------------------------------------------------
The logging level can be changed after instantiation using the `set_level()`
method. All messages of that level and higher will be logged going forward
(i.e. after the logging level is changed).

Here is an example of changing the logging level to DEBUG:
    >>> logger.set_level(Logger.DEBUG)

Here is an example of changing the logging level to ERROR:
    >>> logger.set_level(Logger.ERROR)

The logger level can be reset to the default level using:
    >>> logger.set_level()

Example 4: Saving logs to a file
--------------------------------
The logger can be configured to simultaneously save logs to a file using
the `start_saving()` method. This will save all logs **AFTER** this
function is called. *Messages before this function is called will not be
saved*. The filename can be specified as an argument to the function.
If no filename is specified, the default filename will be used, which is
the current date and time in the format "YYYYmmdd-HHMS-bci_essentials.log".

Here is an example of saving logs to a file with the default filename:
    >>> logger.start_saving()

Here is an example of saving logs to a file with a specified filename:
    >>> logger.start_saving(filename="my_log_file.log")

"""

import logging
import datetime


class Logger:
    """
    A custom logger for the 'bci_essentials' package.

    This class provides an easy-to-use interface for logging within the 'bci_essentials'
    package. It supports standard logging levels and allows saving log messages to a
    file.

    Class Properties
    ----------------
    DEBUG : logging.Level
        Debug level logging.
    INFO : logging.Level
        Info level logging.
    WARNING : logging.Level
        Warning level logging.
    ERROR : logging.Level
        Error level logging.
    CRITICAL : logging.Level
        Critical level logging.

    """

    # Define logging levels as a Class Property for easy access.
    # These are the same as the logging levels in the `logging` module.
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Setting the default logging level to INFO
    # Update docstrings on __init__ and set_level() if this is changed
    __default_level = INFO
    # Mapping of logging levels to their string representations
    __level_names = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, level=__default_level, name="bci_essentials"):
        """Initializes and configures the logger.

        Parameters
        ----------
        level : logging.Level
            Logging level as per python `logging` module. This can be set using
            the class properties `Logger.DEBUG`, `Logger.INFO`,
            `Logger.WARNING`, `Logger.ERROR`, and `Logger.CRITICAL`.
            - Default is `Logger.INFO`.
        name : str, *optional*
            Name of the logger. **IMPORTANT**: When Logger() is instantiated
            outside of 'bci_essentials' package (e.g. in a user script),
            **DO NOT** change the default value of this parameter. It is set to
            'bci_essentials' by default, and is used by other modules to set
            set module-specific loggers by inputting `name=__name__`.
            Changing the default name in User files will break behaviour in
            other modules, which will effectively cause log messages to be lost.
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
            fmt="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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
            Logging level as per python `logging` module. This can be set using
            the class properties `Logger.DEBUG`, `Logger.INFO`,
            `Logger.WARNING`, `Logger.ERROR`, and `Logger.CRITICAL`.
            - Default is `Logger.INFO`.

        """
        # Inform the user of the new logging level
        level_name = self.__level_names.get(level, "Unknown Level")
        self.info(f"Setting logging level to {level_name}")

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
            - Default is is "YYYYmmdd-HHMMSS-bci_essentials.log".
        """
        if filename is None:
            filename = (
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-bci_essentials.log"
            )
        self.info(f"Logging to file: {filename}")

        file_handler = logging.FileHandler(filename)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg, *args, **kwargs):
        """
        Logs a DEBUG message.

        Parameters
        ----------
        msg : str
            The log message format string.
        *args
            Arguments merged into msg using string formatting.
        **kwargs
            Additional keyword arguments.

        Examples
        --------
        >>> logger.debug("Debug message with variable: %s", variable_name)
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Logs an INFO message.

        Parameters
        ----------
        msg : str
            The log message format string.
        *args
            Arguments merged into msg using string formatting.
        **kwargs
            Additional keyword arguments.

        Examples
        --------
        >>> logger.info("Info message with variable: %s", variable_name)
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Logs a WARNING message.

        Parameters
        ----------
        msg : str
            The log message format string.
        *args
            Arguments merged into msg using string formatting.
        **kwargs
            Additional keyword arguments.

        Examples
        --------
        >>> logger.warning("Warning message with variable: %s", variable_name)
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Logs an ERROR message.

        Parameters
        ----------
        msg : str
            The log message format string.
        *args
            Arguments merged into msg using string formatting.
        **kwargs
            Additional keyword arguments.

        Examples
        --------
        >>> logger.error("Error message with variable: %s", variable_name)
        """
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Logs a CRITICAL message.

        Parameters
        ----------
        msg : str
            The log message format string.
        *args
            Arguments merged into msg using string formatting.
        **kwargs
            Additional keyword arguments.

        Examples
        --------
        >>> logger.critical("Critical message with variable: %s", variable_name)
        """
        self.logger.critical(msg, *args, **kwargs)
