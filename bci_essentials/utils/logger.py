"""Logger Configuration for bci-essentials package.

It provides functions to configure python's logging module with default
settings for the log format and date format. It also provides a function
to enable logging to a file with a specified filename.

"""

import logging
import datetime


# # Basic logging configuration
# def setup_logging(level=logging.INFO):
#     """Set up the logging configuration for the package.

#     This function configures the basic logging settings for the package. It sets
#     the logging level, format, and date format. It also demonstrates how to add
#     a file handler for logging to a file, which is commented out by default.

#     Parameters
#     ----------
#     level : logging.Level, *optional*
#         The logging level to be set for the logger.
#         – Default is logging.INFO.

#     Returns
#     -------
#     None

#     """
#     logging.basicConfig(
#         level=level,
#         format='%(asctime)s - %(levelname)s - %(name)s : %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )


# # # Logging configuration with a check for existing handlers
# def setup_logging(level=logging.INFO):
#     """Set up the logging configuration for the package.

#     This function configures the basic logging settings for the package. It sets
#     the logging level, format, and date format. It also demonstrates how to add
#     a file handler for logging to a file, which is commented out by default.

#     Parameters
#     ----------
#     level : logging.Level, *optional*
#         The logging level to be set for the logger.
#         – Default is logging.INFO.

#     Returns
#     -------
#     None

#     Examples
#     --------
#     To set up logging with the default level (INFO):

#     >>> import logger
#     >>> logger.setup_logging()

#     To set up logging with a DEBUG level:

#     >>> logger.setup_logging(logging.DEBUG)

#     """
#     # Get the current logger in order to check its level
#     root_logger = logging.getLogger()
    
#     # Configure basic logging settings
#     # Set the level of the logger if it is not already set
#     if not root_logger.hasHandlers():
#         logging.basicConfig(
#             level=level,
#             format='%(asctime)s - %(levelname)s - %(name)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
#         )
#     else:
#         # Set the formatter for existing handlers
#         formatter = logging.Formatter(
#             '%(asctime)s - %(levelname)s - %(name)s : %(message)s', '%Y-%m-%d %H:%M:%S'
#         )
#         for handler in root_logger.handlers:
#             handler.setFormatter(formatter)


# Logging configuration with a check for existing handlers and set package specific logger settings
def setup_logging(level=logging.INFO):
    """Set up logging for the 'bci_essentials' package.

    This function configures logging specifically for the 'bci_essentials' package.
    It respects any pre-existing configurations set by the user. 
    If no custom configuration is detected, it sets a default level and format.

    Parameters
    ----------
    level : logging.Level, *optional*
        The logging level to be set for the 'bci_essentials' logger.
        - Default is `logging.INFO`.

    Notes
    -----
    The logger for 'bci_essentials' will have its level set to `level` only if no
    handlers have been set by the user. If handlers are already present, it implies
    that the user has configured logging, and this function will not alter the level.

    Examples
    --------
    Set up default logging (to INFO level) for 'bci_essentials':
    >>> import logger
    >>> logger.setup_logging()

    Package Users can set a different logging level for 'bci_essentials' as follows:
    >>> import logging
    >>> logger = logging.getLogger('bci_essentials')
    >>> logger.setLevel(logging.DEBUG)
    This can be set-up before importing the package or after importing it.
    Furthermore, the logging level and thus behaviour of logging for the package
    can be changed at any time

    """
    # Set the format for log messages
    logging_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Fetch the logger for 'bci_essentials'
    logger = logging.getLogger('bci_essentials')

    # Check if this logger already has handlers set up
    if not logger.hasHandlers():
        # No existing handlers, so set up a new console handler
        logger.setLevel(level)

        # Create a console handler that logs to standard output
        console_handler = logging.StreamHandler()

        # Define the format for log messages
        console_formatter = logging_format

        # Apply the formatter to the console handler
        console_handler.setFormatter(console_formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)
    else:
        # Handlers exist, so we only update the formatter of existing handlers
        # This respects any level or handlers that the user might have already set
        formatter = logging_format
        for handler in logger.handlers:
            handler.setFormatter(formatter)


def save_logs(filename=None):
    """Enables saving logs logging to a file.

    Enable logging to a file with an option to specify the filename.
    If no filename is provided, a default filename with a timestamp is used.

    Parameters
    ----------
    filename : str, *optional*
        The name of the file to log messages to.
        – Default is `None`, which will result in a default filename
        that consists of the current timestamp and the string
        "bci_essentials.log". For example:
            20111231-235959-bci_essentials_python.log

    Returns
    -------
    `None`

    Examples
    --------
    To enable file logging with a default timestamped filename:
    >>> logger.enable_file_logging()

    To enable file logging with a custom filename:
    >>> import logger
    >>> logger.enable_file_logging('custom_logfile.log')

    """
    # Set default filename if none is provided
    if filename is None:
        filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + \
            '-bci_essentials.log'

    # Create a FileHandler instance to handle file-based logging
    file_handler = logging.FileHandler(filename)

    # Create a formatter for log messages in the file
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s : %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(file_formatter)

    # Add the file handler to the root logger, enabling file-based logging
    logging.getLogger('').addHandler(file_handler)
