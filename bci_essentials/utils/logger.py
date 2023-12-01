import logging
import datetime


class Logger():
    def __init__(self, level=logging.INFO, name='bci_essentials'):
        """Initializes and configures the logger.

        Parameters
        ----------
        name : str
            Name of the logger, default is 'bci_essentials'.
        level : logging.Level
            Logging level, default is logging.INFO.
        """
        self.logger = logging.getLogger(name)
        self.configure_logger(level)

    def configure_logger(self, level):
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
            # Set the logging level
            self.logger.setLevel(level)

            # Create a console handler that logs to standard output
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging_format)

            # Add the console handler to the logger
            self.logger.addHandler(console_handler)
        else:
            # Update the formatter of existing handlers
            for handler in self.logger.handlers:
                handler.setFormatter(logging_format)

    def save_logs(self, filename=None):
        """Enables saving logs to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to log messages to. If not specified, a default
            filename with a timestamp is used.
        """
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-bci_essentials.log'

        file_handler = logging.FileHandler(filename)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s : %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)