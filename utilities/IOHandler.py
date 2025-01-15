#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# File handler for application's I/O.
#
# ------------------------------------------------------------------------------
# Author:
# Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

# ! Under development !
# Manage IO with levels of message priority that can be set by the user.
# Ok da Mattia, ma dopo tesi e al suo ritorno.


import logging


class LoggerHandler:
    def __init__(self, log_name=__file__, log_level=logging.DEBUG) -> None:
        self.log_level = log_level
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        if log_level == logging.DEBUG:
            self.formatter = logging.Formatter(
                "%(asctime)s : %(name)s : %(levelname)s \n %(message)s \n"
            )
        else:
            self.formatter = logging.Formatter("%(message)s \n")

    def initialise_console_handler(self):
        new_streamHandler = logging.StreamHandler()
        new_streamHandler.setLevel(self.log_level)
        new_streamHandler.setFormatter(self.formatter)

        # add to logger
        self.logger.addHandler(new_streamHandler)

    def debug(self, message):
        return self.logger.debug(message)

    def info(self, message):
        return self.logger.info(message)

    def warning(self, message):
        return self.logger.warning(message)

    def error(self, message):
        return self.logger.error(message)

    def critical(self, message):
        return self.logger.critical(message)


log = LoggerHandler("nome", logging.INFO)

log.initialise_console_handler()
# 'application' code
log.debug("debug message")
log.info("info message")
log.warning("warn message")
log.error("error message")
log.critical("critical message")

log2 = LoggerHandler()

log2.initialise_console_handler()
# 'application' code
log2.debug("debug message")
log2.info("info message")
log2.warning("warn message")
log2.error("error message")
log2.critical("critical message")
