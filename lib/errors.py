"""This module contains all the custom exceptions used in the library."""


class InvalidDistanceMetricException(Exception):
    "Raised when the p parameter in distance metric is not valid."

class InvalidAxesException(Exception):
    "Raised when the axis parameter in distance metric is not valid."
