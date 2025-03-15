#Exceptions for the SimFin API wrapper

class SimFinError(Exception):
    #Base exception for SimFin errors
    pass

class ResourceNotFoundError(SimFinError):
    #Exception raised when a resource is not found
    pass

class InvalidParameterError(SimFinError):
    #Exception raised when an invalid parameter is provided
    pass

class RateLimitError(SimFinError):
    #Exception raised when the API rate limit is exceeded
    pass

class NetworkError(SimFinError):
    #Exception raised when there's a network error
    pass

class ParsingError(SimFinError):
    #Exception raised when there's an error parsing the API response
    pass 