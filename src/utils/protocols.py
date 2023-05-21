from typing import Any, Protocol


class CallableObjectProtocol(Protocol):
    """Protocol for callable objects."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call method of the callable object.

        :param args: Arguments.
        :type args: Any
        :param kwargs: Keyword arguments.
        :type kwargs: Any
        :return: Result of the call.
        :rtype: Any
        """
