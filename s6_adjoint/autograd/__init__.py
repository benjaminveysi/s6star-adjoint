try:
    from .function import S6HamiltonianFunction
    from .module import S6HamiltonianODE
    __all__ = ["S6HamiltonianFunction", "S6HamiltonianODE"]
except (ImportError, AttributeError):
    __all__ = []
