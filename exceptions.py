class ModelInfeasible(Exception):
    def __str__(self) -> str:
        return "the model is infeasible, please check constraints"


class SolutionNotFound(Exception):
    def __str__(self) -> str:
        return "solution not found, please increase timeout limit and retry"
