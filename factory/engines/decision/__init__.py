"""Generic decision dispatch; question formatting and native answers belong to adapters."""

from factory_contracts.errors import InvalidRequestError
from factory.core.model_adapter import Engine


class DecisionEngine(Engine):
    engine = "decision"

    def evaluate(self, request):
        if not request.model or not request.context.strip():
            raise InvalidRequestError("A model and nonempty decision context are required.")
        return self.execute(request, request.task)

    def create(self, request):
        if not request.input.strip() or not request.questions:
            raise InvalidRequestError("Nonempty input and questions required.")
        return self.execute(request, request.task)
