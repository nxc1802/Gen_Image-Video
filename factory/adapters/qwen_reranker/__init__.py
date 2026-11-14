"""Original Qwen3 reranker through one explicit vLLM /score protocol."""
import math

from factory_contracts.errors import InvalidRequestError, ModelUnavailableError
from factory.core.model_adapter import ModelAdapter, PreparedCall
from factory.engines.vector.contracts import RerankResponse


class Adapter(ModelAdapter):
    capabilities = frozenset({"vector.reranking"})
    supported_backends = frozenset({"vllm"})

    @staticmethod
    def texts(request):
        return [item if isinstance(item,str) else item.get("text") for item in request.documents]

    def validate_request(self, model, request, capability):
        if not request.query.strip() or not request.documents or any(not isinstance(text,str) or not text.strip() for text in self.texts(request)):
            raise InvalidRequestError("Reranking requires a query and nonempty document text.")
        if request.top_n is not None and (type(request.top_n) is not int or not 1 <= request.top_n <= len(request.documents)):
            raise InvalidRequestError("top_n must be within the candidate document count.")

    def prepare(self, context, request, capability):
        return PreparedCall(capability, {"model":context.binding.model,"text_1":request.query,"text_2":self.texts(request)},
            {"documents":request.documents,"top_n":request.top_n})

    def infer(self, context, prepared):
        return context.post_json("/score", prepared.payload)

    def normalize_output(self, context, prepared, output):
        try:
            if output.get("model") != context.binding.model:
                raise ValueError("Scoring backend returned another model")
            rows, documents = output["data"], prepared.metadata["documents"]
            if len(rows) != len(documents) or {row["index"] for row in rows} != set(range(len(documents))):
                raise ValueError("Scoring backend returned duplicate/missing indices")
            results = []
            for row in rows:
                index, score = row["index"], row["score"]
                if type(index) is not int or type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Scoring backend returned an invalid score")
                results.append({"index":index,"document":documents[index],"relevance_score":score})
            results.sort(key=lambda item:(-item["relevance_score"],item["index"]))
            if prepared.metadata["top_n"] is not None:
                results = results[:prepared.metadata["top_n"]]
            return RerankResponse(model=context.binding.model,results=results)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelUnavailableError("Reranking backend returned an invalid result; no lexical fallback was used.") from exc
