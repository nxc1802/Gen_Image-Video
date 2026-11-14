"""Explicit Qwen2 architecture registration for native vLLM label reads.

No logits, probabilities, text, scores or checkpoint selection are implemented
here. vLLM's real NextTokenStrategy runs the selected original model on CUDA.
"""


def register():
    from vllm.entrypoints.openai.decisions.strategies import READ_STRATEGIES, NextTokenStrategy

    existing = READ_STRATEGIES.get("Qwen2ForCausalLM")
    if existing is not None and existing is not NextTokenStrategy:
        raise RuntimeError("Conflicting Qwen2 native Decisions strategy registration")
    READ_STRATEGIES["Qwen2ForCausalLM"] = NextTokenStrategy
