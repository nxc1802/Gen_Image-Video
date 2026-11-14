"""Canonical ResponsesEngine compatibility import; helpers are retained for legacy fixtures."""
from factory.engines.responses import ResponsesEngine
from services.text.src.legacy_engine import StreamingReasoningFilter, decouple_think_tokens
TextEngine = ResponsesEngine
