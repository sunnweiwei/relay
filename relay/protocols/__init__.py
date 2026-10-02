from .anthropic_messages import AnthropicMessages
from .base import Codec
from .gemini import Gemini
from .openai_chat import OpenAIChat
from .openai_responses import OpenAIResponses

CODECS: tuple[Codec, ...] = (OpenAIResponses(), AnthropicMessages(), OpenAIChat(), Gemini())


def codec_for(path: str) -> Codec | None:
    """The codec managing a conversation endpoint, matched by the path's suffix."""

    return next((codec for codec in CODECS if path.endswith(codec.paths)), None)


__all__ = ["CODECS", "AnthropicMessages", "Codec", "Gemini", "OpenAIChat", "OpenAIResponses", "codec_for"]
