"""
Core Gemini functionality: image generation and multimodal oracle reasoning.
"""
import base64
import os
from datetime import datetime
from pathlib import Path

import httpx

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ORACLE_MODEL = "gemini-2.5-pro"

# Max output tokens for oracle responses — Gemini 2.5 Pro supports up to 65536.
# Set high to allow large responses for complex tasks (code review, SVG gen, etc.).
ORACLE_MAX_OUTPUT_TOKENS = 65536


def load_dotenv(env_path: Path | str | None = None):
    """Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. If None, searches cwd and common Rhode locations.
    """
    def _load_file(p: Path) -> None:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value

    if env_path is not None:
        _load_file(Path(env_path))
        return

    # Search order: current directory, then common Rhode deployment locations
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parents[4] / ".env",       # repo root if installed in-tree
        Path.home() / "Code" / "Rhode" / ".env",  # Rhode project standard location
    ]
    for candidate in candidates:
        _load_file(candidate)


def _get_mime_type(path: str) -> str:
    """Determine MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    mime_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_type_map.get(ext, "image/png")


def _image_part_from_path(path: str):
    """Create a Gemini Part from an image file path with SDK compatibility fallbacks."""
    if types is None:
        raise RuntimeError("google-genai package not found. Install with: pip install google-genai")

    with open(path, "rb") as f:
        img_bytes = f.read()

    mime_type = _get_mime_type(path)
    return _image_part_from_bytes(img_bytes, mime_type)


def _image_part_from_bytes(img_bytes: bytes, mime_type: str):
    """Create a Gemini Part from image bytes with SDK compatibility fallbacks."""
    if types is None:
        raise RuntimeError("google-genai package not found. Install with: pip install google-genai")

    image_part = None

    # Try different SDK methods for compatibility
    if hasattr(types.Part, "from_bytes"):
        try:
            image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        except Exception:
            pass

    if image_part is None and hasattr(types.Part, "from_image"):
        try:
            image_part = types.Part.from_image(image=img_bytes, mime_type=mime_type)
        except Exception:
            pass

    if image_part is None:
        try:
            blob_cls = getattr(types, "Blob", None)
            if blob_cls:
                image_part = types.Part(inline_data=blob_cls(data=img_bytes, mime_type=mime_type))
            else:
                image_part = types.Part(
                    inline_data={"mime_type": mime_type, "data": img_bytes}
                )
        except Exception as e:
            raise RuntimeError(f"Failed to construct image part. SDK version might be incompatible. Error: {e}")

    return image_part


def _setup_ssl_bypass():
    """Monkey-patch httpx to disable SSL verification."""
    _original_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        return _original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init

    _original_async_client_init = httpx.AsyncClient.__init__
    def _patched_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        return _original_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_client_init


def oracle_call(
    prompt: str,
    *,
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
    context: str | None = None,
    system_instruction: str | None = None,
    max_output_tokens: int | None = None,
    model: str = DEFAULT_ORACLE_MODEL,
) -> str:
    """
    Send a reasoning request to Gemini and return the text response.

    Supports optional image input (via path or raw bytes), large context text
    (up to ~1M tokens with gemini-2.5-pro), and a system instruction.

    Args:
        prompt: Text prompt for the reasoning request.
        image_path: Optional path to an image file to analyze.
        image_bytes: Optional raw image bytes (alternative to image_path).
        image_mime_type: MIME type of the image when using image_bytes.
        context: Additional context text prepended before the prompt.
                 Can be very large (up to ~1M tokens).
        system_instruction: Optional system-level instruction for the model.
        max_output_tokens: Maximum tokens in the response (default: ORACLE_MAX_OUTPUT_TOKENS=65536).
        model: Gemini model ID to use (default: gemini-2.5-pro).

    Returns:
        The text response from Gemini.
    """
    if genai is None:
        raise RuntimeError("google-genai package not found. Install with: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_TEXT_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GEMINI_TEXT_API_KEY) env var is not set.")

    client = genai.Client(api_key=api_key)

    # Build contents array: image first (if provided), then context + prompt
    contents = []

    # Add image part if provided
    has_image = image_path is not None or image_bytes is not None
    if has_image:
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            img_part = _image_part_from_path(image_path)
        else:
            img_part = _image_part_from_bytes(image_bytes, image_mime_type)
        contents.append(img_part)

    # Add context text if provided
    if context:
        contents.append(types.Part.from_text(text=context))

    # Add the main prompt
    contents.append(types.Part.from_text(text=prompt))

    # Build generation config — always set max_output_tokens to allow large responses
    gen_config_kwargs: dict = {
        "max_output_tokens": max_output_tokens if max_output_tokens is not None else ORACLE_MAX_OUTPUT_TOKENS,
    }
    if system_instruction is not None:
        gen_config_kwargs["system_instruction"] = system_instruction

    gen_config = None
    config_cls = getattr(types, "GenerateContentConfig", None)
    if config_cls is not None:
        try:
            gen_config = config_cls(**gen_config_kwargs)
        except Exception:
            gen_config = None

    # Try direct parts list first, then wrapped in Content object
    attempts = [contents]
    try:
        content_cls = getattr(types, "Content", None)
        if content_cls is not None:
            attempts.append([content_cls(role="user", parts=contents)])
    except Exception:
        pass

    response = None
    last_error = None

    for attempt_contents in attempts:
        try:
            call_kwargs: dict = {
                "model": model,
                "contents": attempt_contents,
            }
            if gen_config is not None:
                call_kwargs["config"] = gen_config
            response = client.models.generate_content(**call_kwargs)
            last_error = None
            break
        except Exception as e:
            last_error = e
            continue

    if last_error is not None or response is None:
        raise RuntimeError(f"Gemini API request failed: {last_error}")

    if not response.candidates:
        raise RuntimeError("No candidates returned from Gemini.")

    # Extract text response
    candidate = response.candidates[0]

    # Check finish reason — warn on truncation but still return partial content
    finish_reason = getattr(candidate, "finish_reason", None)
    finish_reason_str = str(finish_reason) if finish_reason is not None else ""

    parts = getattr(response, "parts", None)
    if parts is None and candidate.content:
        parts = candidate.content.parts

    # Collect whatever text was returned (may be partial on MAX_TOKENS)
    text_parts = []
    if parts:
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)

    if not text_parts:
        # Gemini 2.5 Pro uses thinking tokens — if max_output_tokens is too low,
        # the model exhausts the budget on internal reasoning with nothing left for output.
        if "MAX_TOKENS" in finish_reason_str:
            raise RuntimeError(
                "Response was empty: max_output_tokens budget was fully consumed by model "
                "thinking before any output could be produced. Increase max_output_tokens "
                f"(current effective limit caused {finish_reason_str})."
            )
        reason_hint = f" (finish_reason={finish_reason_str})" if finish_reason_str else ""
        raise RuntimeError(f"No text found in response{reason_hint}. Response: {response}")

    result = "\n".join(text_parts)

    # Append a truncation notice if response was cut off
    if "MAX_TOKENS" in finish_reason_str:
        result += "\n\n[Note: Response was truncated at the max_output_tokens limit. Increase max_output_tokens if you need the full response.]"

    return result


def generate_image(
    prompt: str,
    out_path: str | None = None,
    *,
    input_image_path: str | None = None,
    input_image_bytes: bytes | None = None,
    input_image_mime_type: str = "image/png",
    style_ref_image_path: str | None = None,
    style_ref_image_bytes: bytes | None = None,
    style_ref_mime_type: str = "image/png",
    model: str = DEFAULT_IMAGE_MODEL,
    no_ssl_verify: bool = False,
    return_bytes: bool = False,
) -> str | tuple[bytes, str]:
    """
    Generate an image from a text prompt, optionally revising an existing image.

    Args:
        prompt: The text description of the image to generate or changes to make
        out_path: Output file path. If None, auto-generates based on timestamp.
        input_image_path: Optional path to an existing image to revise/edit
        input_image_bytes: Optional raw bytes of an image to revise (alternative to path)
        input_image_mime_type: MIME type when using input_image_bytes
        style_ref_image_path: Optional path to a style reference image
        style_ref_image_bytes: Optional raw bytes of a style reference image
        style_ref_mime_type: MIME type when using style_ref_image_bytes
        model: Gemini model ID to use
        no_ssl_verify: If True, disable SSL certificate verification
        return_bytes: If True, return (image_bytes, mime_type) instead of saving to file

    Returns:
        If return_bytes is False: The path to the saved image file
        If return_bytes is True: Tuple of (image_bytes, mime_type)
    """
    if genai is None:
        raise RuntimeError("google-genai package not found. Install with: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_TEXT_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GEMINI_TEXT_API_KEY) env var is not set.")

    # Handle SSL verification bypass
    if no_ssl_verify:
        _setup_ssl_bypass()

    client = genai.Client(api_key=api_key)

    # Determine if we have an input image or style reference
    has_input_image = input_image_path is not None or input_image_bytes is not None
    has_style_ref = style_ref_image_path is not None or style_ref_image_bytes is not None

    # Auto-generate output path if not provided and not returning bytes
    if out_path is None and not return_bytes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if has_input_image:
            prefix = "revised"
        elif has_style_ref:
            prefix = "styled"
        else:
            prefix = "generated"
        out_path = f"{prefix}_{timestamp}.png"

    # Build contents array - images first, then text (per SDK pattern)
    contents = []
    image_index = 1

    # Add input image if provided (for editing)
    if has_input_image:
        if input_image_path:
            if not os.path.exists(input_image_path):
                raise FileNotFoundError(f"Input image not found: {input_image_path}")
            image_part = _image_part_from_path(input_image_path)
        else:
            image_part = _image_part_from_bytes(input_image_bytes, input_image_mime_type)
        contents.append(image_part)
        input_image_index = image_index
        image_index += 1

    # Add style reference image if provided
    if has_style_ref:
        if style_ref_image_path:
            if not os.path.exists(style_ref_image_path):
                raise FileNotFoundError(f"Style reference image not found: {style_ref_image_path}")
            style_part = _image_part_from_path(style_ref_image_path)
        else:
            style_part = _image_part_from_bytes(style_ref_image_bytes, style_ref_mime_type)
        contents.append(style_part)
        style_image_index = image_index
        image_index += 1

    # Build the prompt based on what images we have
    if has_input_image and has_style_ref:
        # Both input image and style reference
        structured_prompt = f"""IMAGE ROLES:
[{input_image_index}] = Input image to be edited/modified.
[{style_image_index}] = Style reference image. Use this image's visual style, color palette, artistic treatment, and aesthetic.

TASK:
Edit the input image [{input_image_index}] according to the instructions below, applying the style from [{style_image_index}].

INSTRUCTIONS:
{prompt}

STYLE RULES:
- Apply the visual style, color palette, and artistic treatment from [{style_image_index}]
- Match the mood, lighting quality, and aesthetic of the style reference
- Blend the style seamlessly with the content

OUTPUT:
Generate only the modified image."""
        contents.append(types.Part.from_text(text=structured_prompt))

    elif has_input_image:
        # Only input image (editing mode)
        structured_prompt = f"""IMAGE ROLES:
[1] = Reference image to be edited/modified. This defines the base composition, style, colors, and content.

TASK:
Edit the reference image [1] according to the following instructions.

REQUESTED CHANGES:
{prompt}

PRESERVATION RULES:
- Keep the same visual style, color palette, and artistic treatment from [1]
- Maintain the same composition and layout unless changes are specifically requested
- Preserve lighting direction and quality from the original
- Keep aspect ratio and proportions consistent

GENERATION RULES:
- Use [1] as the direct base - this is an EDIT operation, not new generation
- Apply only the requested modifications
- Blend any changes seamlessly with the existing style

OUTPUT:
Generate only the modified image."""
        contents.append(types.Part.from_text(text=structured_prompt))

    elif has_style_ref:
        # Only style reference (generation with style)
        structured_prompt = f"""IMAGE ROLES:
[1] = Style reference image. Use this image's visual style, color palette, artistic treatment, and aesthetic.

TASK:
Generate a NEW image based on the description below, using the style from [1].

DESCRIPTION:
{prompt}

STYLE RULES:
- Apply the visual style, color palette, and artistic treatment from [1]
- Match the mood, lighting quality, and aesthetic of the style reference
- Generate new content that fits naturally with this style

OUTPUT:
Generate only the new styled image."""
        contents.append(types.Part.from_text(text=structured_prompt))

    else:
        # Text-only generation
        contents.append(types.Part.from_text(
            text=f"Generate an image based on the following description:\n\n{prompt}"
        ))

    # Try multiple content encodings for SDK compatibility
    attempts = []

    # Attempt 1: Direct parts list
    attempts.append(contents)

    # Attempt 2: Wrapped in Content object
    try:
        content_cls = getattr(types, "Content", None)
        if content_cls is not None:
            attempts.append([content_cls(role="user", parts=contents)])
    except Exception:
        pass

    response = None
    last_error = None

    for attempt_contents in attempts:
        try:
            response = client.models.generate_content(
                model=model,
                contents=attempt_contents,
            )
            last_error = None
            break
        except Exception as e:
            last_error = e
            continue

    if last_error is not None or response is None:
        raise RuntimeError(f"Gemini API request failed: {last_error}")

    if not response.candidates:
        raise RuntimeError("No candidates returned from Gemini.")

    # Extract image data
    parts = None
    candidate = response.candidates[0]

    parts = getattr(response, "parts", None)
    if parts is None and candidate.content:
        parts = candidate.content.parts

    if not parts:
        raise RuntimeError(f"No parts returned. Response: {response}")

    image_bytes = None
    image_mime = "image/png"
    text_response = None

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            inline = getattr(part, "inlineData", None)

        if inline is not None:
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or getattr(inline, "mimeType", "")
            if data is not None and str(mime).startswith("image/"):
                if isinstance(data, bytes):
                    image_bytes = data
                elif isinstance(data, str):
                    image_bytes = base64.b64decode(data)
                image_mime = str(mime)
                break

        text = getattr(part, "text", None)
        if text:
            text_response = text

    if not image_bytes:
        error_msg = "No image data found in response."
        if text_response:
            error_msg += f" Model said: {text_response}"
        raise RuntimeError(error_msg)

    if return_bytes:
        return image_bytes, image_mime

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)

    return out_path
