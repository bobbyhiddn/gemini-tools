"""
MCP Server for Gemini AI — image generation and multimodal reasoning.
"""
import base64
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .core import generate_image, oracle_call, load_dotenv, DEFAULT_IMAGE_MODEL, DEFAULT_ORACLE_MODEL


# Load environment variables
load_dotenv()

server = Server("gemini-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Gemini tools."""
    return [
        Tool(
            name="generate_image",
            description="Generate a new image from a text description using Gemini AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_IMAGE_MODEL})",
                        "default": DEFAULT_IMAGE_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="revise_image",
            description="Revise/edit an existing image based on text instructions using Gemini AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instructions describing the changes to make to the image"
                    },
                    "input_image_path": {
                        "type": "string",
                        "description": "Path to the image file to revise. Required if input_image_base64 is not provided."
                    },
                    "input_image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data to revise. Required if input_image_path is not provided."
                    },
                    "input_image_mime_type": {
                        "type": "string",
                        "description": "MIME type of the base64 image (e.g., 'image/png'). Required when using input_image_base64.",
                        "default": "image/png"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the revised image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_IMAGE_MODEL})",
                        "default": DEFAULT_IMAGE_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="style_generate",
            description="Generate a new image in the style of a reference image. Provide a text prompt describing the image content and a path to a style reference image — the generated image will adopt the visual style, color palette, and aesthetic of the reference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image content to generate"
                    },
                    "style_image_path": {
                        "type": "string",
                        "description": "Path to the style reference image whose visual style will be applied"
                    },
                    "style_image_base64": {
                        "type": "string",
                        "description": "Base64-encoded style reference image data. Alternative to style_image_path."
                    },
                    "style_image_mime_type": {
                        "type": "string",
                        "description": "MIME type of the base64 style image (e.g., 'image/png'). Required when using style_image_base64.",
                        "default": "image/png"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the generated image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_IMAGE_MODEL})",
                        "default": DEFAULT_IMAGE_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="gemini_oracle",
            description=(
                "Send a reasoning request to Gemini with optional image and/or large context. "
                "Returns a text response. Supports up to 1M token context window. "
                "Use this for multimodal analysis, document reasoning, image understanding, "
                "or any task requiring Gemini's reasoning capabilities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for the reasoning request"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Optional path to an image file to analyze alongside the prompt"
                    },
                    "image_base64": {
                        "type": "string",
                        "description": "Optional base64-encoded image data (alternative to image_path)"
                    },
                    "image_mime_type": {
                        "type": "string",
                        "description": "MIME type of the image (e.g., 'image/png', 'image/jpeg'). Used with image_path or image_base64.",
                        "default": "image/png"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context text to include before the prompt. Can be very long (up to ~1M tokens)."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_ORACLE_MODEL})",
                        "default": DEFAULT_ORACLE_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="list_models",
            description="List available Gemini models for image generation and oracle reasoning",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Handle tool calls."""

    if name == "list_models":
        models_info = f"""Available Gemini models:

IMAGE GENERATION:
- {DEFAULT_IMAGE_MODEL} (default for image gen)
  Latest image generation model

- gemini-2.5-flash-image
  Fast image generation

ORACLE REASONING (text responses, up to 1M context):
- {DEFAULT_ORACLE_MODEL} (default for oracle)
  Gemini 3.1 Pro Preview — 1M token context, multimodal reasoning

- gemini-3-flash-preview
  Fast oracle with 1M token context

- gemini-2.5-pro
  Stable reasoning model

Set GEMINI_API_KEY environment variable to use these models."""
        return [TextContent(type="text", text=models_info)]

    elif name == "generate_image":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_IMAGE_MODEL)

        try:
            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                model=model,
            )

            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Generated image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating image: {e}")]

    elif name == "revise_image":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        input_image_path = arguments.get("input_image_path")
        input_image_base64 = arguments.get("input_image_base64")
        input_image_mime_type = arguments.get("input_image_mime_type", "image/png")
        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_IMAGE_MODEL)

        if not input_image_path and not input_image_base64:
            return [TextContent(type="text", text="Error: Either 'input_image_path' or 'input_image_base64' is required")]

        try:
            input_bytes = None
            if input_image_base64:
                input_bytes = base64.b64decode(input_image_base64)

            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                input_image_path=input_image_path,
                input_image_bytes=input_bytes,
                input_image_mime_type=input_image_mime_type,
                model=model,
            )

            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Revised image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error revising image: {e}")]

    elif name == "style_generate":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        style_image_path = arguments.get("style_image_path")
        style_image_base64 = arguments.get("style_image_base64")
        style_image_mime_type = arguments.get("style_image_mime_type", "image/png")
        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_IMAGE_MODEL)

        if not style_image_path and not style_image_base64:
            return [TextContent(type="text", text="Error: Either 'style_image_path' or 'style_image_base64' is required for style reference")]

        try:
            style_bytes = None
            if style_image_base64:
                style_bytes = base64.b64decode(style_image_base64)

            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                style_ref_image_path=style_image_path,
                style_ref_image_bytes=style_bytes,
                style_ref_mime_type=style_image_mime_type,
                model=model,
            )

            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Style-generated image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating styled image: {e}")]

    elif name == "gemini_oracle":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        image_path = arguments.get("image_path")
        image_base64 = arguments.get("image_base64")
        image_mime_type = arguments.get("image_mime_type", "image/png")
        context = arguments.get("context")
        model = arguments.get("model", DEFAULT_ORACLE_MODEL)

        try:
            image_bytes = None
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

            response_text = oracle_call(
                prompt=prompt,
                image_path=image_path,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                context=context,
                model=model,
            )
            return [TextContent(type="text", text=response_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error calling Gemini oracle: {e}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
