# gemini-tools

MCP server for Gemini AI — image generation, style transfer, and multimodal reasoning.

## Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text prompts |
| `revise_image` | Edit/revise existing images with text instructions |
| `style_generate` | Generate images in the style of a reference image |
| `gemini_oracle` | Multimodal reasoning with 1M token context and 65K output tokens |
| `list_models` | List available Gemini models |

## Setup

Requires Python 3.11+ and a `GEMINI_API_KEY` environment variable.

```bash
# Install with uv
uv pip install -e .

# Or run directly
uv run gemini-tools-mcp
```

## MCP Configuration

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "gemini-tools": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/gemini-tools", "gemini-tools-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Models

**Image Generation:**
- `gemini-3-pro-image-preview` (default) — latest multimodal image gen
- `gemini-2.5-flash-image` — faster generation

**Oracle Reasoning:**
- `gemini-2.5-pro` (default) — 1M context, best reasoning
- `gemini-2.5-flash` — fast, large context
- `gemini-3.1-pro-preview` — cutting-edge preview
- `gemini-3-flash-preview` — fast preview

## Usage Examples

**Generate an image:**
```
generate_image(prompt="A cyberpunk cityscape at sunset")
```

**Revise an image:**
```
revise_image(prompt="Make the sky more dramatic", input_image_path="/path/to/image.png")
```

**Style transfer:**
```
style_generate(prompt="A mountain landscape", style_image_path="/path/to/reference.png")
```

**Oracle reasoning (multimodal):**
```
gemini_oracle(
    prompt="Analyze this architecture diagram",
    image_path="/path/to/diagram.png",
    context="<large codebase or document text>",
    system_instruction="You are an expert software architect."
)
```

## License

MIT
