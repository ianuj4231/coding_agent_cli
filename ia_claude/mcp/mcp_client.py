from langchain_mcp_adapters.client import MultiServerMCPClient
from ia_claude.mcp.mcp_config import load_mcp_configs
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)


def _remove_schema_dialect(value):
    """Remove JSON Schema dialect metadata unsupported by Gemini tools."""

    if isinstance(value, dict):
        return {
            key: _remove_schema_dialect(item)
            for key, item in value.items()
            if key != "$schema"
        }
    if isinstance(value, list):
        return [_remove_schema_dialect(item) for item in value]
    return value


async def get_mcp_tools() -> list:
    """Connect to all configured MCP servers and return their tools."""
    configs = load_mcp_configs()
    logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()
    for tool in tools:
        if isinstance(tool.args_schema, dict):
            tool.args_schema = _remove_schema_dialect(tool.args_schema)
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    return tools
