# -*- coding: utf-8 -*-
"""An SSE MCP server with a simple add tool function."""

from mcp.server import FastMCP


mcp = FastMCP("Add", port=8001)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print(f"Adding {a} and {b}...\n")
    return a + b

if __name__ == '__main__':
    mcp.run(transport="sse")
