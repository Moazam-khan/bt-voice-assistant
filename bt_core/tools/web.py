"""Web-related tools for BT: opening URLs and running searches.

webbrowser.open() spawns the browser via a subprocess launch, which can
block briefly on process creation — wrapped in asyncio.to_thread per this
project's async I/O rule.
"""

from __future__ import annotations

import asyncio
import webbrowser
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from bt_core.tools.base import PermissionTier, Tool, ToolError


class OpenWebsiteArgs(BaseModel):
    """Arguments for open_website."""

    url: str = Field(description="The URL to open, e.g. https://github.com")


class OpenWebsiteTool(Tool):
    """Opens a URL in the user's default web browser."""

    name = "open_website"
    description = "Open a website in the default browser"
    permission_tier = PermissionTier.SAFE

    def _args_model(self) -> type[BaseModel]:
        return OpenWebsiteArgs

    async def _run(self, args: OpenWebsiteArgs) -> str:
        url = args.url if "://" in args.url else f"https://{args.url}"
        opened = await asyncio.to_thread(webbrowser.open, url)
        if not opened:
            raise ToolError(f"No browser available to open {url}")
        return f"Opened {args.url}"


class GoogleSearchArgs(BaseModel):
    """Arguments for google_search."""

    query: str = Field(description="The search query")


class GoogleSearchTool(Tool):
    """Opens a Google search for the given query in the default browser."""

    name = "google_search"
    description = "Search Google for a query and open the results in the browser"
    permission_tier = PermissionTier.SAFE

    def _args_model(self) -> type[BaseModel]:
        return GoogleSearchArgs

    async def _run(self, args: GoogleSearchArgs) -> str:
        url = f"https://www.google.com/search?q={quote_plus(args.query)}"
        opened = await asyncio.to_thread(webbrowser.open, url)
        if not opened:
            raise ToolError("No browser available to run the search")
        return f"Searched Google for {args.query}"
