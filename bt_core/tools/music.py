"""Music/video search for BT: search YouTube and play the top result.

No YouTube API key is used — this reads YouTube's own search results page
and picks the first real video out of the same JSON data the page itself
uses to render results (embedded as `ytInitialData`). Two honest tradeoffs
that come with that choice, over a real API key:

- No stability guarantee. YouTube can change this structure at any time
  without notice, which would silently break this tool until updated.
- Scraping a results page like this falls outside YouTube's terms of
  service. Accepted here as a deliberate tradeoff in exchange for needing
  no API key, Google Cloud project, or account setup.
"""

from __future__ import annotations

import asyncio
import json
import re
import webbrowser
from urllib.parse import quote_plus

import requests
from pydantic import BaseModel, Field

from bt_core.logging_setup import get_logger
from bt_core.tools.base import PermissionTier, Tool, ToolError

log = get_logger(__name__)

_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"
_REQUEST_TIMEOUT_S = 10
_INITIAL_DATA_PATTERN = re.compile(r"var ytInitialData = (\{.*?\});", re.DOTALL)
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _find_first_video(node: object) -> tuple[str, str] | None:
    """Depth-first search ytInitialData for the first real video result.

    Walking for the "videoRenderer" marker key (rather than a fixed path
    into the structure) is what keeps this tolerant of YouTube's frequent,
    unannounced layout/schema changes elsewhere in the page data.

    Args:
        node: A parsed JSON value (dict, list, or scalar) to search.

    Returns:
        (title, video_id) for the first video result found, or None.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            renderer = current.get("videoRenderer")
            if isinstance(renderer, dict):
                video_id = renderer.get("videoId")
                title_runs = renderer.get("title", {}).get("runs", [])
                title = title_runs[0]["text"] if title_runs else None
                if video_id and title:
                    return title, video_id
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _search_top_video(query: str) -> tuple[str, str]:
    """Search YouTube and return the top result's (title, video_id).

    Runs in a worker thread — uses the blocking ``requests`` library.
    """
    url = _SEARCH_URL.format(query=quote_plus(query))
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()

    match = _INITIAL_DATA_PATTERN.search(response.text)
    if not match:
        raise ToolError(f"I couldn't read YouTube's results for '{query}'.")

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ToolError(f"I couldn't read YouTube's results for '{query}'.") from exc

    result = _find_first_video(data)
    if result is None:
        raise ToolError(f"I couldn't find any videos for '{query}' on YouTube.")
    return result


class PlayMusicArgs(BaseModel):
    """Arguments for play_music."""

    query: str = Field(description="Song name or search query to find and play on YouTube")


class PlayMusicTool(Tool):
    """Searches YouTube for a song/video and plays the top result."""

    name = "play_music"
    description = "Search YouTube for a song or video and play the top result"
    permission_tier = PermissionTier.SAFE

    def _args_model(self) -> type[BaseModel]:
        return PlayMusicArgs

    async def _run(self, args: PlayMusicArgs) -> str:
        try:
            title, video_id = await asyncio.to_thread(_search_top_video, args.query)
        except ToolError:
            raise
        except requests.RequestException as exc:
            log.error("youtube_search_failed", query=args.query, exc_info=True)
            raise ToolError("I couldn't reach YouTube right now.") from exc

        watch_url = _WATCH_URL.format(video_id=video_id)
        opened = await asyncio.to_thread(webbrowser.open, watch_url)
        if not opened:
            raise ToolError("No browser available to play the video")
        return f"Playing '{title}' on YouTube"
