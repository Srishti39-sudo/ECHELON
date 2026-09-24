"""Make map.html actually standalone.

Folium writes a page that pulls eleven files from four CDNs: Leaflet, the heat
plugin, and then jQuery, Bootstrap, Font Awesome, Awesome Markers and
Glyphicons, none of which this map uses. On a machine with no network that page
renders as a blank white rectangle.

That is not acceptable for this project. A survey map is looked at on a vessel,
in a shed, on a conference floor behind a firewall. So the page is rewritten
before it is saved:

    KEPT AND INLINED   leaflet.css, leaflet.js, leaflet.heat
    REMOVED            jQuery, Bootstrap, Font Awesome, Awesome Markers,
                       Glyphicons, and folium's rotate stylesheet

The removed ones are not a judgement call. Nothing this map draws uses a
Bootstrap class, a jQuery selector or an icon font: the markers are Leaflet
circles and one styled div, and the dashboard is hand-written CSS. Dropping
them removes about 300 KB of unused code and four more chances to fail.

The kept files are vendored into vendor/ the first time a map is built with a
network available, and read from there afterwards. Commit that directory and
the build works offline from a fresh clone.

Versions are read out of the rendered HTML rather than pinned here, so a folium
upgrade that moves to a new Leaflet cannot leave this file silently inlining
the old one.

LICENCES. Leaflet is BSD-2-Clause, (c) Vladimir Agafonkin and CloudMade.
Leaflet.heat is BSD-2-Clause, (c) Vladimir Agafonkin. Both permit
redistribution in source form with their notices intact, which is what
vendoring a minified file with its header comment does.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("deepecho.hazard")

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

# Substrings that identify an asset this map genuinely needs. Matched against
# the URL, so a version bump still matches.
KEEP = ("leaflet.css", "leaflet.js", "leaflet_heat", "leaflet.heat")

# Everything else folium links is dropped. Named rather than implied, so it is
# obvious what was removed and why nothing broke.
DROP_REASON = {
    "bootstrap": "no Bootstrap class is used on this page",
    "jquery": "no jQuery selector is used on this page",
    "fontawesome": "no icon font is used; markers are Leaflet circles",
    "awesome-markers": "no awesome-marker is used",
    "glyphicons": "no glyphicon is used",
    "awesome.rotate": "no rotated marker is used",
}

_LINK = re.compile(
    r'<link[^>]*?rel="stylesheet"[^>]*?href="(https?://[^"]+)"[^>]*?/?>', re.I)
_SCRIPT = re.compile(
    r'<script[^>]*?src="(https?://[^"]+)"[^>]*?>\s*</script>', re.I)

# Hosts folium is known to emit. An asset from anywhere else is dropped rather
# than downloaded, so a compromised or unexpected template cannot make this
# fetch from an arbitrary origin.
ALLOWED_HOSTS = ("cdn.jsdelivr.net", "cdnjs.cloudflare.com", "code.jquery.com",
                 "netdna.bootstrapcdn.com", "unpkg.com")

NETWORK_TIMEOUT = 20


class AssetError(RuntimeError):
    """A required asset is neither vendored nor reachable."""


def _cache_name(url: str) -> str:
    """A flat filename for a URL, keeping the version so two can coexist."""
    tail = url.split("?")[0].rstrip("/").split("/")
    name = tail[-1]
    version = next((part for part in reversed(tail[:-1]) if re.search(r"\d", part)), "")
    version = re.sub(r"[^A-Za-z0-9._@-]", "-", version)
    return f"{version}-{name}" if version and version not in name else name


def _wanted(url: str) -> bool:
    lowered = url.lower()
    if not lowered.startswith("https://"):
        return False
    host = lowered.split("://", 1)[1].split("/", 1)[0]
    if host not in ALLOWED_HOSTS:
        return False
    return any(token in lowered for token in KEEP)


def _dropped_reason(url: str) -> str:
    lowered = url.lower()
    for token, reason in DROP_REASON.items():
        if token in lowered:
            return reason
    return "not used by this map"


def fetch(url: str, vendor_dir: Path | None = None) -> str:
    """The asset's text, from the vendor cache or from the network once."""
    vendor_dir = VENDOR_DIR if vendor_dir is None else Path(vendor_dir)
    cached = vendor_dir / _cache_name(url)
    if cached.is_file():
        return cached.read_text(encoding="utf-8", errors="replace")

    # Only https, and only a host folium itself emitted. This function is fed
    # URLs parsed out of folium's own generated page, never out of user input,
    # but downgrading to http on the strength of a regex match is not something
    # to leave available.
    if not url.lower().startswith("https://"):
        raise AssetError(f"refusing to fetch a non-https asset: {url}")

    log.info("vendoring %s", url)
    try:
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise AssetError(
            f"{url} is not in {vendor_dir} and could not be downloaded ({exc}). "
            f"Build one map on a machine with a network to populate vendor/, "
            f"commit it, and every build after that works offline.") from exc

    vendor_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def inline(html: str, vendor_dir: Path | None = None) -> tuple[str, dict]:
    """Rewrite a folium page to carry its own Leaflet. Returns (html, report)."""
    inlined: list[str] = []
    dropped: list[str] = []

    def replace_link(match: re.Match) -> str:
        url = match.group(1)
        if not _wanted(url):
            dropped.append(url)
            return f"<!-- removed {url}: {_dropped_reason(url)} -->"
        inlined.append(url)
        # Leaflet's stylesheet refers to marker-icon.png and layers.png. This
        # map draws no default marker and shows the layer control expanded, so
        # neither image is ever requested.
        return f"<style>/* {url} */\n{fetch(url, vendor_dir)}\n</style>"

    def replace_script(match: re.Match) -> str:
        url = match.group(1)
        if not _wanted(url):
            dropped.append(url)
            return f"<!-- removed {url}: {_dropped_reason(url)} -->"
        inlined.append(url)
        return f"<script>/* {url} */\n{fetch(url, vendor_dir)}\n</script>"

    html = _LINK.sub(replace_link, html)
    html = _SCRIPT.sub(replace_script, html)

    # Only resources the browser actually FETCHES count. A hyperlink in the
    # Leaflet attribution is an <a href> the page never loads, and counting it
    # as an external dependency would report a self-contained file as not one.
    remaining = (re.findall(r'<link[^>]+href="(https?://[^"]+)"', html)
                 + re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
                 + re.findall(r'<img[^>]+src="(https?://[^"]+)"', html)
                 + re.findall(r'url\((?:\'|")?(https?://[^)\'"]+)', html))
    report = {
        "inlined": inlined,
        "dropped": dropped,
        "remaining_external": sorted(set(remaining)),
    }
    log.info("assets: %d inlined, %d unused links removed, %d external reference(s) left",
             len(inlined), len(dropped), len(report["remaining_external"]))
    return html, report
