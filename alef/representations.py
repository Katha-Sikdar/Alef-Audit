"""Stage 2: Structural Representation Modeling (Section 6.2, Table 5).

Transforms a canonical page (title, body text, and a small DOM-like element
list) into the five structural representations ALEF evaluates: plain text,
HTML, raw HTTP response, rendered snapshot (a plain-text approximation of
what an agent's screenshot-captioning step would surface), and accessibility
tree (AXTree).

These are deliberately simple, deterministic string transforms -- the point
is not to reimplement a browser rendering engine, but to give every provider
adapter and every trajectory a *representation-labeled* observation string so
that Table 5/Table 8-style breakdowns (compliance rate by representation) are
reproducible without a real browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

from alef.trajectory import PageRepresentation


@dataclass
class PageElement:
    """A single interactive/content DOM node, used to build HTML/AXTree/
    rendered-snapshot views consistently from one source of truth."""

    role: str  # e.g. "heading", "paragraph", "button", "link", "comment", "meta"
    text: str
    attrs: dict[str, str] = field(default_factory=dict)
    hidden: bool = False  # CSS-hidden or metadata-only node (Section 9.2 case study)


@dataclass
class CanonicalPage:
    """The representation-agnostic source of truth for one page in a
    trajectory. `elements` may include an injected payload embedded as a
    hidden or visible element, matching the paper's threat model
    (Section 3.1: hidden HTML attributes, metadata, or visible DOM nodes)."""

    page_id: str
    title: str
    elements: list[PageElement] = field(default_factory=list)

    def with_injection(self, payload: str, hidden: bool = True, role: str = "comment") -> "CanonicalPage":
        """Return a copy of this page with an injected element appended."""
        new_elements = list(self.elements) + [PageElement(role=role, text=payload, hidden=hidden)]
        return CanonicalPage(page_id=self.page_id, title=self.title, elements=new_elements)

    def injected_text(self) -> str | None:
        for el in self.elements:
            if el.role in ("comment", "meta", "system_override") and el.hidden:
                # Heuristic: hidden comment/meta/system_override nodes are treated
                # as the injection carrier, matching the paper's case studies.
                return el.text
        return None


def render(page: CanonicalPage, representation: PageRepresentation) -> str:
    """Render a CanonicalPage into the requested structural representation."""
    if representation == PageRepresentation.PLAIN_TEXT:
        return _render_plain_text(page)
    if representation == PageRepresentation.HTML:
        return _render_html(page)
    if representation == PageRepresentation.RAW_HTTP:
        return _render_raw_http(page)
    if representation == PageRepresentation.RENDERED_SNAPSHOT:
        return _render_snapshot(page)
    if representation == PageRepresentation.ACCESSIBILITY_TREE:
        return _render_axtree(page)
    raise ValueError(f"Unknown representation: {representation!r}")


def _render_plain_text(page: CanonicalPage) -> str:
    lines = [page.title]
    for el in page.elements:
        # Plain text extraction typically still surfaces hidden text nodes,
        # since there's no visual/DOM channel to hide behind -- this matches
        # the paper's framing that plain text has the *least* structural
        # cover for an attacker (Table 5: highest TC, but also comparatively
        # high AC, since there is no representational suppression at all).
        lines.append(el.text)
    return "\n".join(lines)


def _render_html(page: CanonicalPage) -> str:
    body_parts = [f"<h1>{escape(page.title)}</h1>"]
    for el in page.elements:
        style = ' style="display:none"' if el.hidden else ""
        tag = {"heading": "h2", "paragraph": "p", "button": "button", "link": "a",
               "comment": "!--", "meta": "meta", "system_override": "span"}.get(el.role, "div")
        if tag == "!--":
            body_parts.append(f"<!-- {el.text} -->")
        elif tag == "meta":
            body_parts.append(f'<meta name="{escape(el.role)}" content="{escape(el.text)}">')
        else:
            body_parts.append(f'<{tag}{style} data-role="{escape(el.role)}">{escape(el.text)}</{tag}>')
    return f"<html><body>{''.join(body_parts)}</body></html>"


def _render_raw_http(page: CanonicalPage) -> str:
    html_body = _render_html(page)
    headers = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html_body)}\r\n"
        "\r\n"
    )
    return headers + html_body


def _render_snapshot(page: CanonicalPage) -> str:
    # A "rendered snapshot" observation approximates what a vision/screenshot
    # pipeline would surface: hidden (display:none / CSS-hidden) elements are
    # dropped, since they are not visually rendered, but visible DOM nodes
    # (including visible injected instructions) remain.
    lines = [f"[Rendered page: {page.title}]"]
    for el in page.elements:
        if el.hidden:
            continue
        lines.append(f"- {el.text}")
    return "\n".join(lines)


def _render_axtree(page: CanonicalPage) -> str:
    # Accessibility trees surface functional roles (buttons, links, headings)
    # prominently and tend to compress or drop non-interactive hidden text --
    # matching the paper's finding that AXTree suppresses textual compliance
    # more than any other representation (Table 5) without proportionally
    # suppressing action compliance (Section 11.2).
    lines = [f'root [document] "{page.title}"']
    for el in page.elements:
        if el.hidden and el.role not in ("button", "link"):
            # Hidden non-interactive content is compressed to a low-salience
            # generic node -- demoted in prominence/role, but its full text
            # is NOT truncated or dropped, mirroring how AXTrees still expose
            # the complete semantic content to the tool-parameter channel
            # even when it's stripped of prominence in the visible summary
            # (this is the structural basis for Table 5/Section 11.2: AXTree
            # suppresses textual compliance without proportionally
            # suppressing action compliance -- a mock/model that only sees a
            # *truncated* version of the injected text couldn't reproduce
            # that finding, so we deliberately keep the full text here).
            lines.append(f'  generic (low-salience) "{el.text}"')
            continue
        role = {"heading": "heading", "paragraph": "text", "button": "button",
                 "link": "link", "comment": "generic", "meta": "generic",
                 "system_override": "generic"}.get(el.role, "generic")
        lines.append(f'  {role} "{el.text}"')
    return "\n".join(lines)
