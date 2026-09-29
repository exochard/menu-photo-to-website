#!/usr/bin/env python3
"""Static-site builder — one YAML file in, a finished multi-page site out.

The whole point: filling a client job means editing `site.yaml` and running this.
No template surgery, no per-client HTML forks. Build once, sell many.

    python3 build.py                 # site.yaml -> dist/
    python3 build.py --config c.yaml --out build/
    python3 build.py --serve         # build, then serve dist/ on :8000

Requires PyYAML (the only dependency).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    sys.exit("PyYAML missing. Install it: pip install PyYAML")

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- helpers

def esc(value: Any) -> str:
    """Escape untrusted config text before it reaches the page."""
    return html.escape(str(value if value is not None else ""), quote=True)


def slugify(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return out or "section"


def md_inline(text: str) -> str:
    """Escape, then re-enable a deliberately tiny inline subset: **bold**, *italic*, [a](b)."""
    out = esc(text)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+|/[^\s)]*|#[^\s)]*|mailto:[^\s)]+|tel:[^\s)]+)\)",
                 r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    return out


def paragraphs(text: str) -> str:
    blocks = [b.strip() for b in str(text or "").split("\n\n") if b.strip()]
    return "\n".join(f"<p>{md_inline(b)}</p>" for b in blocks)


def page_url(slug: str) -> str:
    return "index.html" if slug == "index" else f"{slug}.html"


def require(cond: bool, message: str) -> None:
    if not cond:
        sys.exit(f"site.yaml: {message}")


# --------------------------------------------------------------------------- sections

def s_hero(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    kicker = f'<p class="kicker">{esc(sec["kicker"])}</p>' if sec.get("kicker") else ""
    sub = f'<p class="hero__sub">{md_inline(sec["subtitle"])}</p>' if sec.get("subtitle") else ""
    media = ""
    if sec.get("image"):
        media = (f'<div class="hero__media"><img src="{esc(sec["image"])}" '
                 f'alt="{esc(sec.get("image_alt", ""))}" loading="eager" width="1200" height="800"></div>')
    return f"""<section class="hero{' hero--split' if media else ''}">
  <div class="wrap hero__in">
    <div class="hero__text">
      {kicker}
      <h1>{md_inline(sec.get('title', ''))}</h1>
      {sub}
      {buttons(sec.get('buttons', []))}
    </div>
    {media}
  </div>
</section>"""


def s_features(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    cards = "".join(
        f"""<li class="card">
      {f'<span class="card__icon" aria-hidden="true">{esc(i["icon"])}</span>' if i.get('icon') else ''}
      <h3>{esc(i.get('title', ''))}</h3>
      <p>{md_inline(i.get('text', ''))}</p>
    </li>""" for i in sec.get("items", []))
    return block(sec, f'<ul class="grid grid--3 plain">{cards}</ul>')


def s_services(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    cards = ""
    for i in sec.get("items", []):
        price = f'<p class="price">{esc(i["price"])}</p>' if i.get("price") else ""
        bullets = "".join(f"<li>{md_inline(b)}</li>" for b in i.get("bullets", []))
        cards += f"""<li class="card card--service">
      <h3>{esc(i.get('title', ''))}</h3>
      {price}
      <p>{md_inline(i.get('text', ''))}</p>
      {f'<ul class="ticks">{bullets}</ul>' if bullets else ''}
    </li>"""
    return block(sec, f'<ul class="grid grid--3 plain">{cards}</ul>')


def s_about(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    media = (f'<div class="about__media"><img src="{esc(sec["image"])}" '
             f'alt="{esc(sec.get("image_alt", ""))}" loading="lazy" width="900" height="700"></div>'
             ) if sec.get("image") else ""
    return f"""<section class="section about" id="{esc(sec.get('id') or slugify(sec.get('title', 'about')))}">
  <div class="wrap about__in{' about__in--split' if media else ''}">
    <div class="about__text">
      <h2>{esc(sec.get('title', ''))}</h2>
      {paragraphs(sec.get('body', ''))}
      {buttons(sec.get('buttons', []))}
    </div>
    {media}
  </div>
</section>"""


def s_gallery(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    items = "".join(
        f'<li><img src="{esc(i.get("src", ""))}" alt="{esc(i.get("alt", ""))}" '
        f'loading="lazy" width="800" height="600"></li>' for i in sec.get("items", []))
    return block(sec, f'<ul class="gallery plain">{items}</ul>')


def s_testimonials(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    items = "".join(
        f"""<li class="quote">
      <blockquote>{md_inline(i.get('text', ''))}</blockquote>
      <p class="quote__by">{esc(i.get('name', ''))}{f" — {esc(i['role'])}" if i.get('role') else ''}</p>
    </li>""" for i in sec.get("items", []))
    return block(sec, f'<ul class="grid grid--3 plain">{items}</ul>')


def s_hours(sec: Dict[str, Any], site: Dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><th scope="row">{esc(r.get("day", ""))}</th><td>{esc(r.get("time", ""))}</td></tr>'
        for r in sec.get("items", []))
    contact = site.get("contact", {}) or {}
    aside = ""
    if contact.get("address"):
        aside = f"""<div class="hours__where">
      <h3>{esc(sec.get('where_title', 'Where'))}</h3>
      <p>{md_inline(contact['address'])}</p>
      {f'<p><a class="btn btn--ghost" href="{esc(contact["maps_url"])}">{esc(sec.get("maps_label", "Open in Maps"))}</a></p>' if contact.get('maps_url') else ''}
    </div>"""
    return block(sec, f'<div class="hours"><table class="hours__table">{rows}</table>{aside}</div>')


def s_menu(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    groups = ""
    for g in sec.get("groups", []):
        rows = "".join(
            f'<li class="menu__item"><span class="menu__name">{esc(i.get("name", ""))}</span>'
            f'<span class="menu__price">{esc(i.get("price", ""))}</span></li>'
            for i in g.get("items", []))
        heading = f"<h3>{esc(g['name'])}</h3>" if g.get("name") else ""
        groups += f'<div class="menu__group">{heading}<ul class="plain">{rows}</ul></div>'
    return block(sec, f'<div class="menu">{groups}</div>')


def s_faq(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    items = "".join(
        f"""<details class="faq__item"><summary>{esc(i.get('q', ''))}</summary>
      <div class="faq__a">{paragraphs(i.get('a', ''))}</div></details>""" for i in sec.get("items", []))
    return block(sec, f'<div class="faq">{items}</div>')


def s_contact(sec: Dict[str, Any], site: Dict[str, Any]) -> str:
    c = site.get("contact", {}) or {}
    rows = []
    if c.get("phone"):
        rows.append(f'<a class="btn" href="tel:{esc(re.sub(r"[^+0-9]", "", c["phone"]))}">{esc(c["phone"])}</a>')
    if c.get("whatsapp"):
        rows.append(f'<a class="btn btn--ghost" href="https://wa.me/{esc(re.sub(r"[^0-9]", "", c["whatsapp"]))}">WhatsApp</a>')
    if c.get("email"):
        rows.append(f'<a class="btn btn--ghost" href="mailto:{esc(c["email"])}">{esc(c["email"])}</a>')
    body = paragraphs(sec.get("body", ""))
    return block(sec, f'{body}<div class="btnrow">{"".join(rows)}</div>')


def s_cta(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    return f"""<section class="cta">
  <div class="wrap cta__in">
    <div>
      <h2>{md_inline(sec.get('title', ''))}</h2>
      {f'<p>{md_inline(sec["text"])}</p>' if sec.get('text') else ''}
    </div>
    {buttons(sec.get('buttons', []))}
  </div>
</section>"""


def s_richtext(sec: Dict[str, Any], _site: Dict[str, Any]) -> str:
    return block(sec, paragraphs(sec.get("body", "")))


def block(sec: Dict[str, Any], inner: str) -> str:
    title = f"<h2>{md_inline(sec['title'])}</h2>" if sec.get("title") else ""
    intro = f'<p class="section__intro">{md_inline(sec["intro"])}</p>' if sec.get("intro") else ""
    sid = esc(sec.get("id") or slugify(sec.get("title", "section")))
    return f"""<section class="section" id="{sid}">
  <div class="wrap">
    {title}
    {intro}
    {inner}
  </div>
</section>"""


def buttons(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    out = "".join(
        f'<a class="btn{"" if i.get("primary", True) else " btn--ghost"}" '
        f'href="{esc(i.get("href", "#"))}">{esc(i.get("label", ""))}</a>' for i in items)
    return f'<div class="btnrow">{out}</div>'


SECTIONS = {
    "hero": s_hero, "features": s_features, "services": s_services, "about": s_about,
    "gallery": s_gallery, "testimonials": s_testimonials, "hours": s_hours,
    "menu": s_menu, "faq": s_faq, "contact": s_contact, "cta": s_cta, "richtext": s_richtext,
}


# --------------------------------------------------------------------------- page shell

def render_nav(cfg: Dict[str, Any], current: str) -> str:
    links = ""
    for item in cfg.get("nav", []):
        href = item.get("href") or page_url(item.get("page", "index"))
        active = ' aria-current="page"' if item.get("page") == current else ""
        links += f'<a href="{esc(href)}"{active}>{esc(item.get("label", ""))}</a>'
    return links


def render_page(cfg: Dict[str, Any], page: Dict[str, Any]) -> str:
    site = cfg["site"]
    slug = page.get("slug", "index")
    body = "".join(
        SECTIONS[sec["type"]](sec, site)
        for sec in page.get("sections", [])
        if sec.get("type") in SECTIONS
    )
    unknown = [s.get("type") for s in page.get("sections", []) if s.get("type") not in SECTIONS]
    for u in unknown:
        print(f"  ! unknown section type {u!r} on page {slug!r} — skipped", file=sys.stderr)

    contact = site.get("contact", {}) or {}
    foot_bits = [esc(contact.get(k)) for k in ("address", "phone", "email") if contact.get(k)]
    social = "".join(
        f'<a href="{esc(url)}" rel="noopener">{esc(name.title())}</a>'
        for name, url in (site.get("social", {}) or {}).items() if url)

    title = f"{page.get('title', site['name'])} — {site['name']}" if slug != "index" else \
            f"{site['name']}{' — ' + site['tagline'] if site.get('tagline') else ''}"
    desc = page.get("description") or site.get("description", "")

    return f"""<!doctype html>
<html lang="{esc(site.get('lang', 'en'))}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:type" content="website">
{f'<meta property="og:image" content="{esc(site["og_image"])}">' if site.get('og_image') else ''}
{f'<link rel="canonical" href="{esc(site["base_url"].rstrip("/"))}/{page_url(slug)}">' if site.get('base_url') else ''}
{font_links(site.get('theme', {}))}<link rel="stylesheet" href="assets/base.css">
{f'<link rel="icon" href="{esc(site["favicon"])}">' if site.get('favicon') else ''}
<style>:root{{{css_vars(site.get('theme', {}))}}}</style>
{'<meta name="robots" content="noindex, nofollow">' if site.get('noindex') else json_ld(site)}
</head>
<body>
<a class="skip" href="#main">{esc(site.get('skip_label', 'Skip to content'))}</a>
{f'<p class="preview-notice" role="note">{esc(site["preview_notice"])}</p>' if site.get('preview_notice') else ''}
<header class="bar">
  <div class="wrap bar__in">
    <a class="brand" href="index.html">{esc(site['name'])}</a>
    <nav class="nav">{render_nav(cfg, slug)}</nav>
    <button class="themetoggle" type="button" data-theme-toggle
            aria-label="{esc(site.get('theme_toggle_label', 'Switch light / dark'))}">◐</button>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="foot">
  <div class="wrap foot__in">
    <p>{' · '.join(foot_bits)}</p>
    {f'<p class="foot__social">{social}</p>' if social else ''}
    <p class="foot__legal">© {esc(site.get('year', '2026'))} {esc(site['name'])}
    {f" · {esc(site['legal'])}" if site.get('legal') else ''}</p>
  </div>
</footer>
<script>
(function(){{
  var K='theme',r=document.documentElement,s=null;
  try{{s=localStorage.getItem(K)}}catch(e){{}}
  if(s)r.setAttribute('data-theme',s);
  var b=document.querySelector('[data-theme-toggle]');
  if(b)b.addEventListener('click',function(){{
    var dark=r.getAttribute('data-theme')==='dark'||
      (!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
    var next=dark?'light':'dark';
    r.setAttribute('data-theme',next);
    try{{localStorage.setItem(K,next)}}catch(e){{}}
  }});
}})();
</script>
</body>
</html>
"""


CSS_VALUE = re.compile(r"^[A-Za-z0-9 ,.#%()'\"_-]+$")


def css_vars(theme: Dict[str, Any]) -> str:
    """Build the per-client `:root{}` override.

    Values go inside a <style> element, where HTML entities are NOT decoded — so
    `esc()` would corrupt them (a font-family arrives as &#x27;Inter&#x27;). They
    are whitelisted against CSS syntax instead of escaped.
    """
    allowed = {
        "accent": "--accent", "accent_dark": "--accent-dark", "radius": "--radius",
        "font_body": "--font-body", "font_display": "--font-display", "maxw": "--maxw",
    }
    out = ""
    for key, var in allowed.items():
        value = str(theme.get(key, "") or "").strip()
        if not value:
            continue
        if not CSS_VALUE.match(value):
            print(f"  ! theme.{key} has characters that aren't valid here — ignored: {value!r}",
                  file=sys.stderr)
            continue
        out += f"{var}:{value};"
    return out


def font_links(theme: Dict[str, Any]) -> str:
    """Emit the client's web-font stylesheet. `fonts_url` must be an https URL."""
    url = str(theme.get("fonts_url", "") or "")
    if not url.startswith("https://"):
        if url:
            print(f"  ! theme.fonts_url must start with https:// — ignored: {url!r}", file=sys.stderr)
        return ""
    host = url.split("/")[2]
    return (f'<link rel="preconnect" href="https://{esc(host)}" crossorigin>'
            f'<link rel="stylesheet" href="{esc(url)}">\n')


def json_ld(site: Dict[str, Any]) -> str:
    biz = site.get("business")
    if not biz:
        return ""
    contact = site.get("contact", {}) or {}
    data = {
        "@context": "https://schema.org",
        "@type": biz.get("type", "LocalBusiness"),
        "name": site.get("name"),
        "description": site.get("description", ""),
    }
    if site.get("base_url"):
        data["url"] = site["base_url"]
    if contact.get("phone"):
        data["telephone"] = contact["phone"]
    if contact.get("address"):
        data["address"] = {"@type": "PostalAddress", "streetAddress": contact["address"]}
    if biz.get("price_range"):
        data["priceRange"] = biz["price_range"]
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/ld+json">{payload}</script>'


# --------------------------------------------------------------------------- build

def load(config_path: Path) -> Dict[str, Any]:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    require(isinstance(cfg.get("site"), dict), "missing top-level `site:` block")
    require(bool(cfg["site"].get("name")), "`site.name` is required")
    require(bool(cfg.get("pages")), "at least one entry under `pages:` is required")
    slugs = [p.get("slug") for p in cfg["pages"]]
    require("index" in slugs, "one page must have `slug: index`")
    require(len(slugs) == len(set(slugs)), f"duplicate page slugs: {slugs}")
    return cfg


def build(config_path: Path, out_dir: Path) -> List[Path]:
    cfg = load(config_path)
    src = config_path.parent
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True)

    shutil.copy2(HERE / "theme" / "base.css", out_dir / "assets" / "base.css")
    for extra in ("images", "assets"):
        if (src / extra).is_dir():
            shutil.copytree(src / extra, out_dir / extra, dirs_exist_ok=True)

    written: List[Path] = []
    for page in cfg["pages"]:
        target = out_dir / page_url(page.get("slug", "index"))
        target.write_text(render_page(cfg, page), encoding="utf-8")
        written.append(target)
        print(f"  → {target.relative_to(out_dir.parent)}")

    if cfg["site"].get("noindex"):
        # A preview: tell crawlers to stay out and publish no sitemap.
        (out_dir / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    elif cfg["site"].get("base_url"):
        base = cfg["site"]["base_url"].rstrip("/")
        urls = "".join(f"<url><loc>{base}/{page_url(p.get('slug','index'))}</loc></url>"
                       for p in cfg["pages"])
        (out_dir / "sitemap.xml").write_text(
            f'<?xml version="1.0" encoding="UTF-8"?><urlset '
            f'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>', encoding="utf-8")
        (out_dir / "robots.txt").write_text(
            f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n", encoding="utf-8")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a static site from site.yaml")
    ap.add_argument("--config", default=str(HERE / "site.yaml"))
    ap.add_argument("--out", default=str(HERE / "dist"))
    ap.add_argument("--serve", action="store_true", help="serve the output on :8000 after building")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    pages = build(Path(args.config).resolve(), out)
    print(f"built {len(pages)} page(s) → {out}")

    if args.serve:
        import functools
        import http.server
        import socketserver
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        print("serving http://localhost:8000  (ctrl-c to stop)")
        with socketserver.TCPServer(("", 8000), handler) as httpd:
            httpd.serve_forever()


if __name__ == "__main__":
    main()
