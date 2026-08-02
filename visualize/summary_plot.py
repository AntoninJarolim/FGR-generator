"""Bokeh scatter figures for the span-extraction performance page.

One figure per subset (plus the macro average) that has plottable points. A
point is one run with BOTH a plausibility (x) and an answer-bearing (y) value;
top-right is best. The two questions the figures answer:

    A) constrained vs unconstrained — marker shape encodes the prompt format /
       constraint (open circle = xml, filled circle = xml-constrained,
       cross = json), and a line of the model's colour joins that model's
       variants so the shift is visible.
    B) which models are better/worse — colour encodes the model, so the same
       model keeps one colour across every figure and legend.

Rendered server-side to a self-contained HTML document (BokehJS inlined via
INLINE, so it works offline) and embedded by summary.html in an <iframe>.
"""
from collections import defaultdict

from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import Div, HoverTool
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure
from bokeh.resources import INLINE

# short_name prefix -> (variant label, marker, filled). json-constrained and
# the baselines are intentionally absent, so they are skipped in the figures.
VARIANTS = {
    "xml-unc ": ("xml", "circle", False),
    "xml-con ": ("xml-constrained", "circle", True),
    "json-unc ": ("json", "cross", None),
}


def _split(system):
    """('xml', 'circle', False, 'gemma-4-12B-it') or None for non-plottable rows."""
    for prefix, (variant, marker, filled) in VARIANTS.items():
        if system.startswith(prefix):
            return variant, marker, filled, system[len(prefix):]
    return None


def _palette(n):
    if n <= 10:
        return Category10[10]
    if n <= 20:
        return Category20[20]
    base = Category20[20]
    return [base[i % 20] for i in range(n)]


def _color_map(tables):
    models = set()
    for table in tables:
        for row in table["rows"]:
            parsed = _split(row["system"])
            if parsed:
                models.add(parsed[3])
    models = sorted(models)
    pal = _palette(len(models))
    return {m: pal[i % len(pal)] for i, m in enumerate(models)}


def _points(table):
    """Rows of `table` that have both metrics, parsed and scaled to percent."""
    out = []
    for row in table["rows"]:
        parsed = _split(row["system"])
        if not parsed:
            continue
        variant, marker, filled, model = parsed
        x, y = row["cells"].get("plaus"), row["cells"].get("gold-ans")
        if x is None or y is None:
            continue
        out.append({"model": model, "variant": variant, "marker": marker,
                    "filled": filled, "x": x * 100, "y": y * 100})
    return out


def _figure(table, colors):
    points = _points(table)
    if not points:
        return None
    p = figure(title=table["title"], height=360, sizing_mode="stretch_width",
               x_axis_label="plausibility — NDCG@10 (%)",
               y_axis_label="answer-bearing — gold-ans (%)",
               tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")
    p.title.text_font_size = "13px"

    # One line per model joins its variants (sorted by x); legend swatch = model.
    by_model = defaultdict(list)
    for pt in points:
        by_model[pt["model"]].append(pt)
    for model, pts in sorted(by_model.items()):
        pts = sorted(pts, key=lambda d: d["x"])
        p.line([d["x"] for d in pts], [d["y"] for d in pts],
               color=colors[model], line_width=2, alpha=0.55,
               legend_label=model)

    # One scatter per marker style, coloured per point by model.
    scatter_renderers = []
    by_style = defaultdict(list)
    for pt in points:
        by_style[(pt["marker"], pt["filled"])].append(pt)
    for (marker, filled), grp in by_style.items():
        data = {"x": [d["x"] for d in grp], "y": [d["y"] for d in grp],
                "color": [colors[d["model"]] for d in grp],
                "model": [d["model"] for d in grp],
                "variant": [d["variant"] for d in grp]}
        kw = dict(marker=marker, size=13, line_color="color", line_width=2)
        if marker != "cross":
            kw["fill_color"] = "color"
            kw["fill_alpha"] = 0.9 if filled else 0.0
        scatter_renderers.append(p.scatter("x", "y", source=data, **kw))

    p.add_tools(HoverTool(renderers=scatter_renderers, tooltips=[
        ("model", "@model"), ("variant", "@variant"),
        ("plausibility", "@x{0.0}%"), ("gold-ans", "@y{0.0}%")]))

    p.legend.title = "model"
    p.legend.label_text_font_size = "10px"
    p.legend.click_policy = "hide"
    p.add_layout(p.legend[0], "right")
    return p


CAPTION = (
    "<b>Reading the figures.</b> Each point is one run with both metrics; "
    "<b>top-right is best</b> (high plausibility, high answer-bearing). "
    "<b>Colour = model</b> (consistent across figures, see legend). "
    "<b>Marker = prompt format:</b> ○ open circle = xml, ● filled circle = "
    "xml-constrained, ✕ cross = json. A line of the model's colour joins its "
    "variants, so a rising line means the constrained run is better. "
    "Only runs that have <i>both</i> a plausibility and a gold-ans value can "
    "be plotted — subsets without gold answers (qmsum, summ_screen_fd) have "
    "no figure."
)


def render_html(summary):
    colors = _color_map(summary["tables"])
    figs = [f for f in (_figure(t, colors) for t in summary["tables"]) if f is not None]
    caption = Div(text=CAPTION, sizing_mode="stretch_width",
                  styles={"font": "13px -apple-system, BlinkMacSystemFont, 'Segoe UI', "
                          "Roboto, sans-serif", "color": "#374151", "line-height": "1.5",
                          "padding": "0 2px 4px"})
    if not figs:
        body = column(Div(text="<i>No runs yet have both a plausibility and a "
                          "gold-ans value — nothing to plot.</i>",
                          styles={"font": "14px sans-serif", "color": "#6b7280"}),
                      sizing_mode="stretch_width")
    else:
        body = column(caption, *figs, sizing_mode="stretch_width", spacing=18)
    return file_html(body, INLINE, title="span-extraction figures")
