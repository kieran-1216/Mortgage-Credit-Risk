"""Mortgage Delinquency Risk Calculator: a research tool built on the project's two tested models (Logistic Regression
and LightGBM). The logic lives in tool.py; this file is the page.

Run (from the repository root):  streamlit run live-tool/streamlit_app.py
"""

import html
import random

import altair as alt
import pandas as pd
import streamlit as st

from tool import APP_DIR, CATEGORY_FIELDS, GBM, LR, NOTES, NUMBER_FIELDS, OTHER, YES_NO, Tool, pct, readout_cells, stop_label

st.set_page_config(page_title="Mortgage Delinquency Risk Calculator", layout="wide")

ACCENT = "#6CA6E0"
TEXT, MUTED, LINE = "#E3E7EC", "#9AA4B2", "#2C333D"
RAISE, LOWER, NEUTRAL = "#E06C6C", "#5DBB8A", "#8A94A3"
APPLICANT, OPTIMAL, CURVE = "#E3A857", "#D5DAE1", "#B7C3D0"


# ---------------------------------------------------------------- the header drawing (original, drawn in code)

def _house(rnd, x, g, w, svg):
    """A two-story gabled house: siding, a shingled roof, framed windows (some lit), a door with a step, and sometimes a
    chimney or a porch."""
    h, roof = rnd.randint(40, 52), rnd.randint(20, 28)
    top = g - h
    svg.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" class="face"/>')
    for y in range(top + 5, g, 5):
        svg.append(f'<line x1="{x + 1}" y1="{y}" x2="{x + w - 1}" y2="{y}" class="siding"/>')
    if rnd.random() < 0.6:
        cx = x + w * 0.7
        svg.append(f'<rect x="{cx:.1f}" y="{top - roof * 0.9:.1f}" width="7" height="{roof * 0.7:.1f}" class="roof"/>')
    svg.append(f'<polygon points="{x - 5},{top} {x + w / 2:.1f},{top - roof} {x + w + 5},{top}" class="roof"/>')
    for k in range(1, 4):
        f = k / 4
        svg.append(f'<line x1="{x - 5 + f * (w / 2 + 5):.1f}" y1="{top - roof * f:.1f}" x2="{x + w + 5 - f * (w / 2 + 5):.1f}" '
                   f'y2="{top - roof * f:.1f}" class="shingle"/>')
    for wx in (x + 7, x + w - 19):
        for wy in (top + 7, top + 25):
            if wy + 11 > g - 4:
                continue
            lit = rnd.random() < 0.22
            svg.append(f'<rect x="{wx}" y="{wy}" width="12" height="11" class="{"lit" if lit else "win"}"/>')
            svg.append(f'<line x1="{wx + 6}" y1="{wy}" x2="{wx + 6}" y2="{wy + 11}" class="frame"/>')
            svg.append(f'<line x1="{wx}" y1="{wy + 5.5}" x2="{wx + 12}" y2="{wy + 5.5}" class="frame"/>')
    dx = x + w / 2 - 5
    svg.append(f'<rect x="{dx:.1f}" y="{g - 16}" width="10" height="16" class="door"/>')
    svg.append(f'<line x1="{dx - 3:.1f}" y1="{g - 0.5}" x2="{dx + 13:.1f}" y2="{g - 0.5}" class="frame"/>')
    if rnd.random() < 0.45:
        svg.append(f'<polygon points="{dx - 9:.1f},{g - 20} {dx + 19:.1f},{g - 20} {dx + 16:.1f},{g - 24} {dx - 6:.1f},{g - 24}" class="roof"/>')
        for px in (dx - 7, dx + 17):
            svg.append(f'<line x1="{px:.1f}" y1="{g - 20}" x2="{px:.1f}" y2="{g}" class="frame"/>')


def _apartment(rnd, x, g, w, svg):
    """A small apartment building: parapet, a grid of framed windows (some lit), an entrance canopy."""
    floors = rnd.randint(4, 5)
    h = floors * 17 + 8
    top = g - h
    svg.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" class="face"/>')
    svg.append(f'<rect x="{x - 2}" y="{top - 4}" width="{w + 4}" height="4" class="roof"/>')
    cols = max(3, (w - 10) // 14)
    for f in range(floors - 1):
        for c in range(cols):
            wx = x + 6 + c * (w - 12) / cols
            wy = top + 7 + f * 17
            svg.append(f'<rect x="{wx:.1f}" y="{wy}" width="{(w - 12) / cols - 5:.1f}" height="10" '
                       f'class="{"lit" if rnd.random() < 0.3 else "win"}"/>')
    svg.append(f'<rect x="{x + w / 2 - 7:.1f}" y="{g - 15}" width="14" height="15" class="door"/>')
    svg.append(f'<rect x="{x + w / 2 - 11:.1f}" y="{g - 18}" width="22" height="3" class="roof"/>')


def _townhouses(rnd, x, g, svg):
    """Three joined townhouses with cornices and stoops."""
    for k in range(3):
        w, h = 30, rnd.randint(48, 60)
        top = g - h
        svg.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" class="face"/>')
        svg.append(f'<rect x="{x - 1}" y="{top - 3}" width="{w + 2}" height="3" class="roof"/>')
        for wy in (top + 7, top + 24):
            svg.append(f'<rect x="{x + 6}" y="{wy}" width="18" height="11" class="{"lit" if rnd.random() < 0.3 else "win"}"/>')
            svg.append(f'<line x1="{x + 15}" y1="{wy}" x2="{x + 15}" y2="{wy + 11}" class="frame"/>')
        svg.append(f'<rect x="{x + 10}" y="{g - 15}" width="10" height="15" class="door"/>')
        svg.append(f'<polyline points="{x + 7},{g} {x + 7},{g - 3} {x + 23},{g - 3} {x + 23},{g}" class="frame"/>')
        x += w
    return x


def _tree(rnd, x, g, svg):
    r = rnd.randint(11, 15)
    svg.append(f'<line x1="{x + r}" y1="{g}" x2="{x + r}" y2="{g - 18}" class="trunk"/>')
    for dx, dy, rr in ((0, 0, r), (-r * 0.55, r * 0.35, r * 0.7), (r * 0.55, r * 0.4, r * 0.65)):
        svg.append(f'<circle cx="{x + r + dx:.1f}" cy="{g - 18 - r + dy:.1f}" r="{rr:.1f}" class="leaf"/>')
    return x + 2 * r + rnd.randint(8, 14)


def _lamp(x, g, svg):
    svg.append(f'<circle cx="{x + 9}" cy="{g - 50}" r="16" fill="url(#glow)"/>')
    svg.append(f'<line x1="{x}" y1="{g}" x2="{x}" y2="{g - 52}" class="frame"/>')
    svg.append(f'<line x1="{x}" y1="{g - 52}" x2="{x + 9}" y2="{g - 52}" class="frame"/>')
    svg.append(f'<rect x="{x + 6}" y="{g - 52}" width="6" height="4" class="lit"/>')
    return x + 18


def skyline_svg(width=1400, height=130, seed=23):
    """A night-time street of houses, townhouses, a small apartment building, trees and street lamps, drawn in code."""
    rnd = random.Random(seed)
    g = height - 8
    svg = []
    x = 6
    plan = ["house", "tree", "town", "lamp", "house", "apt", "tree", "house", "house", "lamp", "town", "tree", "apt",
            "house", "tree", "house", "lamp", "town", "house", "tree", "apt", "house"]
    k = 0
    while x < width - 60:
        kind = plan[k % len(plan)]
        k += 1
        if kind == "house":
            w = rnd.randint(54, 66)
            _house(rnd, x, g, w, svg)
            x += w + rnd.randint(12, 18)
        elif kind == "apt":
            w = rnd.randint(62, 74)
            _apartment(rnd, x, g, w, svg)
            x += w + rnd.randint(10, 16)
        elif kind == "town":
            x = _townhouses(rnd, x, g, svg) + rnd.randint(10, 16)
        elif kind == "tree":
            x = _tree(rnd, x, g, svg)
        else:
            x = _lamp(x + 6, g, svg) + 6
    style = (".face{fill:#1C222A;stroke:#3D4754;stroke-width:1}.roof{fill:#232A33;stroke:#465260;stroke-width:1}"
             ".siding{stroke:#262D36;stroke-width:0.6}.shingle{stroke:#333C47;stroke-width:0.6}"
             ".win{fill:#262E38;stroke:#465260;stroke-width:0.8}.lit{fill:#E9C46A;fill-opacity:0.26;stroke:#6E6245;stroke-width:0.8}"
             ".frame{stroke:#4C5868;stroke-width:0.9;fill:none}.door{fill:#2A323C;stroke:#4C5868;stroke-width:0.8}"
             ".trunk{stroke:#3B4450;stroke-width:2}.leaf{fill:#1A2A24;stroke:#2F4A3E;stroke-width:0.9}"
             ".ground{stroke:#3A434E;stroke-width:1}")
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMax slice" width="100%" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs><style>{style}</style>'
            f'<radialGradient id="glow"><stop offset="0" stop-color="#E9C46A" stop-opacity="0.22"/>'
            f'<stop offset="1" stop-color="#E9C46A" stop-opacity="0"/></radialGradient></defs>'
            f'<g opacity="0.75">{"".join(svg)}<line x1="0" y1="{g}" x2="{width}" y2="{g}" class="ground"/></g></svg>')


st.markdown(f"""
<style>
.block-container {{padding-top: 1rem; padding-bottom: 3rem; max-width: 1220px;}}
.banner {{margin: 0 0 0.6rem 0; line-height: 0; overflow: hidden; border-bottom: 1px solid {LINE};}}
.pill {{display: inline-block; font-size: 0.8rem; color: {ACCENT}; border: 1px solid #355373; background: #18222E;
       border-radius: 999px; padding: 0.15rem 0.75rem; margin: 0 0 0.5rem 0;}}
.scope {{color: {MUTED}; font-size: 0.92rem; margin-bottom: 0.4rem;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background: #1A1E25; border-color: {LINE} !important; border-radius: 12px;}}
.group {{color: {ACCENT}; font-size: 0.95rem; font-weight: 600; margin: 0.2rem 0 0.5rem 0;}}
.range {{color: #7D8796; font-size: 0.76rem; margin: -0.65rem 0 0.7rem 0.1rem;}}
.calc {{color: {MUTED}; font-size: 0.85rem; line-height: 1.55; border-top: 1px solid {LINE}; margin-top: 0.6rem; padding-top: 0.6rem;}}
.calc b {{color: #C2CAD4; font-weight: 600;}}
.lead {{color: {MUTED}; font-size: 0.92rem; line-height: 1.55; margin-bottom: 0.8rem;}}
table.tbl, table.drv {{border: none !important; border-collapse: collapse;}}
table.tbl th, table.tbl td, table.drv th, table.drv td {{border-left: none !important; border-right: none !important;
              border-top: none !important; background: transparent !important;}}
table.tbl {{width: 100%; font-size: 0.92rem;}}
table.tbl th {{text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.85rem; padding: 0.45rem 0.6rem;
              border-bottom: 1px solid #3A424D !important;}}
table.tbl td {{padding: 0.5rem 0.6rem; border-bottom: 1px solid {LINE} !important; vertical-align: top;}}
table.tbl td.k {{color: {MUTED};}}
table.tbl td.v {{white-space: nowrap;}}
table.tbl td.big {{font-size: 1.6rem; font-weight: 600; color: {TEXT};}}
.approve {{color: {LOWER}; font-weight: 700; letter-spacing: 0.03em;}}
.decline {{color: {RAISE}; font-weight: 700; letter-spacing: 0.03em;}}
table.drv {{width: 100%; font-size: 0.84rem; table-layout: fixed;}}
table.drv th {{text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.8rem; padding: 0.35rem 0.4rem;
              border-bottom: 1px solid #3A424D !important;}}
table.drv td {{padding: 0.3rem 0.4rem; border-bottom: 1px solid #222830 !important; vertical-align: middle;}}
table.drv td.num {{text-align: right; font-variant-numeric: tabular-nums;}}
.track {{position: relative; height: 18px;
        background: repeating-linear-gradient(to right, #262C34 0, #262C34 1px, transparent 1px, transparent 5%);
        border-right: 1px solid #262C34;}}
.track .zero {{position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #5A6575;}}
.track .bar {{position: absolute; top: 4px; height: 10px; border-radius: 2px;}}
.axis {{position: relative; height: 14px; color: #7D8796; font-size: 0.7rem;}}
.axis span {{position: absolute; transform: translateX(-50%);}}
.note {{color: {MUTED}; font-size: 0.85rem; margin-top: 0.6rem;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_tool(root, data_dir):
    return Tool(root=root, data_dir=data_dir)


# This script's folder, holding models/ and data/.
tool = load_tool(str(APP_DIR), None)
if "stop" not in st.session_state:
    st.session_state["stop"] = tool.optimal

# ---------------------------------------------------------------- header

st.markdown(f'<div class="banner">{skyline_svg()}</div>', unsafe_allow_html=True)
st.title("Mortgage Delinquency Risk Calculator")
st.markdown(f'<span class="pill">{NOTES["research"]}</span><div class="scope">{NOTES["scope"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Applicant Information (results change only on Submit)

LABELS = {key: label for key, label, _ in NUMBER_FIELDS}
GROUPS = [("Borrower", ["credit_score", "dti"], ["num_borrowers", "first_time_homebuyer"]),
          ("Loan", ["loan_amount", "loan_term", "interest_rate"], ["loan_purpose", "num_units", "property_type"]),
          ("Property and Program", ["cltv", "mi_pct"], ["occupancy", "channel", "special_program"])]


def current_entry():
    keys = [k for _, nums, cats in GROUPS for k in nums + cats]
    return {k: st.session_state.get(f"in_{k}") for k in keys}


def number_box(key, entry):
    lo, hi = tool.limits[key]
    if key == "interest_rate":
        val = st.number_input(LABELS[key], min_value=float(lo), max_value=float(hi), value=None, step=0.125, format="%.3f",
                              placeholder="Blank", key=f"in_{key}")
    else:
        val = st.number_input(LABELS[key], min_value=int(lo), max_value=int(hi), value=None, step=1000 if key == "loan_amount" else 1,
                              format="%d", placeholder="Blank", key=f"in_{key}")
    st.markdown(f'<div class="range">{html.escape(tool.range_note(key, entry))}</div>', unsafe_allow_html=True)
    return val


def choice_box(col):
    label, options = CATEGORY_FIELDS[col]
    plain = dict(options)
    return st.selectbox(label, [v for v, _ in options], index=None, format_func=plain.get, placeholder="Blank", key=f"in_{col}",
                        help=NOTES["r"] if col == "special_program" else None)


with st.container(border=True):
    st.subheader("Applicant Information")
    live = current_entry()
    entry = {}
    for holder, (group, nums, cats) in zip(st.columns(3, gap="large"), GROUPS):
        with holder:
            st.markdown(f'<div class="group">{group}</div>', unsafe_allow_html=True)
            for key in nums:
                entry[key] = number_box(key, live)
            for col in cats:
                entry[col] = choice_box(col)
    if st.button("Submit", type="primary"):
        res = tool.score(entry)
        st.session_state["result"] = None if res["problems"] else res
        st.session_state["problems"] = res["problems"]
    result = st.session_state.get("result")
    problems = st.session_state.get("problems", [])
    if result is not None:
        c = result["calc"]
        spread = "blank" if c["rate_spread"] is None else f"{c['rate_spread']:+.4f}".rstrip("0").rstrip(".") + " pts"
        ratio = "blank" if c["pct_of_limit"] is None else f"{c['pct_of_limit']:.2f}%"
        yn = {**YES_NO, None: "blank"}
        lines = [f"<b>Calculated From Your Inputs</b>",
                 f"Rate spread: {spread} &nbsp;·&nbsp; Loan size vs. 2023 limit: {ratio} &nbsp;·&nbsp; "
                 f"At the loan limit: {yn[c['at_limit']]} &nbsp;·&nbsp; High-balance loan: {yn[c['high_balance']]}",
                 html.escape(NOTES["spread"])]
        if c["median_missing"]:
            lines.append(f"No 2023 median rate for {c['product']} loans, so the rate spread is blank.")
        if result["blank_note"]:
            lines.append(html.escape(f"Left blank: {', '.join(result['blank_note'])}. {NOTES['blank']}"))
        st.markdown(f'<div class="calc">{"<br>".join(lines)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- results

with st.container(border=True):
    st.subheader(NOTES["title"])
    if problems:
        for p in problems:
            st.error(p)
    elif result is None:
        st.markdown('<div class="lead">Fill in what you know (blank fields are allowed) and press Submit.</div>', unsafe_allow_html=True)
    else:
        cells = {m: (pct(result["risk"][m]), f"{100 * result['rank'][m]:.1f}% of 2023 loans") for m in (GBM, LR)}
        st.markdown(f"""<table class="tbl"><tr><th></th><th>{GBM}</th><th>{LR}</th></tr>
<tr><td class="k">Probability of Serious Delinquency</td><td class="big">{cells[GBM][0]}</td><td class="big">{cells[LR][0]}</td></tr>
<tr><td class="k">Riskier Than</td><td>{cells[GBM][1]}</td><td>{cells[LR][1]}</td></tr></table>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Approval Trade-off Curve

risk_g = None if result is None else result["risk"][GBM]


def reset_cutoff():
    st.session_state["stop"] = tool.optimal


with st.container(border=True):
    st.subheader("Approval Trade-off Curve")
    st.markdown(f'<div class="lead">{html.escape(NOTES["curve"])}</div>', unsafe_allow_html=True)
    chart_col, read_col = st.columns([1, 1], gap="large")
    with read_col:
        stop = st.select_slider("% of applications declined (riskiest first)", options=list(range(len(tool.curve))),
                                format_func=lambda i: stop_label(tool.curve[i]), key="stop")
        st.button("Reset to Optimal Cutoff", on_click=reset_cutoff)
        mine, best = readout_cells(tool.curve[stop], risk_g), readout_cells(tool.curve[tool.optimal], risk_g)

        def decision(ro):
            if ro["decision"] is None:
                return '<span style="color:#7D8796">Submit to see</span>'
            return f'<span class="{ro["decision"].lower()}">{ro["decision"]}</span>'
        rows = [("Applications Declined", "declined"), ("Declined If Risk Is At Least", "cutoff"),
                ("Delinquent Balance Caught", "bad"), ("Performing Balance Turned Away", "good")]
        body = "".join(f'<tr><td class="k">{name}</td><td class="v">{mine[k]}</td><td class="v">{best[k]}</td></tr>' for name, k in rows)
        body += f'<tr><td class="k">This Applicant</td><td>{decision(mine)}</td><td>{decision(best)}</td></tr>'
        st.markdown(f'<table class="tbl"><tr><th></th><th>Your Cutoff</th><th>Optimal Cutoff</th></tr>{body}</table>',
                    unsafe_allow_html=True)
    with chart_col:
        pts = pd.DataFrame([{"x": p["good_pct"], "y": p["bad_pct"]} for p in tool.curve])
        x = alt.X("x:Q", title="Performing balance turned away (%)", scale=alt.Scale(domain=[0, 100]))
        y = alt.Y("y:Q", title="Delinquent balance caught (%)", scale=alt.Scale(domain=[0, 100]))

        def marker(p, label, shape, color, size, dy):
            d = pd.DataFrame([{"x": p["good_pct"], "y": p["bad_pct"], "label": label}])
            return [alt.Chart(d).mark_point(shape=shape, filled=True, size=size, color=color, opacity=1).encode(x, y, tooltip=["label"]),
                    alt.Chart(d).mark_text(align="left", dx=10, dy=dy, color=color, fontSize=12, fontWeight=600).encode(x, y, text="label")]
        lab = min(tool.curve, key=lambda p: abs(p["good_pct"] - 62))
        layers = [alt.Chart(pts).mark_line(color=CURVE, strokeWidth=2).encode(x, y),
                  alt.Chart(pd.DataFrame([{"x": lab["good_pct"], "y": lab["bad_pct"], "t": "LightGBM"}]))
                  .mark_text(dy=-12, color=CURVE, fontSize=12).encode(x, y, text="t")]
        layers += marker(tool.curve[tool.optimal], "Optimal Cutoff", "diamond", OPTIMAL, 130, -10)
        if stop != tool.optimal:
            layers += marker(tool.curve[stop], "Your Cutoff", "circle", ACCENT, 120, -10)
        else:
            layers.append(alt.Chart(pd.DataFrame([{"x": tool.curve[stop]["good_pct"], "y": tool.curve[stop]["bad_pct"]}]))
                          .mark_point(shape="circle", size=260, color=ACCENT, strokeWidth=2).encode(x, y))
        if result is not None and result["applicant_stop"] is not None:
            layers += marker(tool.curve[result["applicant_stop"]], "This Applicant", "triangle-up", APPLICANT, 130, 14)
        chart = (alt.layer(*layers).properties(height=380, background="transparent")
                 .configure_view(strokeWidth=0)
                 .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domainColor="#3A424D", tickColor="#3A424D",
                                 titleFontWeight="normal"))
        st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------------- Main Drivers of Calculated Risk


def driver_table(rows):
    """Variable | Value | Share | bar, on a fixed -100 to +100 scale (gridlines every 10)."""
    out = []
    for r in rows:
        v = r["signed"]
        color = RAISE if r["direction"] > 0 else LOWER if r["direction"] < 0 else NEUTRAL
        left = 50 if v >= 0 else 50 + v / 2
        value = "" if r["variable"] == OTHER else html.escape(str(r["value"]))
        out.append(f'<tr><td>{html.escape(r["label"])}</td><td>{value}</td><td class="num">{v:+.1f}%</td>'
                   f'<td><div class="track"><div class="zero"></div><div class="bar" style="left:{left:.2f}%;width:{abs(v) / 2:.2f}%;'
                   f'background:{color}"></div></div></td></tr>')
    axis = "".join(f'<span style="left:{(t + 100) / 2:.0f}%">{t:+d}</span>'.replace("+0<", "0<") for t in (-100, -50, 0, 50, 100))
    return (f'<table class="drv"><colgroup><col style="width:29%"><col style="width:19%"><col style="width:12%"><col style="width:40%">'
            f'</colgroup><tr><th>Variable</th><th>Value</th><th style="text-align:right">Share</th><th></th></tr>{"".join(out)}'
            f'<tr><td></td><td></td><td></td><td style="border:none"><div class="axis">{axis}</div></td></tr></table>')


with st.container(border=True):
    st.subheader("Main Drivers of Calculated Risk")
    if result is None:
        st.markdown('<div class="lead">Shown after Submit.</div>', unsafe_allow_html=True)
    else:
        for holder, model in zip(st.columns(2, gap="large"), (GBM, LR)):
            with holder:
                st.markdown(f"**{model}**")
                st.markdown(driver_table(result["drivers"][model]), unsafe_allow_html=True)
        st.markdown(f'<div class="note">{NOTES["drivers"]}</div>', unsafe_allow_html=True)
