
import json
import re
import plotly.graph_objects as go


def extract_chart(answer: str):
    """Answer se chart block extract karta hai."""
    pattern = r"```chart\s*\n(.*?)\n```"
    match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)

    if not match:
        return answer, None

    clean_text = (answer[:match.start()] + answer[match.end():]).strip()

    try:
        spec = json.loads(match.group(1))

        x = spec["x"]
        series = spec["series"]
        kind = spec.get("type", "line")

        if not isinstance(x, list) or len(x) < 2:
            return clean_text, None

        if not isinstance(series, list) or not series:
            return clean_text, None

        for s in series:
            if len(s["values"]) != len(x):
                return clean_text, None

        if kind not in ("line", "bar", "pie"):
            return clean_text, None

        return clean_text, spec

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return clean_text, None


def build_figure(spec):
    """Chart specification se Plotly figure banata hai."""
    kind = spec.get("type", "line")
    x = spec["x"]
    series = spec["series"]

    fig = go.Figure()

    if kind == "pie":
        fig.add_trace(
            go.Pie(
                labels=x,
                values=series[0]["values"],
                name=series[0].get("name", "Values")
            )
        )

    elif kind == "bar":
        for s in series:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=s["values"],
                    name=s.get("name", "Series")
                )
            )

        fig.update_layout(barmode="group")

    else:
        for s in series:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=s["values"],
                    name=s.get("name", "Series"),
                    mode="lines+markers"
                )
            )

    fig.update_layout(
        title=spec.get("title", "Financial Chart"),
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=40),
        legend=dict(orientation="h")
    )

    return fig
