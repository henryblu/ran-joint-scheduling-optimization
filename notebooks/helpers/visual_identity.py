from __future__ import annotations

"""Shared visual identity definitions for thesis notebooks.

This module centralizes the theme-based palette system used by notebook
helpers. Each notebook should select one institutional identity at setup time
and then consume semantic color roles such as ``primary`` and ``highlight``
instead of hardcoded hex values.

The shared layer intentionally stays small:
1. resolve one supported notebook theme,
2. expose semantic palette roles,
3. apply restrained matplotlib defaults,
4. render clean HTML tables that match the active theme.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd


_THEME_ROLE_NAMES = (
    "primary",
    "secondary",
    "accent",
    "highlight",
    "neutral_dark",
    "neutral_light",
    "background",
    "text",
    "grid",
)


@dataclass(frozen=True)
class NotebookTheme:
    """Semantic color contract shared by notebook helpers."""

    name: str
    primary: str
    secondary: str
    accent: str
    highlight: str
    neutral_dark: str
    neutral_light: str
    background: str
    text: str
    grid: str

    @property
    def default_cycle(self) -> tuple[str, str, str, str]:
        """Return the restrained default plotting cycle for the theme."""

        return (
            self.primary,
            self.secondary,
            self.accent,
            self.neutral_dark,
        )

    def color(self, role: str) -> str:
        """Return one semantic theme color by role name."""

        if role not in _THEME_ROLE_NAMES:
            supported_roles = ", ".join(_THEME_ROLE_NAMES)
            raise KeyError(f"Unknown theme role '{role}'. Supported roles: {supported_roles}.")
        return str(getattr(self, role))

    def colors(self, *roles: str) -> tuple[str, ...]:
        """Return a tuple of semantic colors in the requested role order."""

        if not roles:
            return self.default_cycle
        return tuple(self.color(role) for role in roles)

    def as_dict(self) -> dict[str, str]:
        """Return the theme as a plain role-to-color mapping."""

        return {role: self.color(role) for role in _THEME_ROLE_NAMES}


NOTEBOOK_THEMES: dict[str, NotebookTheme] = {
    "aalto_elec": NotebookTheme(
        name="aalto_elec",
        primary="#6F2DBD",
        secondary="#4B1D8F",
        highlight="#FFD500",
        accent="#000000",
        neutral_dark="#4D4D4D",
        neutral_light="#D9D9D9",
        background="#FFFFFF",
        text="#000000",
        grid="#D9D9D9",
    ),
    "upc_eetac": NotebookTheme(
        name="upc_eetac",
        primary="#003A8F",
        secondary="#2E6FDC",
        highlight="#F39200",
        accent="#F39200",
        neutral_dark="#4D4D4D",
        neutral_light="#D9D9D9",
        background="#FFFFFF",
        text="#000000",
        grid="#D9D9D9",
    ),
}


def list_notebook_themes() -> tuple[str, ...]:
    """Return the supported notebook theme names."""

    return tuple(NOTEBOOK_THEMES)


def get_notebook_theme(theme: str | NotebookTheme) -> NotebookTheme:
    """Resolve one supported notebook theme.

    Accepting an existing ``NotebookTheme`` keeps future helper constructors
    simple: callers can pass either a canonical theme name or a pre-resolved
    theme object.
    """

    if isinstance(theme, NotebookTheme):
        return theme

    theme_name = str(theme).strip().lower()
    resolved_theme = NOTEBOOK_THEMES.get(theme_name)
    if resolved_theme is not None:
        return resolved_theme

    supported_themes = ", ".join(list_notebook_themes())
    raise ValueError(
        f"Unsupported notebook theme '{theme}'. Supported themes: {supported_themes}."
    )


def build_color_cycle(
    theme: str | NotebookTheme,
    *,
    include_highlight: bool = False,
) -> tuple[str, ...]:
    """Return the restrained semantic color cycle for one themed figure."""

    resolved_theme = get_notebook_theme(theme)
    base_cycle = list(resolved_theme.default_cycle[:3])
    if include_highlight:
        base_cycle.append(resolved_theme.highlight)
    return tuple(base_cycle)


def create_themed_figure(
    *,
    theme: str | NotebookTheme,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    gridspec_kw: Mapping[str, object] | None = None,
    constrained_layout: bool = False,
) -> tuple[plt.Figure, object]:
    """Create a subplot figure and apply the shared notebook style."""

    resolved_theme = get_notebook_theme(theme)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        gridspec_kw=None if gridspec_kw is None else dict(gridspec_kw),
        constrained_layout=constrained_layout,
    )
    fig.patch.set_facecolor(resolved_theme.background)

    for ax in _iter_axes(axes):
        apply_axis_style(ax, theme=resolved_theme)

    return fig, axes


def apply_axis_style(
    ax,
    *,
    theme: str | NotebookTheme,
    grid_axis: str = "both",
    use_theme_cycle: bool = True,
    hide_spines: Sequence[str] = ("top", "right"),
) -> None:
    """Apply restrained white-background styling to one matplotlib axis."""

    resolved_theme = get_notebook_theme(theme)

    ax.set_facecolor(resolved_theme.background)
    ax.set_axisbelow(True)
    ax.tick_params(colors=resolved_theme.neutral_dark, labelcolor=resolved_theme.neutral_dark)
    ax.xaxis.label.set_color(resolved_theme.text)
    ax.yaxis.label.set_color(resolved_theme.text)
    ax.title.set_color(resolved_theme.text)

    if use_theme_cycle:
        ax.set_prop_cycle(color=list(build_color_cycle(resolved_theme)))

    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(
            True,
            axis=grid_axis,
            color=resolved_theme.grid,
            linewidth=0.8,
            alpha=0.8,
        )

    for spine_name, spine in ax.spines.items():
        is_hidden = spine_name in hide_spines
        spine.set_visible(not is_hidden)
        if is_hidden:
            continue
        spine.set_color(resolved_theme.neutral_dark)
        spine.set_linewidth(0.9)


def style_legend(legend_or_ax, *, theme: str | NotebookTheme):
    """Style an existing legend or the legend attached to an axis."""

    resolved_theme = get_notebook_theme(theme)
    legend = _resolve_legend(legend_or_ax)
    if legend is None:
        return None

    frame = legend.get_frame()
    frame.set_facecolor(resolved_theme.background)
    frame.set_edgecolor(resolved_theme.grid)
    frame.set_alpha(1.0)
    frame.set_linewidth(0.9)

    for text in legend.get_texts():
        text.set_color(resolved_theme.text)

    return legend


def format_table_for_display(
    df: pd.DataFrame,
    *,
    formats: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return a copy of the table with optional display-only formatting."""

    display_df = df.copy()
    if not formats:
        return display_df

    for column, formatter in formats.items():
        if column not in display_df.columns:
            continue
        display_df[column] = _format_display_column(
            display_df[column],
            formatter=formatter,
        )

    return display_df


def render_html_table(
    df: pd.DataFrame,
    *,
    theme: str | NotebookTheme,
    formats: Mapping[str, object] | None = None,
    caption: str | None = None,
):
    """Render a clean HTML table that matches the active notebook theme."""

    from IPython.display import HTML

    resolved_theme = get_notebook_theme(theme)
    display_df = format_table_for_display(df, formats=formats)
    table_tokens = _build_table_style_tokens(resolved_theme)

    header_cells = "".join(
        (
            '<th style="'
            f'text-align:left; padding:{table_tokens["header_padding"]}; '
            f'border:1px solid {table_tokens["grid"]}; '
            f'background:{table_tokens["header_background"]}; '
            f'color:{table_tokens["header_text"]};'
            f'">{escape(str(column))}</th>'
        )
        for column in display_df.columns
    )

    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(
            (
                '<td style="'
                f'text-align:left; padding:{table_tokens["body_padding"]}; '
                f'border:1px solid {table_tokens["grid"]}; '
                f'background:{table_tokens["body_background"]}; '
                f'color:{table_tokens["body_text"]}; '
                'vertical-align:top;'
                f'">{escape(str(value))}</td>'
            )
            for value in row.tolist()
        )
        body_rows.append(f"<tr>{cells}</tr>")

    caption_html = ""
    if caption:
        caption_html = (
            '<caption style="'
            f'caption-side:top; text-align:left; padding-bottom:6px; '
            f'color:{table_tokens["caption_text"]}; font-weight:600;'
            f'">{escape(caption)}</caption>'
        )

    html = (
        '<table style="'
        f'border-collapse:collapse; background:{table_tokens["table_background"]}; '
        f'color:{table_tokens["body_text"]};'
        '">'
        f"{caption_html}"
        f"<thead><tr>{header_cells}</tr></thead>"
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
    )
    return HTML(html)


def _iter_axes(axes) -> Iterable:
    if hasattr(axes, "flat"):
        for axis in axes.flat:
            yield axis
        return
    if isinstance(axes, Iterable) and not hasattr(axes, "plot"):
        for axis in axes:
            yield axis
        return
    yield axes


def _resolve_legend(legend_or_ax):
    if legend_or_ax is None:
        return None
    if hasattr(legend_or_ax, "get_frame") and hasattr(legend_or_ax, "get_texts"):
        return legend_or_ax
    if hasattr(legend_or_ax, "get_legend"):
        return legend_or_ax.get_legend()
    return None


def _format_table_value(value, *, formatter) -> str:
    if pd.isna(value):
        return ""
    if callable(formatter):
        return str(formatter(value))
    if isinstance(formatter, str):
        return formatter.format(value)
    return str(value)


def _format_display_column(column: pd.Series, *, formatter) -> pd.Series:
    return column.map(
        lambda value: _format_table_value(
            value,
            formatter=formatter,
        )
    )


def _build_table_style_tokens(theme: NotebookTheme) -> dict[str, str]:
    return {
        "table_background": theme.background,
        "header_background": theme.neutral_light,
        "header_text": theme.text,
        "body_background": theme.background,
        "body_text": theme.text,
        "caption_text": theme.neutral_dark,
        "grid": theme.grid,
        "header_padding": "6px 10px",
        "body_padding": "6px 10px",
    }


__all__ = [
    "NOTEBOOK_THEMES",
    "NotebookTheme",
    "apply_axis_style",
    "build_color_cycle",
    "create_themed_figure",
    "format_table_for_display",
    "get_notebook_theme",
    "list_notebook_themes",
    "render_html_table",
    "style_legend",
]
