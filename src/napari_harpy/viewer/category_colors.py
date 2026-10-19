"""Per-category colour selection for table-backed categorical viewer styling.

Harpy resolves a categorical palette from ``table.uns[f"{column}_colors"]`` when a
valid, length-matching palette is stored there (see
:func:`napari_harpy.viewer._styling.resolve_table_categorical_palette`), and falls
back to the default categorical palette otherwise. Both the styled-shapes and
styled-labels paths derive their categories from ``series.cat.categories`` for a
categorical observation column.

This module is the small, framework-free engine that lets a user choose an explicit
colour per category and persist it into that ``uns`` key — the napari-harpy
equivalent of the scanpy ``adata.uns`` colour convention. Writing the palette here
makes the stored palette win on the next (re)styling. The Qt editor in
``napari_harpy.widgets.viewer.category_color_editor`` drives these helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba

from napari_harpy.viewer._styling import (
    is_valid_color,
    normalize_category_value,
    resolve_table_categorical_palette,
)

if TYPE_CHECKING:
    from anndata import AnnData


def uns_colors_key(column: str) -> str:
    """Return the ``table.uns`` key that stores the palette for ``column``."""
    return f"{column}_colors"


def is_editable_category_column(table: AnnData, column: str) -> bool:
    """Return whether ``column`` is a categorical observation column with colours to edit."""
    try:
        category_values(table, column)
    except (KeyError, ValueError):
        return False
    return True


def category_values(table: AnnData, column: str) -> list:
    """Return the ordered categories for a categorical observation column.

    The order matches how harpy derives categories when styling shapes/labels
    (``series.cat.categories``), so a palette written in this order lines up with
    :func:`resolve_table_categorical_palette`.
    """
    if column not in table.obs:
        raise ValueError(f"Observation column `{column}` is not available in the selected table.")
    series = table.obs[column]
    if not isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError(
            f"Observation column `{column}` is not a categorical dtype; per-category colours apply to "
            f"categorical columns. Convert it with "
            f"`table.obs['{column}'] = table.obs['{column}'].astype('category')` to edit colours."
        )
    return [normalize_category_value(value) for value in series.cat.categories]


def storage_hex(color: object) -> str:
    """Return a scanpy-compatible hex string, preserving alpha only when below 1."""
    red, green, blue, alpha = to_rgba(color)
    if alpha >= 1.0:
        return to_hex((red, green, blue))
    return to_hex((red, green, blue, alpha), keep_alpha=True)


def get_category_palette(table: AnnData, column: str) -> tuple[list, list[str]]:
    """Return ``(categories, hex_colors)`` for the currently effective palette.

    ``hex_colors`` is the stored ``uns`` palette when it is valid, otherwise harpy's
    default categorical palette, normalised to hex and aligned to ``categories``.
    """
    categories = category_values(table, column)
    _source, palette = resolve_table_categorical_palette(
        table=table,
        column_name=column,
        categories=categories,
    )
    return categories, [storage_hex(color) for color in palette]


def store_category_colors(table: AnnData, column: str, colors_by_category: Mapping) -> list[str]:
    """Persist an explicit per-category palette into ``table.uns[f"{column}_colors"]``.

    ``colors_by_category`` maps each category to any matplotlib-recognised colour. The
    palette is written as a hex list ordered to match the column's categories, so
    harpy's stored-palette path honours it on the next (re)styling. Returns the stored
    hex list.
    """
    categories = category_values(table, column)
    lookup = {normalize_category_value(key): value for key, value in colors_by_category.items()}
    hex_colors: list[str] = []
    for category in categories:
        if category not in lookup:
            raise ValueError(f"Missing colour for category `{category}` in column `{column}`.")
        color = lookup[category]
        if not is_valid_color(color):
            raise ValueError(f"Invalid colour for category `{category}` in column `{column}`: {color!r}.")
        hex_colors.append(storage_hex(color))
    table.uns[uns_colors_key(column)] = np.array(hex_colors, dtype=object)
    return hex_colors


def reset_category_colors(table: AnnData, column: str) -> bool:
    """Remove any stored palette so harpy falls back to its default. Return whether one was removed."""
    key = uns_colors_key(column)
    if key in table.uns:
        del table.uns[key]
        return True
    return False


def hidden_categories(table: AnnData, column: str) -> frozenset:
    """Return the normalized categories currently *hidden* for a column.

    A category is hidden when its stored palette entry is fully transparent (alpha 0) -- the
    convention the category editor writes when its visibility checkbox is unticked. Returns an
    empty set for a non-categorical column or when no palette is stored (nothing hidden).
    """
    try:
        categories, hex_colors = get_category_palette(table, column)
    except (KeyError, ValueError):
        return frozenset()
    return frozenset(
        normalize_category_value(category)
        for category, hex_color in zip(categories, hex_colors, strict=False)
        if to_rgba(hex_color)[3] <= 0.0
    )
