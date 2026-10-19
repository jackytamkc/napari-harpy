from __future__ import annotations

import anndata as ad
import pandas as pd
import pytest

from napari_harpy.viewer._styling import categorical_rgba_for_values, resolve_table_categorical_palette
from napari_harpy.viewer.category_colors import (
    category_values,
    get_category_palette,
    is_editable_category_column,
    reset_category_colors,
    storage_hex,
    store_category_colors,
    uns_colors_key,
)


def _table(categories: tuple[str, ...] = ("A", "B", "C")) -> ad.AnnData:
    series = pd.Categorical(["A", "B", "C", "A"], categories=list(categories))
    return ad.AnnData(obs=pd.DataFrame({"cell_type": series}, index=[f"c{index}" for index in range(4)]))


def test_category_values_follows_categorical_order() -> None:
    table = _table(("B", "A", "C"))
    assert category_values(table, "cell_type") == ["B", "A", "C"]


def test_category_values_rejects_non_categorical() -> None:
    table = ad.AnnData(obs=pd.DataFrame({"score": [1.0, 2.0]}, index=["a", "b"]))
    assert is_editable_category_column(table, "score") is False
    assert is_editable_category_column(table, "missing") is False
    with pytest.raises(ValueError, match="not a categorical dtype"):
        category_values(table, "score")


def test_get_category_palette_defaults_when_unstored() -> None:
    table = _table()
    categories, colors = get_category_palette(table, "cell_type")
    assert categories == ["A", "B", "C"]
    assert len(colors) == 3
    assert all(color.startswith("#") for color in colors)


def test_store_category_colors_writes_ordered_palette_and_is_honored() -> None:
    table = _table()
    stored = store_category_colors(table, "cell_type", {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"})
    assert stored == ["#ff0000", "#00ff00", "#0000ff"]
    assert list(table.uns[uns_colors_key("cell_type")]) == ["#ff0000", "#00ff00", "#0000ff"]

    # Harpy's own resolver now reports the palette as user-stored and returns it verbatim.
    source, palette = resolve_table_categorical_palette(
        table=table, column_name="cell_type", categories=["A", "B", "C"]
    )
    assert source == "stored"
    assert list(palette) == ["#ff0000", "#00ff00", "#0000ff"]

    # End-to-end: the chosen colours drive the RGBA the styling path produces.
    rgba = categorical_rgba_for_values(
        pd.Series(pd.Categorical(["B", "A"], categories=["A", "B", "C"])),
        categories=["A", "B", "C"],
        palette=palette,
    )
    assert rgba[0].tolist() == [0.0, 1.0, 0.0, 1.0]  # B -> green
    assert rgba[1].tolist() == [1.0, 0.0, 0.0, 1.0]  # A -> red


def test_storage_hex_preserves_alpha_only_when_below_one() -> None:
    assert storage_hex("#ff0000") == "#ff0000"
    assert storage_hex((1.0, 0.0, 0.0, 0.5)) == "#ff000080"


def test_store_category_colors_requires_all_categories() -> None:
    table = _table()
    with pytest.raises(ValueError, match="Missing colour for category"):
        store_category_colors(table, "cell_type", {"A": "#ff0000", "B": "#00ff00"})


def test_store_category_colors_rejects_invalid_color() -> None:
    table = _table()
    with pytest.raises(ValueError, match="Invalid colour"):
        store_category_colors(table, "cell_type", {"A": "definitely-not-a-color", "B": "#00ff00", "C": "#0000ff"})


def test_reset_category_colors_removes_key() -> None:
    table = _table()
    store_category_colors(table, "cell_type", {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"})
    assert reset_category_colors(table, "cell_type") is True
    assert uns_colors_key("cell_type") not in table.uns
    assert reset_category_colors(table, "cell_type") is False
