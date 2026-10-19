from __future__ import annotations

import anndata as ad
import pandas as pd

from napari_harpy.viewer.category_colors import store_category_colors
from napari_harpy.widgets.viewer.category_color_editor import CategoryColorEditor


def _table(categories: tuple[str, ...] = ("A", "B", "C")) -> ad.AnnData:
    series = pd.Categorical(["A", "B", "C", "A"], categories=list(categories))
    return ad.AnnData(obs=pd.DataFrame({"cell_type": series}, index=[f"c{index}" for index in range(4)]))


def test_editor_lists_one_swatch_per_category(qtbot) -> None:
    table = _table(("B", "A", "C"))
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)
    assert set(editor._swatch_buttons) == {"A", "B", "C"}
    assert list(editor.selected_colors()) == ["B", "A", "C"]  # categorical order preserved


def test_editor_apply_writes_chosen_palette_in_category_order(qtbot) -> None:
    table = _table()
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    editor.set_category_color("A", "#ff0000")
    editor.set_category_color("B", "#00ff00")
    editor.set_category_color("C", "#0000ff")

    with qtbot.waitSignal(editor.colors_changed):
        stored = editor.apply()

    assert stored == ["#ff0000", "#00ff00", "#0000ff"]
    assert list(table.uns["cell_type_colors"]) == ["#ff0000", "#00ff00", "#0000ff"]


def test_editor_preserves_alpha_from_picked_color(qtbot) -> None:
    from qtpy.QtGui import QColor

    table = _table()
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    translucent = QColor()
    translucent.setRgbF(1.0, 0.0, 0.0, 0.5)
    editor.set_category_color("A", translucent)
    stored = editor.apply()
    assert stored[0] == "#ff000080"


def test_editor_suppresses_alpha_when_disallowed(qtbot) -> None:
    from qtpy.QtGui import QColor

    table = _table()
    editor = CategoryColorEditor(table, "cell_type", allow_alpha=False)
    qtbot.addWidget(editor)

    translucent = QColor()
    translucent.setRgbF(1.0, 0.0, 0.0, 0.5)
    editor.set_category_color("A", translucent)
    stored = editor.apply()
    assert stored[0] == "#ff0000"  # alpha forced opaque for shapes-style usage


def test_editor_reset_removes_stored_palette(qtbot) -> None:
    table = _table()
    store_category_colors(table, "cell_type", {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"})
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    with qtbot.waitSignal(editor.colors_changed):
        editor.reset_to_default()

    assert "cell_type_colors" not in table.uns


def test_editor_hidden_category_is_written_as_transparent(qtbot) -> None:
    table = _table()
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    editor.set_category_color("A", "#ff0000")
    editor.set_category_color("B", "#00ff00")
    editor.set_category_color("C", "#0000ff")
    editor.set_category_visible("B", False)

    stored = editor.apply()
    assert stored == ["#ff0000", "#00ff0000", "#0000ff"]  # B hidden -> alpha 0, RGB preserved
    assert editor.hidden_categories() == {"B"}


def test_editor_initial_visibility_reflects_stored_alpha(qtbot) -> None:
    table = _table()
    store_category_colors(table, "cell_type", {"A": "#ff0000", "B": (0.0, 1.0, 0.0, 0.0), "C": "#0000ff"})

    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    assert editor.hidden_categories() == {"B"}
    assert editor._visible_checkboxes["B"].isChecked() is False
    assert editor._visible_checkboxes["A"].isChecked() is True


def test_editor_hide_all_then_show_all(qtbot) -> None:
    table = _table()
    editor = CategoryColorEditor(table, "cell_type")
    qtbot.addWidget(editor)

    editor._set_all_visible(False)
    assert editor.hidden_categories() == {"A", "B", "C"}

    editor._set_all_visible(True)
    assert editor.hidden_categories() == set()
