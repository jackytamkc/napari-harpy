"""A small non-modal editor to choose an explicit colour and visibility per obs category.

Colours are persisted into ``table.uns[f"{column}_colors"]`` via
:mod:`napari_harpy.viewer.category_colors`, so harpy's stored-palette path honours them on
the next (re)styling. A category is *hidden* by writing its palette entry with alpha 0 — the
labels colormap and the shapes/points styling both render an alpha-0 category transparent —
so visibility shares the single stored palette (no extra ``uns`` key). ``colors_changed`` is
emitted after a successful write so the caller can re-apply the styled layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.colors import to_rgba
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_harpy.viewer._styling import normalize_category_value
from napari_harpy.viewer.category_colors import (
    get_category_palette,
    reset_category_colors,
    store_category_colors,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _to_qcolor(color: Any) -> QColor:
    """Build a QColor from any matplotlib-recognised colour (hex, name, or rgba tuple)."""
    red, green, blue, alpha = to_rgba(color)
    qcolor = QColor()
    qcolor.setRgbF(float(red), float(green), float(blue), float(alpha))
    return qcolor


def _swatch_stylesheet(qcolor: QColor, *, hidden: bool = False) -> str:
    border = "1px dashed #808080" if hidden else "1px solid #202020"
    return (
        f"background-color: rgba({qcolor.red()}, {qcolor.green()}, {qcolor.blue()}, {qcolor.alphaF():.3f});"
        f" border: {border}; min-width: 52px; min-height: 18px;"
    )


class CategoryColorEditor(QDialog):
    """Non-modal per-category colour + visibility editor for a categorical observation column.

    ``allow_alpha`` controls whether the colour picker exposes an alpha channel. It is enabled
    for labels (whose colormap honours RGBA) and disabled for shapes (whose styling forces its
    own edge/face alpha, so a per-category colour alpha would be misleading). Visibility is
    independent of ``allow_alpha`` and works for both via an alpha-0 stored palette entry.
    """

    colors_changed = Signal()

    def __init__(
        self,
        table: AnnData,
        column: str,
        *,
        allow_alpha: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Destroy the C++ object on close so closed editors do not accumulate as hidden children.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._table = table
        self._column = column
        self._allow_alpha = allow_alpha
        self._swatch_buttons: dict[Any, QPushButton] = {}
        self._visible_checkboxes: dict[Any, QCheckBox] = {}
        self._colors: dict[Any, Any] = {}
        self._hidden: set[Any] = set()

        self.setObjectName("category_color_editor")
        self.setWindowTitle(f"Edit colours — {column}")

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(QLabel(f'Category colours for "{column}"'))

        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._rows_host)
        root_layout.addWidget(scroll_area, 1)

        button_row = QHBoxLayout()
        self.show_all_button = QPushButton("Show all")
        self.hide_all_button = QPushButton("Hide all")
        self.reset_button = QPushButton("Reset to default")
        self.apply_button = QPushButton("Apply")
        self.close_button = QPushButton("Close")
        self.show_all_button.setObjectName("category_color_editor_show_all_button")
        self.hide_all_button.setObjectName("category_color_editor_hide_all_button")
        self.reset_button.setObjectName("category_color_editor_reset_button")
        self.apply_button.setObjectName("category_color_editor_apply_button")
        self.close_button.setObjectName("category_color_editor_close_button")
        button_row.addWidget(self.show_all_button)
        button_row.addWidget(self.hide_all_button)
        button_row.addStretch(1)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.close_button)
        root_layout.addLayout(button_row)

        self.show_all_button.clicked.connect(lambda _checked=False: self._set_all_visible(True))
        self.hide_all_button.clicked.connect(lambda _checked=False: self._set_all_visible(False))
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.close_button.clicked.connect(self.reject)

        self._populate_rows()

    @property
    def table(self) -> AnnData:
        """The AnnData table whose palette this editor edits."""
        return self._table

    @property
    def column(self) -> str:
        """The observation column whose category colours this editor edits."""
        return self._column

    def _populate_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._swatch_buttons.clear()
        self._visible_checkboxes.clear()
        self._colors.clear()
        self._hidden.clear()

        categories, hex_colors = get_category_palette(self._table, self._column)
        for category, hex_color in zip(categories, hex_colors, strict=True):
            key = normalize_category_value(category)
            red, green, blue, alpha = to_rgba(hex_color)
            is_hidden = alpha <= 0.0
            # Keep the RGB visible in the swatch even when hidden, so the colour is recoverable.
            self._colors[key] = (red, green, blue, 1.0 if is_hidden else alpha)
            if is_hidden:
                self._hidden.add(key)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            checkbox = QCheckBox()
            checkbox.setObjectName(f"category_visible_checkbox_{category}")
            checkbox.setChecked(not is_hidden)
            checkbox.setToolTip("Show / hide this category")
            checkbox.toggled.connect(lambda checked, category_key=key: self._on_visible_toggled(category_key, checked))

            swatch = QPushButton()
            swatch.setObjectName(f"category_color_swatch_{category}")
            swatch.setStyleSheet(_swatch_stylesheet(_to_qcolor(self._colors[key]), hidden=is_hidden))
            swatch.clicked.connect(lambda _checked=False, category_key=key: self._open_color_dialog(category_key))

            row_layout.addWidget(checkbox)
            row_layout.addWidget(swatch)
            row_layout.addWidget(QLabel(str(category)), 1)
            self._rows_layout.addWidget(row)
            self._swatch_buttons[key] = swatch
            self._visible_checkboxes[key] = checkbox
        self._rows_layout.addStretch(1)

    def _open_color_dialog(self, category: Any) -> None:
        initial = _to_qcolor(self._colors[category])
        title = f"Colour for {category}"
        if self._allow_alpha:
            chosen = QColorDialog.getColor(initial, self, title, QColorDialog.ColorDialogOption.ShowAlphaChannel)
        else:
            chosen = QColorDialog.getColor(initial, self, title)
        if chosen.isValid():
            self.set_category_color(category, chosen)

    def set_category_color(self, category: Any, color: Any) -> None:
        """Set a category's colour in the editor (used by the colour dialog and tests)."""
        key = self._require_category(category)
        qcolor = color if isinstance(color, QColor) else _to_qcolor(color)
        if not self._allow_alpha:
            qcolor = QColor(qcolor)
            qcolor.setAlphaF(1.0)
        self._colors[key] = (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), qcolor.alphaF())
        self._refresh_swatch(key)

    def set_category_visible(self, category: Any, visible: bool) -> None:
        """Show or hide a category in the editor (used by the checkboxes, Show/Hide all, and tests)."""
        key = self._require_category(category)
        if visible:
            self._hidden.discard(key)
        else:
            self._hidden.add(key)
        checkbox = self._visible_checkboxes[key]
        if checkbox.isChecked() != visible:
            checkbox.blockSignals(True)
            checkbox.setChecked(visible)
            checkbox.blockSignals(False)
        self._refresh_swatch(key)

    def selected_colors(self) -> dict[Any, Any]:
        """Return the current ``category -> colour`` mapping (RGB shown in the swatch)."""
        return dict(self._colors)

    def hidden_categories(self) -> set[Any]:
        """Return the set of categories currently toggled off."""
        return set(self._hidden)

    def apply(self) -> list[str]:
        """Persist the chosen palette (hidden categories as alpha 0) and emit ``colors_changed``."""
        colors_by_category: dict[Any, Any] = {}
        for key, color in self._colors.items():
            if key in self._hidden:
                red, green, blue, _alpha = to_rgba(color)
                colors_by_category[key] = (red, green, blue, 0.0)
            else:
                colors_by_category[key] = color
        stored = store_category_colors(self._table, self._column, colors_by_category)
        self.colors_changed.emit()
        return stored

    def reset_to_default(self) -> None:
        """Drop the stored palette, reload defaults into the editor, and emit ``colors_changed``."""
        reset_category_colors(self._table, self._column)
        self._populate_rows()
        self.colors_changed.emit()

    def _on_visible_toggled(self, category: Any, checked: bool) -> None:
        if checked:
            self._hidden.discard(category)
        else:
            self._hidden.add(category)
        self._refresh_swatch(category)

    def _set_all_visible(self, visible: bool) -> None:
        for key in self._colors:
            self.set_category_visible(key, visible)

    def _refresh_swatch(self, category: Any) -> None:
        key = normalize_category_value(category)
        self._swatch_buttons[key].setStyleSheet(
            _swatch_stylesheet(_to_qcolor(self._colors[key]), hidden=key in self._hidden)
        )

    def _require_category(self, category: Any) -> Any:
        key = normalize_category_value(category)
        if key not in self._colors:
            raise KeyError(f"Unknown category {category!r} for column {self._column!r}.")
        return key

    def _on_apply_clicked(self) -> None:
        # Apply leaves the editor open so colours can be tuned iteratively with live re-styling.
        self.apply()

    def _on_reset_clicked(self) -> None:
        self.reset_to_default()
