from __future__ import annotations

from dataclasses import dataclass, replace

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
)

from napari_harpy.core._color_source import (
    ShapeColorSourceKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceKind,
    TableColorSourceSpec,
    gene_module_value_key,
    parse_gene_module_text,
    validate_gene_module_genes,
)
from napari_harpy.viewer._styling import CONTINUOUS_COLORMAP_CHOICES, OVERLAY_CONTINUOUS_COLORMAP
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    CompactComboBox,
    build_input_control_stylesheet,
    create_form_label,
)
from napari_harpy.widgets.viewer.disclosure import _ElidedLabel
from napari_harpy.widgets.viewer.module_smoothing import ModuleSmoothingControl
from napari_harpy.widgets.viewer.styles import (
    CARD_TITLE_STYLESHEET,
    DETAIL_PANEL_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
    SUMMARY_LABEL_STYLESHEET,
)


@dataclass(frozen=True)
class ShapesLoadRequest:
    shapes_name: str
    table_name: str | None
    selected_source_kind: ShapeColorSourceKind | TableColorSourceKind | None
    selected_color_source: ShapeColumnColorSourceSpec | TableColorSourceSpec | None
    fill_shapes: bool


class _ShapesCardWidget(QFrame):
    """Card UI shell for one shapes element in the selected coordinate system."""

    add_update_requested = Signal(object)
    edit_colors_requested = Signal(object)

    def __init__(
        self,
        *,
        shapes_name: str,
        shape_column_color_sources: list[ShapeColumnColorSourceSpec],
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
        table_layers_by_table: dict[str, list[str]] | None = None,
        table_smoothers_by_table: dict[str, list[str]] | None = None,
        table_umap_by_table: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.shapes_name = shapes_name
        self._shape_column_color_sources = shape_column_color_sources
        self._table_color_sources_by_table = table_color_sources_by_table
        self._table_layers_by_table = table_layers_by_table or {}
        self._table_smoothers_by_table = table_smoothers_by_table or {}
        self._table_umap_by_table = table_umap_by_table or {}
        self._expression_layer_by_table: dict[str, str] = {}
        self._filtered_color_sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = []
        self._module_gene_text = ""
        self.setObjectName(f"viewer_widget_shapes_card_{shapes_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(shapes_name, self)
        self.title_label.setObjectName(f"viewer_widget_shapes_card_title_{shapes_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox(minimum_contents_length=8)
        self.linked_table_combo.setObjectName(f"viewer_widget_shapes_linked_table_combo_{shapes_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        table_names = list(table_color_sources_by_table)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_shapes_color_source_kind_combo_{shapes_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Shapes column", "shape_column")
        self.color_source_kind_combo.addItem("Observations", "obs_column")
        self.color_source_kind_combo.addItem("Vars", "x_var")
        self.color_source_kind_combo.addItem("Gene module", "x_module")

        self.color_source_value_label = create_form_label("Shapes column")
        self.color_source_value_input = QLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_shapes_color_source_value_input_{shapes_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self.fill_toggle = QCheckBox("Fill")
        self.fill_toggle.setObjectName(f"viewer_widget_shapes_fill_toggle_{shapes_name}")
        self.fill_toggle.setStyleSheet(CHECKBOX_STYLESHEET)
        self.fill_toggle.setChecked(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._color_source_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        # Observation columns are few and researcher-named -> a scrollable dropdown (like the other combos).
        # Shape columns and vars (the large gene list) keep the type-to-filter line edit. A stack swaps them.
        self.color_source_value_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_value_combo.setObjectName(f"viewer_widget_shapes_color_source_value_combo_{shapes_name}")
        self.color_source_value_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_value_combo.setEnabled(False)
        self.color_source_value_stack = QStackedWidget()
        self.color_source_value_stack.addWidget(self.color_source_value_combo)  # index 0: observation columns
        self.color_source_value_stack.addWidget(self.color_source_value_input)  # index 1: shape columns / vars
        self.color_source_value_stack.setCurrentWidget(self.color_source_value_input)

        self.layer_label = create_form_label("Layer")
        self.layer_combo = CompactComboBox(minimum_contents_length=8)
        self.layer_combo.setObjectName(f"viewer_widget_shapes_layer_combo_{shapes_name}")
        self.layer_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.layer_combo.setToolTip(
            "Expression layer to read gene values from (X, or a table layer like log1p/smoothed)"
        )

        self.smooth_label = create_form_label("Smoothing")
        self.smooth_checkbox = QCheckBox("Smooth over UMAP neighbours")
        self.smooth_checkbox.setObjectName(f"viewer_widget_shapes_smooth_checkbox_{shapes_name}")
        self.smooth_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.smooth_checkbox.setToolTip(
            "Average each gene over every cell's UMAP neighbourhood on the fly (obsp smoother)"
        )

        self.module_smoothing_label = create_form_label("Smoothing")
        self.module_smoothing = ModuleSmoothingControl()
        self.module_smoothing.setObjectName(f"viewer_widget_shapes_module_smoothing_{shapes_name}")

        self.colormap_label = create_form_label("Colourmap")
        self.colormap_combo = CompactComboBox(minimum_contents_length=8)
        self.colormap_combo.setObjectName(f"viewer_widget_shapes_colormap_combo_{shapes_name}")
        self.colormap_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.colormap_combo.addItems(list(CONTINUOUS_COLORMAP_CHOICES))
        self.colormap_combo.setToolTip("Matplotlib colourmap for continuous (gene, module, or numeric) colouring")

        self.action_hint_label = QLabel()
        self.action_hint_label.setObjectName(f"viewer_widget_shapes_action_hint_{shapes_name}")
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet(SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_shapes_button_{shapes_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        self.edit_colors_button = QPushButton("Edit colours…")
        self.edit_colors_button.setObjectName(f"viewer_widget_edit_shapes_colors_button_{shapes_name}")
        self.edit_colors_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_colors_button.setMinimumHeight(28)
        self.edit_colors_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.edit_colors_button.setEnabled(False)
        self.edit_colors_button.setToolTip("Pick a colour per category for the selected observation column")
        self.edit_colors_button.clicked.connect(self._emit_edit_colors_request)

        self._form_layout = form_layout
        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_stack)
        form_layout.addRow(self.layer_label, self.layer_combo)
        form_layout.addRow(self.smooth_label, self.smooth_checkbox)
        form_layout.addRow(self.module_smoothing_label, self.module_smoothing)
        form_layout.addRow(self.colormap_label, self.colormap_combo)
        form_layout.addRow(create_form_label("Display"), self.fill_toggle)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_hint_label)
        layout.addWidget(self.add_update_button)
        layout.addWidget(self.edit_colors_button)

        self.linked_table_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_kind_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_value_input.textChanged.connect(self._on_color_source_text_changed)
        self.color_source_value_input.editingFinished.connect(self._sync_current_source_selection)
        self.color_source_value_combo.currentIndexChanged.connect(self._update_action_status)
        self.layer_combo.currentIndexChanged.connect(self._on_expression_layer_changed)
        self.smooth_checkbox.toggled.connect(self._update_action_status)
        self.module_smoothing.changed.connect(self._update_action_status)
        self.colormap_combo.currentIndexChanged.connect(self._update_action_status)

        self._refresh_color_source_controls()

    @property
    def selected_table_name(self) -> str | None:
        if not self.linked_table_combo.isEnabled():
            return None

        table_name = self.linked_table_combo.currentText()
        return table_name if table_name in self._table_color_sources_by_table else None

    @property
    def selected_source_kind(self) -> ShapeColorSourceKind | TableColorSourceKind | None:
        value = self.color_source_kind_combo.currentData()
        return value if value in {"shape_column", "obs_column", "x_var", "x_module"} else None

    @property
    def selected_color_source(self) -> ShapeColumnColorSourceSpec | TableColorSourceSpec | None:
        if self.selected_source_kind == "x_module":
            genes, _error = self._validated_module_genes()
            if genes is None or self.selected_table_name is None or self._module_smoothing_error() is not None:
                return None
            return TableColorSourceSpec(
                table_name=self.selected_table_name,
                source_kind="x_module",
                value_key=gene_module_value_key(genes),
                value_kind="continuous",
                layer=self._selected_layer(),
                colormap=self._selected_colormap(),
                module_genes=genes,
                umap_neighbors=self.module_smoothing.n_neighbors,
            )

        current_text = self._current_value_text()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                if isinstance(source, TableColorSourceSpec) and source.source_kind == "x_var":
                    source = replace(source, layer=self._selected_layer(), smoother=self._selected_smoother())
                if source.value_kind == "continuous":
                    source = replace(source, colormap=self._selected_colormap())
                return source
        return None

    def _validated_module_genes(self) -> tuple[tuple[str, ...] | None, str | None]:
        genes = parse_gene_module_text(self.color_source_value_input.text())
        table_name = self.selected_table_name
        available_var_names = (
            tuple(
                source.value_key
                for source in self._table_color_sources_by_table.get(table_name, ())
                if source.source_kind == "x_var"
            )
            if table_name is not None
            else ()
        )
        try:
            return validate_gene_module_genes(genes, available_var_names=available_var_names), None
        except ValueError as error:
            return None, str(error)

    def _module_smoothing_error(self) -> str | None:
        if self.module_smoothing.n_neighbors is not None and not self._table_umap_by_table.get(
            self.selected_table_name, False
        ):
            return (
                "UMAP smoothing requires X_umap in the linked table. "
                "For an already smoothed layer, disable UMAP smoothing."
            )
        return None

    def _on_expression_layer_changed(self, _index: int) -> None:
        table_name = self.selected_table_name
        layer = self.layer_combo.currentText()
        if table_name is not None and layer:
            self._expression_layer_by_table[table_name] = layer
        self._update_action_status()

    def _current_value_text(self) -> str:
        """Return the selected value from whichever widget is shown (combo for obs columns, line edit otherwise)."""
        if self.color_source_value_stack.currentWidget() is self.color_source_value_combo:
            return self.color_source_value_combo.currentText().strip()
        return self.color_source_value_input.text().strip()

    def _selected_layer(self) -> str | None:
        text = self.layer_combo.currentText().strip()
        return None if text in ("", "X") else text

    def _selected_smoother(self) -> str | None:
        if not self.smooth_checkbox.isChecked():
            return None
        smoothers = self._table_smoothers_by_table.get(self.selected_table_name) or []
        return smoothers[0] if smoothers else None

    def _selected_colormap(self) -> str | None:
        # None for the viridis default (like _selected_layer returns None for "X"), so unchanged
        # selections keep specs/presets minimal; the styling layer resolves None -> viridis.
        text = self.colormap_combo.currentText().strip()
        return None if text in ("", OVERLAY_CONTINUOUS_COLORMAP) else text

    @property
    def fill_shapes(self) -> bool:
        return self.fill_toggle.isChecked()

    def set_linked_tables(
        self,
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
        table_layers_by_table: dict[str, list[str]] | None = None,
        table_smoothers_by_table: dict[str, list[str]] | None = None,
        table_umap_by_table: dict[str, bool] | None = None,
    ) -> None:
        if table_layers_by_table is not None:
            self._table_layers_by_table = table_layers_by_table
        if table_smoothers_by_table is not None:
            self._table_smoothers_by_table = table_smoothers_by_table
        if table_umap_by_table is not None:
            self._table_umap_by_table = table_umap_by_table
        previous_table_name = self.selected_table_name
        previous_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )

        self._table_color_sources_by_table = table_color_sources_by_table
        table_names = list(table_color_sources_by_table)
        with QSignalBlocker(self.linked_table_combo):
            self.linked_table_combo.clear()
            if table_names:
                self.linked_table_combo.addItems(table_names)
                self.linked_table_combo.setEnabled(True)
                next_index = table_names.index(previous_table_name) if previous_table_name in table_names else 0
                self.linked_table_combo.setCurrentIndex(next_index)
            else:
                self.linked_table_combo.addItem("No linked tables")
                self.linked_table_combo.setEnabled(False)
                self.linked_table_combo.setCurrentIndex(0)

        self._refresh_color_source_controls(preferred_source_identity=previous_source_identity)

    def _refresh_color_source_controls(
        self,
        _index: int | None = None,
        *,
        preferred_source_identity: tuple[object, ...] | None = None,
    ) -> None:
        if preferred_source_identity is not None:
            selected_source_identity = preferred_source_identity
        else:
            selected_source_identity = (
                self.selected_color_source.identity if self.selected_color_source is not None else None
            )
        source_kind = self.selected_source_kind

        if source_kind == "shape_column":
            self.color_source_value_label.setText("Shapes column")
            available_sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = list(
                self._shape_column_color_sources
            )
            placeholder_text = "Search shapes columns"
        elif source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
            table_name = self.selected_table_name
            available_sources = (
                list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
            )
            placeholder_text = "Search observations"
        elif source_kind == "x_var":
            self.color_source_value_label.setText("Var")
            table_name = self.selected_table_name
            available_sources = (
                list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
            )
            placeholder_text = "Search vars"
        elif source_kind == "x_module":
            self.color_source_value_label.setText("Genes")
            table_name = self.selected_table_name
            available_sources = (
                list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
            )
            placeholder_text = "Comma-separated genes (at least two)"
        else:
            self.color_source_value_label.setText("Value source")
            available_sources = []
            placeholder_text = "Select a color source kind first"

        filtered_kind = "x_var" if source_kind == "x_module" else source_kind
        self._filtered_color_sources = [
            source for source in available_sources if filtered_kind is None or source.source_kind == filtered_kind
        ]

        display_names = [source.display_name for source in self._filtered_color_sources]
        target_name = None
        if selected_source_identity is not None:
            match = next((s for s in self._filtered_color_sources if s.identity == selected_source_identity), None)
            if match is not None:
                target_name = match.display_name
        if target_name is None and display_names and source_kind != "x_module":
            target_name = display_names[0]

        if source_kind == "obs_column":
            # Observation columns are few and researcher-named -> a scrollable dropdown (like the other combos).
            with QSignalBlocker(self.color_source_value_combo):
                self.color_source_value_combo.clear()
                self.color_source_value_combo.addItems(display_names)
                self.color_source_value_combo.setEnabled(bool(display_names))
                if target_name is not None:
                    self.color_source_value_combo.setCurrentText(target_name)
            self.color_source_value_stack.setCurrentWidget(self.color_source_value_combo)
        else:
            # Vars (large gene list) -> keep the type-to-filter line edit; None -> disabled line edit.
            with QSignalBlocker(self.color_source_value_input):
                self.color_source_value_input.setEnabled(source_kind is not None and bool(display_names))
                self.color_source_value_input.setText(
                    self._module_gene_text if source_kind == "x_module" else target_name or ""
                )
                self.color_source_value_input.setPlaceholderText(placeholder_text)
                # A normal QCompleter replaces the entire line, which is destructive for a
                # comma-separated module after its first gene. Keep it for single-var search only.
                self._color_source_completer_model.setStringList([] if source_kind == "x_module" else display_names)
            self.color_source_value_stack.setCurrentWidget(self.color_source_value_input)

        is_expression_source = source_kind in {"x_var", "x_module"}
        with QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            if is_expression_source:
                self.layer_combo.addItems(self._table_layers_by_table.get(self.selected_table_name) or ["X"])
                previous_layer = self._expression_layer_by_table.get(self.selected_table_name, "X")
                previous_index = self.layer_combo.findText(previous_layer)
                self.layer_combo.setCurrentIndex(max(0, previous_index))
        self._set_layer_row_visible(is_expression_source)

        smoothers = self._table_smoothers_by_table.get(self.selected_table_name) or []
        if smoothers:
            self.smooth_checkbox.setText(f"Smooth over UMAP neighbours ({smoothers[0]})")
        self._set_smooth_row_visible(source_kind == "x_var" and bool(smoothers))
        if hasattr(self._form_layout, "setRowVisible"):
            self._form_layout.setRowVisible(self.module_smoothing, source_kind == "x_module")
        else:  # pragma: no cover - older Qt
            self.module_smoothing_label.setVisible(source_kind == "x_module")
            self.module_smoothing.setVisible(source_kind == "x_module")

        self._update_action_status()

    def _set_layer_row_visible(self, visible: bool) -> None:
        if hasattr(self._form_layout, "setRowVisible"):
            self._form_layout.setRowVisible(self.layer_combo, visible)
        else:  # pragma: no cover - older Qt without QFormLayout.setRowVisible
            self.layer_label.setVisible(visible)
            self.layer_combo.setVisible(visible)

    def _set_smooth_row_visible(self, visible: bool) -> None:
        if hasattr(self._form_layout, "setRowVisible"):
            self._form_layout.setRowVisible(self.smooth_checkbox, visible)
        else:  # pragma: no cover - older Qt without QFormLayout.setRowVisible
            self.smooth_label.setVisible(visible)
            self.smooth_checkbox.setVisible(visible)

    def _set_colormap_row_visible(self, visible: bool) -> None:
        if hasattr(self._form_layout, "setRowVisible"):
            self._form_layout.setRowVisible(self.colormap_combo, visible)
        else:  # pragma: no cover - older Qt without QFormLayout.setRowVisible
            self.colormap_label.setVisible(visible)
            self.colormap_combo.setVisible(visible)

    def _sync_current_source_selection(self) -> None:
        if not self.color_source_value_input.isEnabled():
            return
        if self.selected_source_kind == "x_module":
            self._update_action_status()
            return

        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                self.color_source_value_input.setText(source.display_name)
                break
        self._update_action_status()

    def _on_color_source_text_changed(self, text: str) -> None:
        if self.selected_source_kind == "x_module":
            self._module_gene_text = text
        self._update_action_status()

    def _update_action_status(self) -> None:
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name
        selected_source = self.selected_color_source
        self._update_fill_toggle_enabled(selected_source is not None)
        self.edit_colors_button.setEnabled(selected_source is not None and selected_source.source_kind == "obs_column")
        # The colourmap only affects continuous (gene / numeric) colouring; hide it for categorical.
        self._set_colormap_row_visible(selected_source is not None and selected_source.value_kind == "continuous")

        if source_kind is None:
            self.action_hint_label.setText("Action: add/update primary shapes layer")
            return

        if source_kind in {"obs_column", "x_var", "x_module"} and table_name is None:
            self.action_hint_label.setText("Action: table-backed shapes coloring requires a linked table")
            return

        if selected_source is None:
            if source_kind == "shape_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a shapes column for a styled shapes layer")
            elif source_kind == "shape_column":
                self.action_hint_label.setText("Action: no colorable shapes columns available")
            elif source_kind == "obs_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select an observation column for a styled shapes layer")
            elif source_kind == "obs_column":
                self.action_hint_label.setText("Action: no colorable observation columns available")
            elif source_kind == "x_module" and not self._filtered_color_sources:
                self.action_hint_label.setText("Action: no vars available for a gene-module layer")
            elif source_kind == "x_module":
                _genes, error = self._validated_module_genes()
                self.action_hint_label.setText(f"Action: {error or self._module_smoothing_error()}")
            elif self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a var for a styled shapes layer")
            else:
                self.action_hint_label.setText("Action: no vars available for a styled shapes layer")
            return

        if selected_source.source_kind == "shape_column":
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for column "{selected_source.value_key}"'
            )
        elif selected_source.source_kind == "obs_column":
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for obs["{selected_source.value_key}"]'
            )
        elif selected_source.source_kind == "x_module":
            layer = self._selected_layer() or "X"
            k = self.module_smoothing.n_neighbors
            smoothed = f" with UMAP smoothing (k={k}, including self)" if k is not None else " (selected layer only)"
            self.action_hint_label.setText(
                "Action: add/update styled shapes layer for module score "
                f"(mean z-score of {len(selected_source.module_genes)} genes) from {layer}{smoothed}"
            )
        else:
            layer = self._selected_layer() or "X"
            smoothed = " (smoothed)" if self._selected_smoother() else ""
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for {layer}[:, "{selected_source.value_key}"]{smoothed}'
            )

    def _update_fill_toggle_enabled(self, enabled: bool) -> None:
        self.fill_toggle.setEnabled(enabled)
        if not enabled:
            self.fill_toggle.setChecked(False)

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ShapesLoadRequest(
                shapes_name=self.shapes_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
                fill_shapes=self.fill_shapes,
            )
        )

    def _emit_edit_colors_request(self, _checked: bool = False) -> None:
        self.edit_colors_requested.emit(
            ShapesLoadRequest(
                shapes_name=self.shapes_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
                fill_shapes=self.fill_shapes,
            )
        )
