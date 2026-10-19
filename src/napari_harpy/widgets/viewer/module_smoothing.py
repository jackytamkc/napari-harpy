from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QSpinBox, QVBoxLayout, QWidget

from napari_harpy.widgets.shared_styles import CHECKBOX_STYLESHEET, build_input_control_stylesheet


class ModuleSmoothingControl(QWidget):
    """UMAP smoothing controls shared by labels and shapes gene modules."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.checkbox = QCheckBox("UMAP neighbours")
        self.checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.checkbox.setChecked(True)
        self.neighbors_spinbox = QSpinBox()
        self.neighbors_spinbox.setRange(2, 1000)
        self.neighbors_spinbox.setValue(100)
        self.neighbors_spinbox.setPrefix("k=")
        self.neighbors_spinbox.setStyleSheet(build_input_control_stylesheet("QSpinBox"))
        self.setToolTip(
            "Average each selected gene over k nearest cells in this table's X_umap (including self), "
            "then compute the mean per-gene z-score across cells with UMAP coordinates. "
            "Disable this when using an already smoothed expression layer."
        )
        layout.addWidget(self.checkbox)
        layout.addWidget(self.neighbors_spinbox)
        self.checkbox.toggled.connect(self._on_toggled)
        self.neighbors_spinbox.valueChanged.connect(self._on_value_changed)

    @property
    def n_neighbors(self) -> int | None:
        return self.neighbors_spinbox.value() if self.checkbox.isChecked() else None

    def _on_toggled(self, checked: bool) -> None:
        self.neighbors_spinbox.setEnabled(checked)
        self.changed.emit()

    def _on_value_changed(self, _value: int) -> None:
        self.changed.emit()
