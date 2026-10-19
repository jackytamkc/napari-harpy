from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import napari
from loguru import logger

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy.viewer.viewer_config import ViewerConfigManager

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata import SpatialData

HarpyWidgetId: TypeAlias = Literal["viewer", "feature_extraction", "object_classification", "shapes_annotation"]
HarpyWidgetSelection: TypeAlias = Literal["all"] | HarpyWidgetId | Sequence[HarpyWidgetId]


class Interactive:
    """
    Thin programmatic launcher for napari-harpy.

    Parameters
    ----------
    sdata
        SpatialData object that Harpy widgets should use as their shared active
        data source.
    viewer
        Existing napari viewer to reuse. If omitted, the current viewer is used
        or a new viewer is created.
    headless
        If True, initialize state and dock widgets without starting the napari
        event loop.
    widgets
        Which Harpy dock widgets to open. Defaults to ``"all"``. Possible
        values are ``"all"``, ``"viewer"``, ``"feature_extraction"``,
        ``"object_classification"``, and ``"shapes_annotation"``. Pass a tuple
        of widget ids to open a subset. ``"all"`` opens every Harpy widget.
    async_slicing
        If ``True`` or ``False``, explicitly enable or disable napari's
        experimental async slicing for this session. If ``None``, leave napari's
        current setting unchanged.
    autoload
        If ``True`` (the default), apply the saved viewer preset whose image
        channel panel matches ``sdata`` right after the widgets are docked, so a
        sample re-opens with the display you last saved. See
        :meth:`save_config` and :mod:`napari_harpy.viewer.viewer_config`.

    Notes
    -----
    Viewer presets capture how a sample is displayed (image channel colormaps and
    contrast, layer styling, category palettes, scale bar, coordinate system and
    camera) and restore it for the same sample -- or any sample sharing the same
    channel panel -- without re-tuning by hand. Press ``Shift-S`` to save the
    current display and ``Shift-D`` to re-apply it, or use :meth:`save_config` /
    :meth:`load_config` programmatically.
    """

    _PLUGIN_NAME = "napari-harpy"
    _WIDGET_NAMES: dict[str, str] = {
        "viewer": "Viewer",
        "feature_extraction": "Feature Extraction",
        "object_classification": "Object Classification",
        "shapes_annotation": "Annotation",
    }
    _ALL_WIDGET_IDS: tuple[HarpyWidgetId, ...] = (
        "viewer",
        "feature_extraction",
        "object_classification",
        "shapes_annotation",
    )

    def __init__(
        self,
        sdata: SpatialData,
        viewer: napari.Viewer | None = None,
        headless: bool = False,
        widgets: HarpyWidgetSelection = "all",
        async_slicing: bool | None = False,
        autoload: bool = True,
    ) -> None:
        widget_ids = self._normalize_widget_selection(widgets)
        if async_slicing is not None:
            _set_napari_async_slicing(async_slicing)
        self._viewer = viewer or napari.current_viewer() or napari.Viewer()
        self._app_state = get_or_create_app_state(self._viewer)
        self._dock_widgets: dict[str, tuple[Any, Any]] = {}
        self._config_manager = ViewerConfigManager(self._app_state)
        self._config_manager.bind_keys()

        self._app_state.set_sdata(sdata)
        self._ensure_harpy_widgets(widget_ids)

        if autoload:
            try:
                self._config_manager.autoload()
            except Exception:  # noqa: BLE001  # pragma: no cover - never block startup on a bad preset
                logger.exception("Failed to auto-load viewer preset")

        if not headless:
            self.run()

    @property
    def viewer(self) -> napari.Viewer:
        """Return the napari viewer managed by the launcher."""
        return self._viewer

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for the active viewer."""
        return self._app_state

    def save_config(self, name: str | None = None) -> Path | None:
        """Save the current viewer display as a preset (same as pressing ``Shift-S``).

        Parameters
        ----------
        name
            Optional preset name. When omitted, a deterministic name derived from
            the image channel panel is used, so re-saving the same panel
            overwrites the existing preset rather than creating a duplicate.

        Returns
        -------
        Path | None
            The path the preset was written to, or ``None`` when there were no
            harpy-managed layers in the viewer to save.
        """
        return self._config_manager.save(name)

    def load_config(self, name: str | None = None) -> bool:
        """Apply a saved viewer preset (same as pressing ``Shift-D``).

        Parameters
        ----------
        name
            Preset name to apply. When omitted, the preset whose channel panel
            matches the loaded data is applied.

        Returns
        -------
        bool
            ``True`` if a preset was applied, ``False`` if none matched.
        """
        if name is not None:
            return self._config_manager.apply_named(name)
        return self._config_manager.apply()

    def list_configs(self) -> list[str]:
        """Return the names of all saved viewer presets."""
        return [cfg.get("name", path.stem) for path, cfg in self._config_manager.list_presets()]

    def run(self) -> None:
        """Run the napari application."""
        napari.run()

    def _ensure_harpy_widgets(self, widget_ids: Sequence[HarpyWidgetId]) -> None:
        for widget_id in widget_ids:
            widget_name = self._WIDGET_NAMES[widget_id]
            self._dock_widgets[widget_name] = self._viewer.window.add_plugin_dock_widget(
                self._PLUGIN_NAME,
                widget_name,
                tabify=True,
            )

    @classmethod
    def _normalize_widget_selection(cls, widgets: HarpyWidgetSelection) -> tuple[HarpyWidgetId, ...]:
        if widgets == "all":
            return cls._ALL_WIDGET_IDS

        if isinstance(widgets, str):
            cls._validate_widget_id(widgets)
            return (widgets,)

        if not isinstance(widgets, Sequence):
            raise ValueError("`widgets` must be 'all', one Harpy widget id, or a sequence of Harpy widget ids.")

        widget_ids: list[HarpyWidgetId] = []
        seen_widget_ids: set[HarpyWidgetId] = set()
        for widget_id in widgets:
            cls._validate_widget_id(widget_id)
            if widget_id not in seen_widget_ids:
                widget_ids.append(widget_id)
                seen_widget_ids.add(widget_id)

        return tuple(widget_ids)

    @classmethod
    def _validate_widget_id(cls, widget_id: object) -> None:
        if widget_id not in cls._WIDGET_NAMES:
            valid_widget_ids = ", ".join(("all", *cls._WIDGET_NAMES))
            raise ValueError(f"Unknown Harpy widget selection {widget_id!r}. Valid options are: {valid_widget_ids}.")


def _set_napari_async_slicing(enabled: bool) -> None:
    from napari.settings import get_settings

    get_settings().experimental.async_ = enabled
