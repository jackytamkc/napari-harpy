"""Save and restore napari-harpy viewer *presets* keyed by image channel panel.

A **preset** captures how a SpatialData object is currently displayed so the same
look can be restored later without re-adjusting the viewer by hand:

- per image-channel layer: ``colormap``, ``contrast_limits``, ``gamma``,
  ``blending``, ``opacity`` and ``visible`` (both ``stack`` and ``overlay`` modes);
- per labels / shapes layer: ``opacity``, ``blending``, ``visible`` and the
  table/shape color source it is styled by (an
  :class:`~napari_harpy.core._color_source.TableColorSourceSpec` or
  :class:`~napari_harpy.core._color_source.ShapeColumnColorSourceSpec`);
- per categorical color source, the per-category palette stored in
  ``table.uns[f"{column}_colors"]``, so custom category colours survive;
- best effort per points value-selection layer: the index column and selected
  values, so a gene selection can be rebuilt;
- the active coordinate system, the scale bar and the camera.

Presets are matched to a :class:`~spatialdata.SpatialData` object by **channel
panel** -- the full set of image channel names. Re-opening the *same* sample, or
any other sample acquired with the *same* channel panel, therefore restores the
same display automatically. That cross-sample reuse is what makes an
open-and-go batch figure workflow possible: ``P2_MASLD1`` and ``P2_MASLD2``
share a panel, so they share a preset.

Storage
-------
One JSON file per preset under ``~/.napari-harpy/viewer_presets`` (override with
the ``NAPARI_HARPY_PRESET_DIR`` environment variable). The default preset name is
derived deterministically from the channel panel, so saving again for the same
panel overwrites the existing preset instead of piling up duplicates.

This is the harpy-native equivalent of napari-spatialdata's viewer presets. It is
built on :class:`~napari_harpy._app_state.HarpyAppState` and
:class:`~napari_harpy.viewer.adapter.ViewerAdapter` rather than reaching through
the Qt widgets, so it works headless and in the batch-render workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from napari.layers import Image, Points, Shapes

from napari_harpy.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from napari_harpy.core.spatialdata import get_image_channel_names_from_sdata
from napari_harpy.viewer.category_colors import category_values, get_category_palette, store_category_colors

if TYPE_CHECKING:
    from napari.layers import Layer
    from spatialdata import SpatialData

    from napari_harpy._app_state import HarpyAppState

__all__ = ["ViewerConfigManager", "get_preset_dir"]

CONFIG_VERSION = 1
ENV_PRESET_DIR = "NAPARI_HARPY_PRESET_DIR"
SAVE_KEY = "Shift-S"
APPLY_KEY = "Shift-D"

# Scale-bar attributes round-tripped. Kept to the ones a user typically sets and
# that accept plain JSON-serialisable values.
_SCALE_BAR_ATTRS = ("visible", "unit", "position", "colored", "font_size")


def get_preset_dir() -> Path:
    """Return the directory holding viewer presets, creating it if needed.

    Honours the ``NAPARI_HARPY_PRESET_DIR`` environment variable so tests and
    users who want project-local presets can redirect storage.
    """
    raw = os.environ.get(ENV_PRESET_DIR)
    base = Path(raw).expanduser() if raw else Path.home() / ".napari-harpy" / "viewer_presets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _enum_value(value: Any) -> Any:
    """Coerce a napari string-enum (e.g. scale-bar ``position``) to its raw value."""
    return value.value if hasattr(value, "value") else value


def _sanitize(name: str) -> str:
    """Make a preset name safe to use as a file stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "preset"


def _panel_name(panel: list[str]) -> str:
    """Deterministic default preset name for a channel panel.

    A stable hash keeps re-saves of the same panel pointed at one file, so the
    user only ever curates a single preset per channel panel unless they pick an
    explicit name.
    """
    if not panel:
        return "preset_default"
    digest = hashlib.sha1("\x1f".join(panel).encode("utf-8")).hexdigest()[:10]
    return f"panel_{digest}"


def _all_finite(values: Any) -> bool:
    """Return whether every scalar is finite, over a flat or nested-list/tuple of scalars.

    Callers pass Python floats (camera fields are ``float()``-ed first); this is not
    meant for numpy arrays, where ``not np.isfinite(array)`` would be ambiguous.
    """
    for value in values:
        if isinstance(value, list | tuple):
            if not _all_finite(value):
                return False
        elif not np.isfinite(value):
            return False
    return True


class ViewerConfigManager:
    """Capture, persist and re-apply viewer presets for one :class:`HarpyAppState`.

    The manager reaches the viewer, the active :class:`~spatialdata.SpatialData`
    object, the active coordinate system and the layer registry through the app
    state, so the apply path re-creates layers exactly the way interactive
    loading does -- via the shared :class:`~napari_harpy.viewer.adapter.ViewerAdapter`.
    """

    def __init__(self, app_state: HarpyAppState) -> None:
        self._app_state = app_state

    @property
    def _viewer(self) -> Any:
        return self._app_state.viewer

    @property
    def _adapter(self) -> Any:
        return self._app_state.viewer_adapter

    # -- key bindings ----------------------------------------------------------

    def bind_keys(self) -> None:
        """Bind the save/apply shortcuts on the napari viewer, if one is present."""
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "bind_key"):
            return
        viewer.bind_key(SAVE_KEY, self._on_save_key, overwrite=True)
        viewer.bind_key(APPLY_KEY, self._on_apply_key, overwrite=True)

    def _on_save_key(self, _viewer: Any) -> None:
        try:
            self.save()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive, keeps napari alive
            logger.exception("Failed to save viewer preset")
            _notify(f"Could not save viewer preset: {exc}")

    def _on_apply_key(self, _viewer: Any) -> None:
        try:
            if not self.apply():
                _notify("No viewer preset matches the current channel panel.")
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive, keeps napari alive
            logger.exception("Failed to apply viewer preset")
            _notify(f"Could not apply viewer preset: {exc}")

    # -- capture ---------------------------------------------------------------

    def capture(self, name: str | None = None) -> dict[str, Any]:
        """Serialise the current viewer state into a preset dictionary."""
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        layers_cfg: list[dict[str, Any]] = []
        shown_channels: set[str] = set()
        panel: set[str] = set()

        for binding in self._bindings_for_active_sdata():
            entry = self._capture_binding(binding)
            if entry is None:
                continue
            layers_cfg.append(entry)
            if entry["kind"] == "image":
                shown_channels.update(entry.get("channels_shown") or [])
                panel.update(self._element_channels(sdata, entry["element"]))

        panel_signature = sorted(panel)
        return {
            "version": CONFIG_VERSION,
            "name": name or _panel_name(panel_signature),
            "channel_signature": sorted(shown_channels),
            "panel_signature": panel_signature,
            "coordinate_system": coordinate_system,
            "scale_bar": self._capture_scale_bar(),
            "camera": self._capture_camera(),
            "palettes": self._capture_palettes(sdata, layers_cfg),
            "layers": layers_cfg,
            "layer_order": self._capture_layer_order(),
        }

    def _capture_layer_order(self) -> list[str]:
        """Ordered layer names (bottom->top) for the active sdata's bound layers.

        napari stacks ``viewer.layers`` bottom (index 0) to top (last index), so a
        shapes layer the user dragged above the images sits at the end here. We store
        the names in that order and replay it on apply — otherwise ``_plan_layers``
        rebuilds images last (on top) and buries the styled shapes/labels beneath them.
        """
        viewer = self._viewer
        active_layer_ids = {id(binding.layer) for binding in self._bindings_for_active_sdata()}
        return [layer.name for layer in getattr(viewer, "layers", []) if id(layer) in active_layer_ids]

    def _bindings_for_active_sdata(self) -> list[Any]:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if sdata is None:
            return []
        bindings = []
        for binding in self._app_state.layer_bindings.iter_bindings():
            if binding.sdata_id != id(sdata):
                continue
            if coordinate_system is not None and binding.coordinate_system != coordinate_system:
                continue
            bindings.append(binding)
        return bindings

    def _capture_binding(self, binding: Any) -> dict[str, Any] | None:
        layer = binding.layer
        base = {
            "element": binding.element_name,
            "coordinate_system": binding.coordinate_system,
            "visible": bool(layer.visible),
            "opacity": float(layer.opacity),
            "blending": str(_enum_value(layer.blending)),
        }
        element_type = binding.element_type
        if element_type == "image":
            return {**base, **self._capture_image(binding, layer)}
        if element_type == "labels":
            return {**base, **self._capture_styled(binding, kind="labels")}
        if element_type == "shapes":
            return {**base, **self._capture_styled(binding, kind="shapes")}
        if element_type == "points":
            return {**base, **self._capture_points(binding, layer)}
        return None

    def _capture_image(self, binding: Any, layer: Image) -> dict[str, Any]:
        colormap = layer.colormap
        contrast = [float(x) for x in layer.contrast_limits]
        gamma = float(layer.gamma)
        mode = binding.image_display_mode or "stack"
        channel_name = binding.channel_name
        if mode == "overlay":
            shown = [channel_name] if channel_name is not None else []
        else:
            # A stacked image shows its whole channel panel in one layer. Sort for a
            # reproducible preset file (the channel set is otherwise unordered).
            shown = sorted(self._element_channels(self._app_state.sdata, binding.element_name))
        return {
            "kind": "image",
            "mode": mode,
            "channel": channel_name,
            "channels_shown": shown,
            "colormap": getattr(colormap, "name", None),
            "colormap_colors": np.asarray(colormap.colors, dtype=float).tolist(),
            # Non-finite intensities (e.g. an all-NaN channel) would serialise as
            # bare NaN tokens -- invalid JSON -- so drop them and skip on restore.
            "contrast_limits": contrast if all(np.isfinite(contrast)) else None,
            "gamma": gamma if np.isfinite(gamma) else None,
        }

    def _capture_styled(self, binding: Any, *, kind: str) -> dict[str, Any]:
        role = getattr(binding, f"{kind}_role", "primary")
        entry: dict[str, Any] = {"kind": kind, "role": role}
        if kind == "shapes":
            entry["fill"] = binding.layer.face_color.shape[0] > 0 and _shapes_layer_is_filled(binding.layer)
        if role == "styled" and binding.style_spec is not None:
            entry["style"] = _serialize_style_spec(binding.style_spec)
        return entry

    def _capture_points(self, binding: Any, layer: Points) -> dict[str, Any]:
        index_column = binding.index_column
        selected_values = _selected_point_values(layer, index_column)
        size = _scalar_point_size(layer)
        return {
            "kind": "points",
            "index_column": index_column,
            "selected_values": selected_values,
            "size": size,
        }

    def _element_channels(self, sdata: SpatialData | None, element_name: str) -> set[str]:
        if sdata is None or element_name not in getattr(sdata, "images", {}):
            return set()
        try:
            return {str(c) for c in get_image_channel_names_from_sdata(sdata, element_name)}
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - malformed element
            logger.debug(f"Could not read channels for image {element_name!r}: {exc}")
            return set()

    def _capture_palettes(self, sdata: SpatialData | None, layers_cfg: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Capture the per-category palette for every categorical table color source used.

        Returned as a list of ``{table, column, categories, colors}`` records --
        categories are stringified for JSON and re-paired with the target table's
        real (possibly non-string) categories on restore.
        """
        if sdata is None:
            return []
        palettes: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for entry in layers_cfg:
            style = entry.get("style")
            if not style or style.get("type") != "table" or style.get("value_kind") != "categorical":
                continue
            if style.get("source_kind") != "obs_column":
                continue
            table_name = style["table_name"]
            column = style["value_key"]
            if (table_name, column) in seen:
                continue
            seen.add((table_name, column))
            table = getattr(sdata, "tables", {}).get(table_name)
            if table is None:
                continue
            try:
                categories, hex_colors = get_category_palette(table, column)
            except (KeyError, ValueError) as exc:
                logger.debug(f"Could not capture palette for {table_name}/{column}: {exc}")
                continue
            palettes.append(
                {
                    "table": table_name,
                    "column": column,
                    "categories": [str(c) for c in categories],
                    "colors": list(hex_colors),
                }
            )
        return palettes

    def _capture_camera(self) -> dict[str, Any]:
        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            return {}
        try:
            camera_state = {
                "center": [float(x) for x in camera.center],
                "zoom": float(camera.zoom),
                "angles": [float(x) for x in camera.angles],
                "perspective": float(camera.perspective),
            }
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            logger.debug(f"Could not capture camera: {exc}")
            return {}
        # Drop the whole camera if any component is non-finite: a bare NaN/inf would
        # make the (allow_nan=False) save raise rather than degrade, and a camera is
        # cosmetic. `apply` already tolerates a missing camera.
        if not _all_finite(camera_state.values()):
            logger.debug("Skipping non-finite camera state in viewer preset.")
            return {}
        return camera_state

    def _capture_scale_bar(self) -> dict[str, Any]:
        scale_bar = getattr(self._viewer, "scale_bar", None)
        if scale_bar is None:
            return {}
        out: dict[str, Any] = {}
        for attr in _SCALE_BAR_ATTRS:
            try:
                out[attr] = _enum_value(getattr(scale_bar, attr))
            except AttributeError:  # pragma: no cover - napari version drift
                continue
        return out

    # -- persistence -----------------------------------------------------------

    def save(self, name: str | None = None) -> Path | None:
        """Capture the current viewer and write it to a preset file.

        Returns the written path, or ``None`` when there is nothing worth saving
        (no harpy-managed layers in the viewer yet).
        """
        cfg = self.capture(name=name)
        if not cfg["layers"]:
            _notify("Nothing to save: add SpatialData elements to the viewer first.")
            return None
        path = get_preset_dir() / f"{_sanitize(cfg['name'])}.json"
        # allow_nan=False is a safety net: capture nulls/drops non-finite contrast,
        # gamma and camera values, so a NaN reaching here means a new code path needs
        # the same treatment -- fail loudly rather than write invalid JSON.
        path.write_text(json.dumps(cfg, indent=2, allow_nan=False))
        panel = ", ".join(cfg["panel_signature"]) or "no channels"
        _notify(f"Saved viewer preset '{cfg['name']}' ({len(cfg['layers'])} layers; {panel}).")
        logger.info(f"Saved viewer preset to {path}")
        return path

    @staticmethod
    def list_presets() -> list[tuple[Path, dict[str, Any]]]:
        """Return ``(path, preset)`` for every readable JSON-object preset on disk.

        Unreadable files and valid JSON that is not an object are skipped with a
        warning, so one bad file can never break matching, autoload or listing.
        """
        presets: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(get_preset_dir().glob("*.json")):
            try:
                cfg = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                # ValueError covers both json.JSONDecodeError and the UnicodeDecodeError
                # a non-UTF-8 file raises -- one corrupt file must not disable the
                # whole preset system (match/autoload/list all iterate this).
                logger.warning(f"Skipping unreadable preset {path}: {exc}")
                continue
            # Valid JSON that is not an object (e.g. `[1,2]`, `42`) would blow up the
            # later `cfg.get(...)` calls; drop it here so one bad file cannot break the set.
            if not isinstance(cfg, dict):
                logger.warning(f"Skipping preset {path}: expected a JSON object, got {type(cfg).__name__}.")
                continue
            presets.append((path, cfg))
        return presets

    def _load_named(self, name: str) -> dict[str, Any]:
        path = get_preset_dir() / f"{_sanitize(name)}.json"
        if path.exists():
            try:
                cfg = json.loads(path.read_text())
            except (OSError, ValueError):
                cfg = None
            # A corrupt or non-object file falls through to the display-name scan
            # (which drops such files) rather than returning a value `apply` can't use.
            if isinstance(cfg, dict):
                return cfg
        for _candidate_path, cfg in self.list_presets():
            if cfg.get("name") == name:
                return cfg
        raise FileNotFoundError(f"No usable viewer preset named {name!r} in {get_preset_dir()}")

    # -- matching --------------------------------------------------------------

    def _sdata_image_channel_sets(self) -> list[set[str]]:
        sdata = self._app_state.sdata
        if sdata is None:
            return []
        sets: list[set[str]] = []
        for element_name in getattr(sdata, "images", {}):
            channels = self._element_channels(sdata, element_name)
            if channels:
                sets.append(channels)
        return sets

    def match_preset(self) -> dict[str, Any] | None:
        """Return the newest preset whose channel panel the active data provides.

        A preset matches when its full ``panel_signature`` is a subset of some
        image element's channels in the active data -- so a different sample
        acquired with the same panel matches, while an image that merely shares a
        stray channel does not. Ties are broken by most-recently-modified file.
        """
        target_panels = self._sdata_image_channel_sets()
        best: dict[str, Any] | None = None
        best_mtime = -1.0
        for path, cfg in self.list_presets():
            if not _preset_matches(cfg, target_panels):
                continue
            mtime = path.stat().st_mtime
            if mtime > best_mtime:
                best, best_mtime = cfg, mtime
        return best

    # -- apply -----------------------------------------------------------------

    def autoload(self) -> bool:
        """Apply the preset matching the active channel panel, if any."""
        cfg = self.match_preset()
        if cfg is None:
            logger.debug("No matching viewer preset for the active channel panel.")
            return False
        logger.info(f"Auto-loading viewer preset '{cfg.get('name')}'.")
        return self.apply(cfg)

    def apply_named(self, name: str) -> bool:
        """Apply a preset selected explicitly by name."""
        return self.apply(self._load_named(name))

    def apply(self, cfg: dict[str, Any] | None = None) -> bool:
        """Apply ``cfg`` (or the auto-matched preset) to the viewer.

        Returns ``True`` when at least one layer was applied, ``False`` when no
        preset matched or every layer failed to apply (e.g. a preset from an
        unrelated sample whose elements are all absent).
        """
        if cfg is None:
            cfg = self.match_preset()
        # None (no match) or a non-dict handed to the public `apply(cfg)` both mean
        # "nothing usable to apply" rather than an error.
        if not isinstance(cfg, dict):
            return False
        sdata = self._app_state.sdata
        if sdata is None:
            _notify("No SpatialData loaded; cannot apply viewer preset.")
            return False

        coordinate_system = self._select_coordinate_system(cfg.get("coordinate_system"))
        if coordinate_system is None:
            _notify("No coordinate system available; cannot apply viewer preset.")
            return False

        # Restore category palettes before styling so the stored-palette path wins.
        self._restore_palettes(sdata, cfg)

        applied = 0
        skipped = 0
        for entry, target in self._plan_layers(cfg, sdata):
            try:
                self._apply_layer(entry, target, sdata, coordinate_system)
                applied += 1
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.warning(f"Skipping preset layer {entry.get('element')!r}: {exc}")

        self._restore_layer_order(cfg.get("layer_order"))
        self._apply_scale_bar(cfg.get("scale_bar"))
        self._apply_camera(cfg.get("camera"))
        message = f"Applied viewer preset '{cfg.get('name')}' ({applied} layers"
        message += f"; {skipped} skipped)." if skipped else ")."
        _notify(message)
        # A preset with layers that all failed to load (wrong sample) is not "applied".
        return applied > 0 or not cfg.get("layers")

    def _restore_layer_order(self, order: list[str] | None) -> None:
        """Re-stack ``viewer.layers`` so the named layers follow the captured order.

        ``order`` is bottom->top. On a fresh autoload the layers named in the preset are
        moved into that relative order at the bottom of the stack; anything not named
        (e.g. a layer added to a live viewer *after* the preset was saved) is left
        floating above them, stably. Old presets predate this field and pass ``None``;
        a malformed field is likewise ignored -- like every sibling restore step here, a
        bad value degrades to a no-op rather than aborting ``apply()`` mid-way.
        """
        if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
            return
        rank = {name: index for index, name in enumerate(order)}
        if not rank:
            return
        viewer = self._viewer
        ordered = sorted(
            (layer for layer in getattr(viewer, "layers", []) if layer.name in rank),
            key=lambda layer: rank[layer.name],
        )
        for dest, layer in enumerate(ordered):
            try:
                src = viewer.layers.index(layer)
                if src != dest:
                    viewer.layers.move(src, dest)
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not reorder layer {layer.name!r}: {exc}")

    def _select_coordinate_system(self, preferred: str | None) -> str | None:
        sdata = self._app_state.sdata
        from napari_harpy.core.spatialdata import get_coordinate_system_names_from_sdata

        available = get_coordinate_system_names_from_sdata(sdata) if sdata is not None else []
        if not available:
            return None
        target = preferred if preferred in available else available[0]
        if preferred is not None and preferred not in available:
            _notify(f"Coordinate system '{preferred}' not available; applying preset in '{target}'.")
        self._app_state.set_coordinate_system(target, source="preset_apply")
        return target

    def _plan_layers(self, cfg: dict[str, Any], sdata: SpatialData) -> list[tuple[dict[str, Any], Any]]:
        """Pair each preset layer with the target element/channels to load it into.

        Overlay image channels of the same element are coalesced into one entry so
        they are loaded in a single ``ensure_image_loaded`` call (loading them one
        at a time would tear down the previously loaded channels).
        """
        panel_signature = set(cfg.get("panel_signature") or [])
        plan: list[tuple[dict[str, Any], Any]] = []
        overlay_groups: dict[str, dict[str, Any]] = {}
        for entry in cfg.get("layers", []):
            kind = entry.get("kind")
            if kind == "image" and entry.get("mode") == "overlay":
                element = self._resolve_image_element(sdata, entry.get("element"), panel_signature)
                if element is None:
                    logger.warning(f"No image element for overlay channel {entry.get('channel')!r}; skipping.")
                    continue
                group = overlay_groups.setdefault(
                    element, {"kind": "image_overlay", "element": element, "channels": []}
                )
                group["channels"].append(entry)
            elif kind == "image":
                element = self._resolve_image_element(sdata, entry.get("element"), panel_signature)
                if element is not None:
                    plan.append((entry, element))
            else:
                plan.append((entry, entry.get("element")))
        for element, group in overlay_groups.items():
            plan.append((group, element))
        return plan

    def _resolve_image_element(
        self, sdata: SpatialData, preset_element: str | None, panel_signature: set[str]
    ) -> str | None:
        images = getattr(sdata, "images", {})
        if not images:
            return None
        # Prefer the same-named element when it hosts the panel (same sample).
        if preset_element in images and panel_signature.issubset(self._element_channels(sdata, preset_element)):
            return preset_element
        # Otherwise any element whose channels cover the panel (a sibling sample).
        if panel_signature:
            for element_name in images:
                if panel_signature.issubset(self._element_channels(sdata, element_name)):
                    return element_name
            return preset_element if preset_element in images else None
        return preset_element if preset_element in images else next(iter(images))

    def _apply_layer(self, entry: dict[str, Any], target: Any, sdata: SpatialData, coordinate_system: str) -> None:
        kind = entry.get("kind")
        if kind == "image":
            self._apply_stack_image(entry, target, sdata, coordinate_system)
        elif kind == "image_overlay":
            self._apply_overlay_image(entry, target, sdata, coordinate_system)
        elif kind == "labels":
            self._apply_labels(entry, target, sdata, coordinate_system)
        elif kind == "shapes":
            self._apply_shapes(entry, target, sdata, coordinate_system)
        elif kind == "points":
            self._apply_points(entry, target, sdata, coordinate_system)

    def _apply_stack_image(self, entry: dict[str, Any], element: str, sdata: SpatialData, cs: str) -> None:
        result = self._adapter.ensure_image_loaded(sdata, element, cs, mode="stack")
        for layer in result.layers:
            _apply_image_props(layer, entry)

    def _apply_overlay_image(self, group: dict[str, Any], element: str, sdata: SpatialData, cs: str) -> None:
        # Resolve captured channel names to indices in THIS element. Names are the
        # cross-sample-stable key, but the adapter matches string names against the
        # element's raw channel coords, which can be integers; mapping to an index
        # via the stringified channel list works for both and picks the sibling's
        # own position when applying a preset to a different sample.
        available = [str(c) for c in get_image_channel_names_from_sdata(sdata, element)]
        entry_by_channel: dict[str, dict[str, Any]] = {}
        channel_indices: list[int] = []
        for entry in group["channels"]:
            channel_name = entry.get("channel")
            if channel_name in available:
                entry_by_channel[channel_name] = entry
                channel_indices.append(available.index(channel_name))
        if not channel_indices:
            raise ValueError(f"none of the preset channels are present in image element {element!r}")
        result = self._adapter.ensure_image_loaded(sdata, element, cs, mode="overlay", channels=channel_indices)
        for layer, channel_name in zip(result.layers, result.channel_names, strict=False):
            entry = entry_by_channel.get(str(channel_name))
            if entry is not None:
                _apply_image_props(layer, entry)

    def _apply_labels(self, entry: dict[str, Any], element: str, sdata: SpatialData, cs: str) -> None:
        if entry.get("role") == "styled" and entry.get("style"):
            spec = _deserialize_style_spec(entry["style"])
            result = self._adapter.ensure_styled_labels_loaded(sdata, element, cs, spec)
        else:
            result = self._adapter.ensure_labels_loaded(sdata, element, cs)
        _apply_common_props(result.layer, entry)

    def _apply_shapes(self, entry: dict[str, Any], element: str, sdata: SpatialData, cs: str) -> None:
        if entry.get("role") == "styled" and entry.get("style"):
            spec = _deserialize_style_spec(entry["style"])
            result = self._adapter.ensure_styled_shapes_loaded(sdata, element, cs, spec, fill=bool(entry.get("fill")))
        else:
            result = self._adapter.ensure_shapes_loaded(sdata, element, cs)
        _apply_common_props(result.layer, entry)

    def _apply_points(self, entry: dict[str, Any], element: str, sdata: SpatialData, cs: str) -> None:
        # Rebuilding a points layer runs a Dask scan; keep it best-effort so a
        # missing element or gene never aborts the rest of the preset.
        from napari_harpy._points_value_index import (
            DEFAULT_RANDOM_STATE,
            build_points_value_table,
            load_points,
            validate_points_element_for_value_selection,
        )
        from napari_harpy.viewer.adapter import PointsLayerIdentity

        index_column = entry.get("index_column")
        values = entry.get("selected_values") or []
        if element not in getattr(sdata, "points", {}) or not index_column or not values:
            return
        validated = validate_points_element_for_value_selection(sdata, element, index_column=index_column)
        value_table = build_points_value_table(validated)
        selection = load_points(validated, value_table, list(values), random_state=DEFAULT_RANDOM_STATE)
        identity = PointsLayerIdentity(
            sdata=sdata, points_name=element, coordinate_system=cs, index_column=index_column
        )
        result = self._adapter._ensure_points_layer_from_selection(identity, selection=selection)
        _apply_common_props(result.layer, entry)
        size = entry.get("size")
        if size is not None:
            try:
                result.layer.current_size = float(size)
                result.layer.size = float(size)
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not restore point size on {result.layer.name}: {exc}")

    def _restore_palettes(self, sdata: SpatialData, cfg: dict[str, Any]) -> None:
        """Write captured per-category palettes back into ``table.uns`` before styling.

        The palette is only restored when every one of the target column's categories
        is present in the captured record, otherwise the column is left to harpy's
        default palette rather than half-applied. Category matching is by *string* --
        the target table's categories may be non-string (integer cluster ids, bools),
        so the colour map is keyed by the table's own category objects (which is what
        ``store_category_colors`` looks them up by) while lookup into the stored,
        stringified record goes through ``str(category)``.

        This writes ``table.uns`` exactly as :class:`CategoryColorEditor` does, and --
        like the editor -- deliberately does not mark the table dirty: the palette is
        a display preference, not user data, so re-opening a sample must not report an
        unsynced change the user never made.
        """
        tables = getattr(sdata, "tables", {})
        palettes = cfg.get("palettes")
        if not isinstance(palettes, list):  # tolerate a legacy/malformed shape
            return
        for record in palettes:
            if not isinstance(record, dict):
                continue
            table_name = record.get("table")
            column = record.get("column")
            table = tables.get(table_name)
            if table is None or not column:
                continue
            color_by_stringified = dict(zip(record.get("categories", []), record.get("colors", []), strict=False))
            try:
                current = category_values(table, column)  # the table's own category objects
            except (KeyError, ValueError):
                continue
            if not current or any(str(category) not in color_by_stringified for category in current):
                continue
            try:
                store_category_colors(
                    table, column, {category: color_by_stringified[str(category)] for category in current}
                )
            except ValueError as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not restore palette for {table_name}/{column}: {exc}")

    def _apply_scale_bar(self, cfg: dict[str, Any] | None) -> None:
        scale_bar = getattr(self._viewer, "scale_bar", None)
        if not cfg or scale_bar is None:
            return
        for attr, value in cfg.items():
            try:
                setattr(scale_bar, attr, value)
            except (ValueError, TypeError, AttributeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not set scale_bar.{attr}: {exc}")

    def _apply_camera(self, cam: dict[str, Any] | None) -> None:
        camera = getattr(self._viewer, "camera", None)
        if not cam or camera is None:
            return
        try:
            center = cam.get("center")
            # Only restore the centre when its dimensionality matches the current
            # display (a 2D/3D mismatch would raise); skip rather than fail.
            if center is not None and len(center) == len(camera.center):
                camera.center = tuple(center)
            if cam.get("angles") is not None:
                camera.angles = tuple(cam["angles"])
            if cam.get("perspective") is not None:
                camera.perspective = float(cam["perspective"])
            # zoom last: its setter derives the view rectangle from the centre.
            if cam.get("zoom") is not None:
                camera.zoom = float(cam["zoom"])
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - cosmetic; must never abort apply
            logger.debug(f"Could not restore camera: {exc}")


# -- module-level helpers ------------------------------------------------------


def _notify(message: str) -> None:
    """Show a napari notification, degrading to a log line when headless."""
    try:
        from napari.utils.notifications import show_info

        show_info(message)
    except Exception:  # noqa: BLE001  # pragma: no cover - headless / no notification manager
        logger.info(message)


def _preset_matches(cfg: dict[str, Any], target_panels: list[set[str]]) -> bool:
    panel = set(cfg.get("panel_signature") or [])
    if panel:
        return any(panel.issubset(channels) for channels in target_panels)
    # Legacy / channel-less presets: fall back to the shown-channel subset test.
    shown = set(cfg.get("channel_signature") or [])
    return bool(shown) and any(shown.issubset(channels) for channels in target_panels)


def _serialize_style_spec(spec: TableColorSourceSpec | ShapeColumnColorSourceSpec) -> dict[str, Any]:
    if isinstance(spec, TableColorSourceSpec):
        return {
            "type": "table",
            "table_name": spec.table_name,
            "source_kind": spec.source_kind,
            "value_key": spec.value_key,
            "value_kind": spec.value_kind,
            "layer": spec.layer,
            "smoother": spec.smoother,
            "colormap": spec.colormap,
            "module_genes": list(spec.module_genes),
            "umap_neighbors": spec.umap_neighbors,
        }
    return {
        "type": "shape",
        "source_kind": spec.source_kind,
        "value_key": spec.value_key,
        "value_kind": spec.value_kind,
        "colormap": spec.colormap,
    }


def _deserialize_style_spec(data: dict[str, Any]) -> TableColorSourceSpec | ShapeColumnColorSourceSpec:
    if data.get("type") == "shape":
        return ShapeColumnColorSourceSpec(
            source_kind=data["source_kind"],
            value_key=data["value_key"],
            value_kind=data["value_kind"],
            colormap=data.get("colormap"),
        )
    return TableColorSourceSpec(
        table_name=data["table_name"],
        source_kind=data["source_kind"],
        value_key=data["value_key"],
        value_kind=data["value_kind"],
        layer=data.get("layer"),
        smoother=data.get("smoother"),
        colormap=data.get("colormap"),
        module_genes=tuple(data.get("module_genes") or ()),
        umap_neighbors=data.get("umap_neighbors"),
    )


def _apply_common_props(layer: Layer, entry: dict[str, Any]) -> None:
    for attr in ("opacity", "blending", "visible"):
        value = entry.get(attr)
        if value is not None:
            try:
                setattr(layer, attr, value)
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not set {attr} on {layer.name}: {exc}")


def _apply_image_props(layer: Image, entry: dict[str, Any]) -> None:
    contrast = entry.get("contrast_limits")
    if contrast and len(contrast) == 2:
        # napari requires monotonically increasing limits; a hand-edited or
        # cross-sample preset could carry them inverted or degenerate.
        low, high = sorted((float(contrast[0]), float(contrast[1])))
        if high > low:
            try:
                current = list(layer.contrast_limits_range)
                layer.contrast_limits_range = [min(current[0], low), max(current[1], high)]
                layer.contrast_limits = [low, high]
            except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
                logger.debug(f"Could not set contrast_limits on {layer.name}: {exc}")

    cmap_name = entry.get("colormap")
    if cmap_name is not None:
        try:
            layer.colormap = cmap_name
        except (KeyError, ValueError, TypeError):
            colors = entry.get("colormap_colors")
            if colors:
                from napari.utils.colormaps import Colormap

                layer.colormap = Colormap(colors=np.asarray(colors, dtype=float), name=str(cmap_name))

    gamma = entry.get("gamma")
    if gamma is not None:
        try:
            layer.gamma = float(gamma)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            logger.debug(f"Could not set gamma on {layer.name}: {exc}")

    _apply_common_props(layer, entry)


def _shapes_layer_is_filled(layer: Shapes) -> bool:
    try:
        return bool(np.any(np.asarray(layer.face_color)[:, 3] > 0.0))
    except (IndexError, ValueError, TypeError):  # pragma: no cover - defensive
        return False


def _selected_point_values(layer: Points, index_column: str) -> list[str]:
    """Recover the selected index-column values from a points value-selection layer."""
    features = getattr(layer, "features", None)
    if features is None or index_column not in getattr(features, "columns", []):
        return []
    series = features[index_column]
    categories = getattr(getattr(series, "cat", None), "categories", None)
    if categories is not None:
        return [str(c) for c in categories]
    return [str(v) for v in dict.fromkeys(series.tolist())]


def _scalar_point_size(layer: Points) -> float | None:
    try:
        sizes = np.atleast_1d(np.asarray(layer.size, dtype=float)).ravel()
        return float(sizes[0]) if sizes.size else None
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return None
