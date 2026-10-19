from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from napari.layers import Image, Points
from spatialdata.datasets import blobs

from napari_harpy import Interactive
from napari_harpy.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from napari_harpy.viewer.category_colors import store_category_colors, uns_colors_key
from napari_harpy.viewer.viewer_config import (
    ViewerConfigManager,
    _all_finite,
    _apply_image_props,
    _deserialize_style_spec,
    _panel_name,
    _preset_matches,
    _sanitize,
    _selected_point_values,
    _serialize_style_spec,
    get_preset_dir,
)

TABLE_SPEC = TableColorSourceSpec(
    table_name="table", source_kind="obs_column", value_key="region", value_kind="categorical"
)
CLUSTER_SPEC = TableColorSourceSpec(
    table_name="table", source_kind="obs_column", value_key="cluster", value_kind="categorical"
)
RADIUS_SPEC = ShapeColumnColorSourceSpec(source_kind="shape_column", value_key="radius", value_kind="continuous")


def _add_int_cluster(sdata, n_categories=3):
    """Add an integer-categorical 'cluster' obs column (categories are ints, not strings)."""
    table = sdata.tables["table"]
    table.obs["cluster"] = pd.Categorical(np.arange(table.n_obs) % n_categories)
    return sdata


@pytest.fixture(autouse=True)
def _preset_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NAPARI_HARPY_PRESET_DIR", str(tmp_path / "presets"))
    return tmp_path


def _interactive(make_napari_viewer, sdata, *, autoload=False):
    viewer = make_napari_viewer(show=False)
    inter = Interactive(sdata, viewer=viewer, headless=True, autoload=autoload)
    inter.app_state.set_coordinate_system("global")
    return inter


def _load_overlay(inter, sdata):
    result = inter.app_state.viewer_adapter.ensure_image_loaded(
        sdata, "blobs_image", "global", mode="overlay", channels=[0, 2], channel_colors=["blue", "red"]
    )
    return result.layers


# -- pure helpers --------------------------------------------------------------


def test_get_preset_dir_honours_env(tmp_path, monkeypatch):
    target = tmp_path / "custom_presets"
    monkeypatch.setenv("NAPARI_HARPY_PRESET_DIR", str(target))
    assert get_preset_dir() == target
    assert target.is_dir()


def test_sanitize_makes_a_safe_stem():
    assert _sanitize("panel/with:weird chars") == "panel_with_weird_chars"
    assert _sanitize("///") == "preset"


def test_panel_name_is_deterministic_and_order_independent():
    assert _panel_name(["DAPI", "PanCK"]) == _panel_name(["DAPI", "PanCK"])
    assert _panel_name(["DAPI", "PanCK"]) != _panel_name(["DAPI", "CD68"])
    assert _panel_name([]) == "preset_default"


def test_table_style_spec_round_trips_through_json():
    spec = TableColorSourceSpec(
        table_name="t",
        source_kind="x_var",
        value_key="CD68",
        value_kind="continuous",
        layer="log1p",
        smoother="k100",
        colormap="inferno",
    )
    restored = _deserialize_style_spec(json.loads(json.dumps(_serialize_style_spec(spec))))
    assert restored == spec
    assert restored.colormap == "inferno"  # the chosen colormap survives a preset save/reload


@pytest.mark.parametrize("umap_neighbors", [None, 25, 100])
def test_gene_module_style_spec_round_trips_through_json(umap_neighbors):
    spec = TableColorSourceSpec(
        table_name="t",
        source_kind="x_module",
        value_key="COL1A1 + COL1A2",
        value_kind="continuous",
        layer="smoothed_k100" if umap_neighbors is None else "log1p",
        colormap="magma",
        module_genes=("COL1A1", "COL1A2"),
        umap_neighbors=umap_neighbors,
    )

    restored = _deserialize_style_spec(json.loads(json.dumps(_serialize_style_spec(spec))))

    assert restored == spec
    assert restored.module_genes == ("COL1A1", "COL1A2")


def test_shape_style_spec_round_trips_through_json():
    spec = ShapeColumnColorSourceSpec(
        source_kind="shape_column", value_key="area", value_kind="continuous", colormap="magma"
    )
    restored = _deserialize_style_spec(json.loads(json.dumps(_serialize_style_spec(spec))))
    assert restored == spec
    assert restored.colormap == "magma"


def test_style_spec_without_colormap_deserializes_to_none():
    # A preset saved before the colormap field existed must still load (colormap -> None -> viridis).
    legacy = {
        "type": "table",
        "table_name": "t",
        "source_kind": "x_var",
        "value_key": "CD68",
        "value_kind": "continuous",
    }
    assert _deserialize_style_spec(legacy).colormap is None


def test_preset_matches_on_panel_subset():
    cfg = {"panel_signature": ["DAPI", "PanCK"]}
    assert _preset_matches(cfg, [{"DAPI", "PanCK", "CD68"}])
    assert not _preset_matches(cfg, [{"DAPI", "CD68"}])


def test_preset_matches_falls_back_to_channel_signature_for_legacy_presets():
    cfg = {"channel_signature": ["DAPI"]}  # no panel_signature
    assert _preset_matches(cfg, [{"DAPI", "PanCK"}])
    assert not _preset_matches({"panel_signature": [], "channel_signature": []}, [{"DAPI"}])


def test_selected_point_values_reads_categorical_features():
    features = pd.DataFrame({"gene": pd.Categorical(["CCL19", "CCL19"], categories=["CCL19", "CD68"])})
    layer = Points(np.zeros((2, 2)), ndim=2, features=features)
    assert _selected_point_values(layer, "gene") == ["CCL19", "CD68"]
    assert _selected_point_values(layer, "absent") == []


def test_apply_image_props_sorts_inverted_contrast_and_sets_colormap():
    layer = Image(np.arange(256, dtype=float).reshape(16, 16))
    _apply_image_props(layer, {"contrast_limits": [200.0, 10.0], "colormap": "green", "gamma": 0.5})
    assert list(layer.contrast_limits) == [10.0, 200.0]
    assert layer.colormap.name == "green"
    assert layer.gamma == pytest.approx(0.5)


def test_apply_image_props_skips_degenerate_equal_contrast():
    layer = Image(np.arange(256, dtype=float).reshape(16, 16))
    layer.contrast_limits = [20.0, 80.0]
    _apply_image_props(layer, {"contrast_limits": [50.0, 50.0]})  # equal limits are degenerate
    assert list(layer.contrast_limits) == [20.0, 80.0]  # left unchanged, not collapsed


# -- save / capture ------------------------------------------------------------


def test_save_refused_when_no_layers(make_napari_viewer):
    inter = _interactive(make_napari_viewer, blobs())
    assert inter.save_config() is None


def test_capture_records_overlay_channels_and_panel(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    cfg = inter._config_manager.capture()
    images = [entry for entry in cfg["layers"] if entry["kind"] == "image"]
    assert {entry["channel"] for entry in images} == {"0", "2"}
    assert all(entry["mode"] == "overlay" for entry in images)
    assert cfg["panel_signature"] == ["0", "1", "2"]
    assert cfg["coordinate_system"] == "global"


def test_save_writes_a_named_file(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    path = inter.save_config("myview")
    assert path is not None and path.exists()
    assert json.loads(path.read_text())["name"] == "myview"
    assert inter.list_configs() == ["myview"]


# -- apply / round trip --------------------------------------------------------


def test_image_display_round_trips_into_a_fresh_viewer(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    channel0, channel2 = _load_overlay(inter, sdata)
    channel0.colormap = "green"
    channel0.contrast_limits = [10.0, 200.0]
    channel0.gamma = 0.7
    channel2.opacity = 0.5
    channel2.visible = False
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    layers = {layer.name: layer for layer in fresh.viewer.layers}
    assert set(layers) >= {"blobs_image[0]", "blobs_image[2]"}
    restored0 = layers["blobs_image[0]"]
    assert restored0.colormap.name == "green"
    assert list(restored0.contrast_limits) == [10.0, 200.0]
    assert restored0.gamma == pytest.approx(0.7)
    restored2 = layers["blobs_image[2]"]
    assert restored2.opacity == pytest.approx(0.5)
    assert restored2.visible is False


def test_camera_and_scale_bar_round_trip(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.viewer.camera.zoom = 4.25
    inter.viewer.scale_bar.visible = True
    inter.viewer.scale_bar.unit = "um"
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    assert fresh.viewer.camera.zoom == pytest.approx(4.25)
    assert fresh.viewer.scale_bar.unit == "um"
    assert fresh.viewer.scale_bar.visible is True


def test_styled_labels_round_trip_with_custom_palette(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    result = inter.app_state.viewer_adapter.ensure_styled_labels_loaded(sdata, "blobs_labels", "global", TABLE_SPEC)
    result.layer.opacity = 0.4
    # A custom per-category palette must survive into the applied sample.
    from napari_harpy.viewer.category_colors import category_values, store_category_colors, uns_colors_key

    categories = category_values(sdata.tables["table"], "region")
    hexes = ["#112233", "#445566", "#778899", "#aabbcc", "#ddeeff", "#010203", "#040506"]
    custom = dict(zip(categories, hexes, strict=False))
    store_category_colors(sdata.tables["table"], "region", {c: custom[c] for c in categories})
    inter.save_config("view")

    target = blobs()  # a different object with the same panel and columns
    target.tables["table"].uns.pop(uns_colors_key("region"), None)  # palette must come from the preset, not the data
    fresh = _interactive(make_napari_viewer, target, autoload=True)
    # The styled *name* pins that the labels layer was actually styled (a primary layer
    # would just be "blobs_labels"), not merely that some blobs_labels layer exists.
    names = [layer.name for layer in fresh.viewer.layers]
    assert "blobs_labels[obs:region]" in names
    assert fresh.viewer.layers["blobs_labels[obs:region]"].opacity == pytest.approx(0.4)
    restored_palette = [str(c) for c in target.tables["table"].uns[uns_colors_key("region")]]
    assert restored_palette[: len(categories)] == [custom[c] for c in categories]


def test_autoload_matches_a_sibling_sample_by_panel(make_napari_viewer):
    source = blobs()
    inter = _interactive(make_napari_viewer, source)
    _load_overlay(inter, source)
    inter.viewer.layers["blobs_image[0]"].colormap = "green"
    inter.save_config()  # panel-hashed name

    sibling = blobs()  # same channel panel, different object
    fresh = _interactive(make_napari_viewer, sibling, autoload=True)
    names = [layer.name for layer in fresh.viewer.layers]
    assert "blobs_image[0]" in names
    assert fresh.viewer.layers["blobs_image[0]"].colormap.name == "green"


def test_autoload_returns_false_without_a_match(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    # No preset saved for this panel.
    assert inter._config_manager.autoload() is False
    assert inter.load_config() is False


def test_apply_named_raises_for_an_unknown_preset(make_napari_viewer):
    inter = _interactive(make_napari_viewer, blobs())
    with pytest.raises(FileNotFoundError):
        inter.load_config("does-not-exist")


def test_load_config_by_name_degrades_on_a_corrupt_named_file(make_napari_viewer):
    inter = _interactive(make_napari_viewer, blobs())
    # A valid-JSON-but-non-object file at the sanitized name path must not raise a raw
    # AttributeError through the public load_config(name); it has no usable preset.
    (get_preset_dir() / "broken.json").write_text("[1, 2, 3]")
    with pytest.raises(FileNotFoundError):
        inter.load_config("broken")


def test_apply_returns_false_for_a_non_dict_cfg(make_napari_viewer):
    inter = _interactive(make_napari_viewer, blobs())
    assert inter._config_manager.apply([1, 2, 3]) is False


def test_primary_labels_round_trip(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    result = inter.app_state.viewer_adapter.ensure_labels_loaded(sdata, "blobs_labels", "global")
    result.layer.opacity = 0.45
    result.layer.visible = False
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    names = [layer.name for layer in fresh.viewer.layers]
    assert "blobs_labels" in names  # the primary (unbracketed) labels layer
    restored = fresh.viewer.layers["blobs_labels"]
    assert restored.opacity == pytest.approx(0.45)
    assert restored.visible is False


def test_points_selection_round_trips_best_effort(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    # Build a points value-selection layer through the same synchronous path the preset uses.
    from napari_harpy._points_value_index import (
        build_points_value_table,
        load_points,
        validate_points_element_for_value_selection,
    )
    from napari_harpy.viewer.adapter import PointsLayerIdentity

    index_column = "genes"  # blobs points carry a categorical gene column
    validated = validate_points_element_for_value_selection(sdata, "blobs_points", index_column=index_column)
    table = build_points_value_table(validated)
    selection = load_points(validated, table, ["gene_a"])
    identity = PointsLayerIdentity(
        sdata=sdata, points_name="blobs_points", coordinate_system="global", index_column=index_column
    )
    inter.app_state.viewer_adapter._ensure_points_layer_from_selection(identity, selection=selection)
    cfg = inter._config_manager.capture()
    points_entries = [entry for entry in cfg["layers"] if entry["kind"] == "points"]
    assert points_entries and points_entries[0]["selected_values"] == ["gene_a"]
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    point_layers = [layer for layer in fresh.viewer.layers if isinstance(layer, Points)]
    assert point_layers  # the gene selection was rebuilt without raising
    # The rebuilt layer must hold the CAPTURED gene, not every gene (an "all" reload).
    rebuilt = point_layers[0]
    assert list(rebuilt.features["genes"].cat.categories) == ["gene_a"]


# -- the paths the first review found unpinned ---------------------------------


def test_integer_categorical_palette_round_trips(make_napari_viewer):
    # The proven bug: capture stringifies categories, but store_category_colors matches raw
    # (integer) categories, so an integer cluster column's custom palette was silently dropped.
    sdata = _add_int_cluster(blobs())
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.app_state.viewer_adapter.ensure_styled_labels_loaded(sdata, "blobs_labels", "global", CLUSTER_SPEC)
    custom = {0: "#111111", 1: "#222222", 2: "#333333"}
    store_category_colors(sdata.tables["table"], "cluster", custom)
    inter.save_config("view")

    target = _add_int_cluster(blobs())
    target.tables["table"].uns.pop(uns_colors_key("cluster"), None)
    _interactive(make_napari_viewer, target, autoload=True)
    restored = [str(c) for c in target.tables["table"].uns[uns_colors_key("cluster")]]
    assert restored[:3] == ["#111111", "#222222", "#333333"]


def test_restore_palettes_skips_when_a_category_is_missing(make_napari_viewer):
    sdata = _add_int_cluster(blobs(), n_categories=3)
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.app_state.viewer_adapter.ensure_styled_labels_loaded(sdata, "blobs_labels", "global", CLUSTER_SPEC)
    store_category_colors(sdata.tables["table"], "cluster", {0: "#111111", 1: "#222222", 2: "#333333"})
    inter.save_config("view")

    # Target has a 4th cluster the preset never saw -> palette must be left to the default.
    target = _add_int_cluster(blobs(), n_categories=4)
    target.tables["table"].uns.pop(uns_colors_key("cluster"), None)
    _interactive(make_napari_viewer, target, autoload=True)
    assert uns_colors_key("cluster") not in target.tables["table"].uns


def test_list_presets_survives_a_malformed_file(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.save_config("good")
    # A non-UTF-8 file (UnicodeDecodeError, a ValueError but not JSONDecodeError) must not
    # break match/autoload/list.
    (get_preset_dir() / "corrupt.json").write_bytes(b"\xff\xfe not utf-8 \x00")
    (get_preset_dir() / "invalid.json").write_text("{ not json")

    assert inter.list_configs() == ["good"]
    fresh = _interactive(make_napari_viewer, blobs(), autoload=True)
    assert "blobs_image[0]" in [layer.name for layer in fresh.viewer.layers]


def test_list_presets_skips_valid_json_that_is_not_an_object(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.save_config("good")
    # Valid JSON, wrong shape: a later cfg.get(...) would raise AttributeError on these.
    (get_preset_dir() / "array.json").write_text("[1, 2, 3]")
    (get_preset_dir() / "scalar.json").write_text("42")

    assert inter.list_configs() == ["good"]
    assert inter._config_manager.match_preset() is not None  # no AttributeError from the bad files


def test_match_preset_prefers_the_most_recently_modified(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.save_config("older")
    inter.save_config("newer")
    # Both presets share the panel and match; the newer file must win the tie-break.
    older, newer = get_preset_dir() / "older.json", get_preset_dir() / "newer.json"
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    assert inter._config_manager.match_preset()["name"] == "newer"
    os.utime(older, (3_000, 3_000))  # make the other one newest instead
    assert inter._config_manager.match_preset()["name"] == "older"


def test_stack_image_round_trip_including_blending(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    result = inter.app_state.viewer_adapter.ensure_image_loaded(sdata, "blobs_image", "global", mode="stack")
    layer = result.layers[0]
    layer.colormap = "magma"
    layer.contrast_limits = [5.0, 150.0]
    layer.gamma = 0.6
    layer.blending = "opaque"
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    restored = fresh.viewer.layers["blobs_image"]
    assert restored.colormap.name == "magma"
    assert list(restored.contrast_limits) == [5.0, 150.0]
    assert restored.gamma == pytest.approx(0.6)
    assert str(restored.blending) == "opaque"


def test_primary_shapes_round_trip(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    result = inter.app_state.viewer_adapter.ensure_shapes_loaded(sdata, "blobs_polygons", "global")
    result.layer.opacity = 0.35
    result.layer.blending = "translucent_no_depth"
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    shapes = [layer for layer in fresh.viewer.layers if layer.name.startswith("blobs_polygons")]
    assert shapes
    assert shapes[0].opacity == pytest.approx(0.35)
    assert str(shapes[0].blending) == "translucent_no_depth"


def test_styled_shapes_round_trip(make_napari_viewer):
    sdata = blobs()
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    result = inter.app_state.viewer_adapter.ensure_styled_shapes_loaded(
        sdata, "blobs_circles", "global", RADIUS_SPEC, fill=True
    )
    result.layer.opacity = 0.6
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    # The styled name pins that the shapes layer was styled by the shape column, not
    # loaded as a primary "blobs_circles" geometry layer.
    names = [layer.name for layer in fresh.viewer.layers]
    assert "blobs_circles[shapes_column:radius]" in names
    assert fresh.viewer.layers["blobs_circles[shapes_column:radius]"].opacity == pytest.approx(0.6)


def test_apply_returns_false_when_no_layer_applies(make_napari_viewer):
    source = blobs()
    inter = _interactive(make_napari_viewer, source)
    result = inter.app_state.viewer_adapter.ensure_shapes_loaded(source, "blobs_polygons", "global")
    result.layer.opacity = 0.5
    cfg = inter._config_manager.capture(name="only-shapes")
    # Rewrite the sole layer to reference an element no sample has, then apply it.
    cfg["layers"] = [{**cfg["layers"][0], "element": "does_not_exist"}]
    assert inter._config_manager.apply(cfg) is False


def test_all_finite_detects_nested_non_finite():
    assert _all_finite([1.0, [2.0, 3.0], (4.0,)])
    assert not _all_finite([1.0, [2.0, float("nan")]])
    assert not _all_finite([float("inf")])


def test_capture_camera_drops_a_non_finite_camera():
    from types import SimpleNamespace

    camera = SimpleNamespace(center=(0.0, 0.0), zoom=float("inf"), angles=(0.0, 0.0, 0.0), perspective=0.0)
    app_state = SimpleNamespace(viewer=SimpleNamespace(camera=camera), viewer_adapter=None)
    manager = ViewerConfigManager(app_state)
    assert manager._capture_camera() == {}


# -- layer stacking order ------------------------------------------------------


class _FakeLayers(list):
    """A faithful stand-in for napari's LayerList.move.

    Mirrors napari 0.7.1 ``EventedList.move`` exactly, including its pre-move-space
    index semantics (``dest_index -= 1`` when moving upward): a plain pop/insert would
    diverge for any ``dest > src`` move and let an upward-move regression slip past.
    """

    def move(self, src_index: int, dest_index: int = 0) -> bool:
        if dest_index < 0:
            dest_index += len(self) + 1
        if dest_index in (src_index, src_index + 1):
            return False
        item = self.pop(src_index)
        if dest_index > src_index:
            dest_index -= 1
        self.insert(dest_index, item)
        return True


def _order_manager(names):
    from types import SimpleNamespace

    layers = _FakeLayers(SimpleNamespace(name=n) for n in names)
    return ViewerConfigManager(SimpleNamespace(viewer=SimpleNamespace(layers=layers))), layers


def test_restore_layer_order_lifts_a_buried_layer_back_on_top():
    # The bug: reload rebuilds images last, so a shapes/labels layer the user put on
    # top comes back at the bottom (index 0), buried under the opaque images.
    manager, layers = _order_manager(["boundaries[obs:niche]", "img[DAPI]", "img[CD68]"])
    manager._restore_layer_order(["img[DAPI]", "img[CD68]", "boundaries[obs:niche]"])
    assert [layer.name for layer in layers] == ["img[DAPI]", "img[CD68]", "boundaries[obs:niche]"]


def test_restore_layer_order_is_a_noop_when_already_ordered():
    manager, layers = _order_manager(["img[DAPI]", "img[CD68]", "boundaries[obs:niche]"])
    manager._restore_layer_order(["img[DAPI]", "img[CD68]", "boundaries[obs:niche]"])
    assert [layer.name for layer in layers] == ["img[DAPI]", "img[CD68]", "boundaries[obs:niche]"]


def test_restore_layer_order_floats_unnamed_layers_above_and_tolerates_missing():
    # An extra layer not in the preset stays above the ordered ones; a preset name with
    # no matching layer (e.g. one that failed to load) is simply skipped.
    manager, layers = _order_manager(["extra", "boundaries[obs:niche]", "img[DAPI]"])
    manager._restore_layer_order(["img[DAPI]", "img[MISSING]", "boundaries[obs:niche]"])
    names = [layer.name for layer in layers]
    assert names[:2] == ["img[DAPI]", "boundaries[obs:niche]"]  # preset order at the bottom
    assert names[-1] == "extra"  # the un-named layer floats above


def test_restore_layer_order_leaves_legacy_presets_untouched():
    manager, layers = _order_manager(["a", "b", "c"])
    manager._restore_layer_order(None)  # old preset saved before layer_order existed
    assert [layer.name for layer in layers] == ["a", "b", "c"]


def test_restore_layer_order_ignores_a_malformed_field_instead_of_aborting():
    # A corrupt-but-valid-JSON preset must degrade to a no-op like the other restore
    # steps, not raise (which would abort apply() after the layers were already built).
    manager, layers = _order_manager(["a", "b", "c"])
    manager._restore_layer_order("not-a-list")  # wrong type
    manager._restore_layer_order([["a"], "b"])  # unhashable entry -> would break dict build
    manager._restore_layer_order([])  # empty
    assert [layer.name for layer in layers] == ["a", "b", "c"]  # untouched, no exception


def test_layer_order_round_trips_through_save_and_reload(make_napari_viewer):
    # End-to-end: drag styled labels above the images, save, reload -> still on top.
    sdata = _add_int_cluster(blobs())
    inter = _interactive(make_napari_viewer, sdata)
    _load_overlay(inter, sdata)
    inter.app_state.viewer_adapter.ensure_styled_labels_loaded(sdata, "blobs_labels", "global", CLUSTER_SPEC)

    viewer = inter.viewer
    styled = next(layer for layer in viewer.layers if layer.name.startswith("blobs_labels"))
    viewer.layers.move(viewer.layers.index(styled), len(viewer.layers) - 1)
    top_name = viewer.layers[-1].name
    assert top_name.startswith("blobs_labels")  # sanity: the styled layer is on top now

    cfg = inter._config_manager.capture()
    assert cfg["layer_order"][-1] == top_name  # capture recorded it on top
    inter.save_config("view")

    fresh = _interactive(make_napari_viewer, sdata, autoload=True)
    fresh_order = [layer.name for layer in fresh.viewer.layers]
    assert fresh_order == cfg["layer_order"]  # the FULL stack is restored, not just the top
    assert fresh_order[-1] == top_name  # ...and the styled layer sits on top, not buried
