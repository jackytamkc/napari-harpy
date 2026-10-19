from __future__ import annotations

from dataclasses import replace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, ShapesModel, TableModel

from napari_harpy.core._color_source import TableColorSourceSpec
from napari_harpy.viewer._styling import continuous_rgba_for_values
from napari_harpy.viewer.adapter import ViewerAdapter
from napari_harpy.viewer.viewer_config import _deserialize_style_spec, _serialize_style_spec


def _module_sdata(kind: str) -> tuple[SpatialData, np.ndarray]:
    # Table order differs from element instance order. Cell 5 has no UMAP.
    instance_ids = [4, 1, 5, 3, 2]
    matrix = np.array([[1.0, 7.0], [2.0, 1.0], [999.0, 999.0], [6.0, 3.0], [3.0, 2.0]])
    table = ad.AnnData(
        X=np.zeros_like(matrix),
        obs=pd.DataFrame({"region": pd.Categorical(["cells"] * 5), "instance_id": instance_ids}, index=list("abcde")),
        var=pd.DataFrame(index=["A", "B"]),
    )
    table.layers["log1p"] = matrix
    table.obsm["X_umap"] = np.array([[0, 0], [10, 0], [np.nan, np.nan], [0.1, 0], [10.1, 0]])
    table = TableModel.parse(table, region="cells", region_key="region", instance_key="instance_id")
    if kind == "labels":
        sdata = SpatialData(
            labels={"cells": Labels2DModel.parse(np.tile(np.arange(1, 7, dtype=np.int32), (3, 1)))},
            tables={"table": table},
        )
    else:
        if kind == "polygons":
            frame = gpd.GeoDataFrame(geometry=[box(i * 2, 0, i * 2 + 1, 1) for i in range(6)])
        else:
            frame = gpd.GeoDataFrame({"radius": [1.0] * 6}, geometry=[Point(i * 2, 0) for i in range(6)])
        frame.index = pd.Index(range(1, 7), name="instance_id")
        sdata = SpatialData(shapes={"cells": ShapesModel.parse(frame)}, tables={"table": table})
    # Explicit reference: UMAP pairs table rows (0, 3) and (1, 4), each with self included.
    smoothed = np.array([matrix[[0, 3]].mean(axis=0), matrix[[1, 4]].mean(axis=0)] * 2)
    expected_valid = ((smoothed - smoothed.mean(axis=0)) / smoothed.std(axis=0)).mean(axis=1)
    expected = np.array([expected_valid[1], expected_valid[1], expected_valid[0], expected_valid[0], np.nan])
    return sdata, expected


@pytest.mark.parametrize("kind", ["labels", "polygons", "circles"])
def test_umap_gene_module_renders_aligned_scores_and_recolors_without_duplicate_layer(make_napari_viewer, kind):
    sdata, expected = _module_sdata(kind)
    viewer = make_napari_viewer(show=False)
    adapter = ViewerAdapter(viewer)
    spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_module",
        value_key="A + B",
        value_kind="continuous",
        module_genes=("A", "B"),
        layer="log1p",
        umap_neighbors=2,
    )
    # Same inputs must survive the format used by viewer presets.
    spec = _deserialize_style_spec(_serialize_style_spec(spec))
    load = adapter.ensure_styled_labels_loaded if kind == "labels" else adapter.ensure_styled_shapes_loaded
    result = load(sdata, "cells", "global", spec)
    layer = result.layer
    assert result.value_kind == "continuous"
    assert "log1p~UMAP-k2:module:A + B" in layer.name
    if kind == "labels":
        features = layer.features.set_index("instance_id").loc[range(1, 6), "A + B"].to_numpy()
    else:
        features = layer.features["A + B"].to_numpy()[:5]
    np.testing.assert_allclose(features, expected, atol=1e-14, equal_nan=True)

    recolored = load(sdata, "cells", "global", replace(spec, colormap="magma"))
    assert recolored.layer is layer
    assert not recolored.created
    assert len(viewer.layers) == 1
    assert adapter.layer_bindings.get_binding(layer).style_spec.colormap == "magma"
    colors = continuous_rgba_for_values(pd.Series(expected), colormap_name="magma")
    if kind == "labels":
        np.testing.assert_allclose(layer.colormap.map(np.array([1, 2, 3, 4])), colors[:4], atol=1e-6)
    elif kind == "polygons":
        np.testing.assert_allclose(layer.edge_color[:4], colors[:4], atol=1e-6)
    else:
        np.testing.assert_allclose(layer.face_color[:4], colors[:4], atol=1e-6)
    assert "A + B" not in sdata.tables["table"].obs
    assert len(sdata.tables["table"].obsp) == 0


@pytest.mark.parametrize("kind", ["labels", "polygons", "circles"])
@pytest.mark.parametrize("failure", ["gene", "layer", "umap"])
def test_failed_module_load_removes_fresh_overlay(make_napari_viewer, kind, failure):
    sdata, _ = _module_sdata(kind)
    viewer = make_napari_viewer(show=False)
    adapter = ViewerAdapter(viewer)
    spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_module",
        value_key="A + B",
        value_kind="continuous",
        module_genes=("A", "Missing") if failure == "gene" else ("A", "B"),
        layer="Missing" if failure == "layer" else "log1p",
        umap_neighbors=2,
    )
    if failure == "umap":
        sdata.tables["table"].obsm["X_umap"][:] = np.nan
    load = adapter.ensure_styled_labels_loaded if kind == "labels" else adapter.ensure_styled_shapes_loaded

    with pytest.raises(ValueError):
        load(sdata, "cells", "global", spec)

    assert len(viewer.layers) == 0
    assert adapter.layer_bindings.find_bindings(element_name="cells") == []


@pytest.mark.parametrize("kind", ["labels", "polygons", "circles"])
def test_failed_module_restyle_preserves_last_successful_binding_and_features(make_napari_viewer, kind):
    sdata, _ = _module_sdata(kind)
    viewer = make_napari_viewer(show=False)
    adapter = ViewerAdapter(viewer)
    spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_module",
        value_key="A + B",
        value_kind="continuous",
        module_genes=("A", "B"),
        layer="log1p",
        umap_neighbors=2,
    )
    load = adapter.ensure_styled_labels_loaded if kind == "labels" else adapter.ensure_styled_shapes_loaded
    layer = load(sdata, "cells", "global", spec).layer
    features = layer.features.copy(deep=True)
    old_name = layer.name
    sdata.tables["table"].obsm["X_umap"][:] = np.nan

    with pytest.raises(ValueError, match="no cells with finite coordinates"):
        load(sdata, "cells", "global", replace(spec, colormap="inferno"))

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    assert layer.name == old_name
    assert adapter.layer_bindings.get_binding(layer).style_spec == spec
    pd.testing.assert_frame_equal(layer.features, features)
