"""Hidden and categorically unassigned rows are absent from styled shapes overlays.

Regression tests for the reported bug: a category hidden via the colour editor (palette alpha 0)
reappeared under napari's ``opaque`` blending, which ignores alpha. Polygon shapes now drop the hidden
geometry entirely; point-radius shapes keep geometry and hide via napari's per-point ``shown`` mask.
Categorical rows with no assigned value use the same mechanism so their missing-colour grey cannot
remain after all real categories are hidden. Both paths are blending-independent.
"""

from __future__ import annotations

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from napari.layers import Points, Shapes
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from napari_harpy import Interactive
from napari_harpy.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from napari_harpy.viewer.category_colors import (
    hidden_categories,
    reset_category_colors,
    store_category_colors,
    uns_colors_key,
)
from napari_harpy.viewer.shapes_styling import source_row_hidden_mask

SPEC = TableColorSourceSpec(
    table_name="table", source_kind="obs_column", value_key="celltype", value_kind="categorical"
)


def _square(x: float) -> Polygon:
    return Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])


def _table(celltypes: list[str | None], region: str, *, instance_ids: list[int] | None = None) -> ad.AnnData:
    n = len(celltypes)
    if instance_ids is None:
        instance_ids = list(range(n))
    obs = pd.DataFrame(
        {
            "region": pd.Categorical([region] * n),
            "instance_id": instance_ids,
            "celltype": pd.Categorical(celltypes),
        }
    )
    return TableModel.parse(
        ad.AnnData(X=np.zeros((n, 1), np.float32), obs=obs),
        region=region,
        region_key="region",
        instance_key="instance_id",
    )


def _polygon_sdata(celltypes: list[str | None]) -> SpatialData:
    geoms = gpd.GeoDataFrame(
        {"geometry": [_square(2 * i) for i in range(len(celltypes))]}, index=list(range(len(celltypes)))
    )
    return SpatialData(shapes={"cells": ShapesModel.parse(geoms)}, tables={"table": _table(celltypes, "cells")})


def _circle_sdata(celltypes: list[str | None]) -> SpatialData:
    geoms = gpd.GeoDataFrame(
        {"geometry": [Point(2 * i, 0) for i in range(len(celltypes))], "radius": [1.0] * len(celltypes)},
        index=list(range(len(celltypes))),
    )
    return SpatialData(shapes={"circ": ShapesModel.parse(geoms)}, tables={"table": _table(celltypes, "circ")})


def _interactive(make_napari_viewer, sdata):
    inter = Interactive(sdata, viewer=make_napari_viewer(show=False), headless=True, autoload=False)
    inter.app_state.set_coordinate_system("global")
    return inter


def _style(inter, sdata, element):
    return inter.app_state.viewer_adapter.ensure_styled_shapes_loaded(sdata, element, "global", SPEC, fill=True).layer


def _hide(sdata, hidden: set[str], categories: list[str]) -> None:
    colors = {c: ((1, 0, 0, 0.0) if c in hidden else (0, 0, 1, 1.0)) for c in categories}
    store_category_colors(sdata.tables["table"], "celltype", colors)


# -- polygon geometry exclusion ------------------------------------------------


def test_polygon_hidden_category_is_excluded_from_geometry(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    assert len(_style(inter, sdata, "cells").data) == 4

    _hide(sdata, {"A"}, ["A", "B"])
    layer = _style(inter, sdata, "cells")
    assert isinstance(layer, Shapes)
    assert len(layer.data) == 2
    # The kept rows are the B source rows (positions 2, 3) — features stay aligned for hover.
    assert list(layer.features.iloc[:, 0]) == [2, 3]


def test_polygon_hidden_stays_excluded_under_opaque_blending(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    layer = _style(inter, sdata, "cells")
    layer.blending = "opaque"  # opaque ignores alpha; geometry exclusion is what keeps A hidden
    assert len(layer.data) == 2


def test_polygon_unhide_restores_geometry(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    assert len(_style(inter, sdata, "cells").data) == 2
    reset_category_colors(sdata.tables["table"], "celltype")
    assert len(_style(inter, sdata, "cells").data) == 4


def test_polygon_hide_all_yields_an_empty_layer(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A", "B"}, ["A", "B"])
    layer = _style(inter, sdata, "cells")  # every category hidden -> empty, not an error
    assert len(layer.data) == 0


def test_multipolygon_hiding_removes_every_rendered_part(make_napari_viewer):
    # Source row 0 is a MultiPolygon (2 parts, category A); row 1 is one square (B).
    multi = MultiPolygon([_square(0), _square(2)])
    geoms = gpd.GeoDataFrame({"geometry": [multi, _square(4)]}, index=[0, 1])
    sdata = SpatialData(shapes={"cells": ShapesModel.parse(geoms)}, tables={"table": _table(["A", "B"], "cells")})
    inter = _interactive(make_napari_viewer, sdata)
    assert len(_style(inter, sdata, "cells").data) == 3  # 2 A-parts + 1 B

    _hide(sdata, {"A"}, ["A", "B"])
    layer = _style(inter, sdata, "cells")
    assert len(layer.data) == 1  # both A parts gone
    assert list(layer.features.iloc[:, 0]) == [1]


# -- categorical missing-value exclusion --------------------------------------


def test_polygon_unassigned_categories_are_excluded_without_hidden_palette(make_napari_viewer):
    sdata = _polygon_sdata(["A", None, "B", None])
    inter = _interactive(make_napari_viewer, sdata)

    layer = _style(inter, sdata, "cells")
    layer.blending = "opaque"

    assert isinstance(layer, Shapes)
    assert len(layer.data) == 2
    assert list(layer.features.iloc[:, 0]) == [0, 2]
    assert len(sdata.shapes["cells"]) == 4  # display filtering never mutates the source element


def test_polygon_hide_all_with_unassigned_categories_yields_empty_layer(make_napari_viewer):
    sdata = _polygon_sdata(["A", None, "B", None])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A", "B"}, ["A", "B"])

    assert len(_style(inter, sdata, "cells").data) == 0


def test_polygon_without_linked_table_rows_is_excluded_for_categorical_overlay(make_napari_viewer):
    geoms = gpd.GeoDataFrame({"geometry": [_square(2 * i) for i in range(4)]}, index=[0, 1, 2, 3])
    # Source rows 1 and 3 have no table row at all, so their aligned categorical value is unassigned.
    table = _table(["A", "B"], "cells", instance_ids=[0, 2])
    sdata = SpatialData(shapes={"cells": ShapesModel.parse(geoms)}, tables={"table": table})
    inter = _interactive(make_napari_viewer, sdata)

    layer = _style(inter, sdata, "cells")

    assert len(layer.data) == 2
    assert list(layer.features.iloc[:, 0]) == [0, 2]


def test_polygon_assignment_change_rebuilds_and_restores_geometry(make_napari_viewer):
    sdata = _polygon_sdata(["A", None, "B", None])
    inter = _interactive(make_napari_viewer, sdata)
    first = _style(inter, sdata, "cells")
    assert len(first.data) == 2

    sdata.tables["table"].obs["celltype"] = pd.Categorical(["A", "A", "B", "B"])
    second = _style(inter, sdata, "cells")

    assert second is not first
    assert len(second.data) == 4


# -- point-radius via `shown` --------------------------------------------------


def test_point_radius_hidden_via_shown_keeps_full_geometry(make_napari_viewer):
    sdata = _circle_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    layer = _style(inter, sdata, "circ")
    assert isinstance(layer, Points)
    assert np.all(np.asarray(layer.shown))

    _hide(sdata, {"A"}, ["A", "B"])
    layer = _style(inter, sdata, "circ")
    assert len(layer.data) == 4  # geometry stays full
    assert list(np.asarray(layer.shown)) == [False, False, True, True]


def test_point_radius_unhide_shows_all(make_napari_viewer):
    sdata = _circle_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    _style(inter, sdata, "circ")
    reset_category_colors(sdata.tables["table"], "celltype")
    layer = _style(inter, sdata, "circ")
    assert np.all(np.asarray(layer.shown))


def test_point_radius_unassigned_categories_are_hidden_via_shown(make_napari_viewer):
    sdata = _circle_sdata(["A", None, "B", None])
    inter = _interactive(make_napari_viewer, sdata)

    layer = _style(inter, sdata, "circ")
    layer.blending = "opaque"

    assert isinstance(layer, Points)
    assert len(layer.data) == 4
    assert list(np.asarray(layer.shown)) == [True, False, True, False]


# -- reuse vs rebuild ----------------------------------------------------------


def test_polygon_restyle_with_same_hidden_set_reuses_the_layer(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    first = _style(inter, sdata, "cells")
    second = _style(inter, sdata, "cells")  # same hidden set -> no rebuild
    assert first is second


def test_polygon_rebuild_preserves_user_view_state(make_napari_viewer):
    # A hidden-set change rebuilds the polygon layer; the user's visibility/opacity/blending/z-order
    # must carry across -- otherwise hiding a second category would reset a deliberately opaque layer.
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    first = _style(inter, sdata, "cells")
    # Add an image ABOVE the styled layer so it is NOT topmost. The rebuild re-appends the new layer
    # on top, so restoring z-order genuinely requires `viewer.layers.move` (not a no-op).
    inter.viewer.add_image(np.zeros((4, 4), np.uint8), name="bg")
    first.visible = False
    first.opacity = 0.3
    first.blending = "opaque"
    first_index = list(inter.viewer.layers).index(first)
    assert first_index < len(inter.viewer.layers) - 1  # precondition: styled layer is not topmost

    _hide(sdata, {"B"}, ["A", "B"])
    second = _style(inter, sdata, "cells")

    assert second is not first  # rebuilt
    assert second.visible is False
    assert second.opacity == pytest.approx(0.3)
    assert str(second.blending) == "opaque"
    assert list(inter.viewer.layers).index(second) == first_index  # z-order restored via move()


def test_polygon_restyle_with_changed_hidden_set_rebuilds(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A"}, ["A", "B"])
    first = _style(inter, sdata, "cells")
    assert len(first.data) == 2
    _hide(sdata, {"B"}, ["A", "B"])
    second = _style(inter, sdata, "cells")
    assert second is not first  # geometry changed -> rebuilt
    assert len(second.data) == 2
    assert list(second.features.iloc[:, 0]) == [0, 1]  # now the A rows are shown


# -- helpers -------------------------------------------------------------------


def test_source_row_hidden_mask_is_none_when_nothing_hidden(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    assert source_row_hidden_mask(sdata, "cells", SPEC) is None


def test_source_row_hidden_mask_marks_hidden_rows(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    _hide(sdata, {"A"}, ["A", "B"])
    mask = source_row_hidden_mask(sdata, "cells", SPEC)
    assert list(mask) == [True, True, False, False]


def test_source_row_hidden_mask_marks_unassigned_and_hidden_rows(make_napari_viewer):
    sdata = _polygon_sdata(["A", None, "B", None])
    _hide(sdata, {"B"}, ["A", "B"])
    mask = source_row_hidden_mask(sdata, "cells", SPEC)
    assert list(mask) == [False, True, True, True]


def test_source_row_hidden_mask_does_not_mask_continuous_missing_values(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    sdata.tables["table"].obs["score"] = pd.Series(
        [0.5, np.nan, 0.9, np.nan],
        index=sdata.tables["table"].obs.index,
    )
    spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="score",
        value_kind="continuous",
    )
    assert source_row_hidden_mask(sdata, "cells", spec) is None


def test_source_row_hidden_mask_uses_render_dtype_not_spec_value_kind(make_napari_viewer):
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    sdata.tables["table"].obs["marker"] = pd.array([0, pd.NA, 1, pd.NA], dtype="Int64")
    # Binary nullable integers render categorically even if an old/persisted spec says continuous.
    stale_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="marker",
        value_kind="continuous",
    )
    mask = source_row_hidden_mask(sdata, "cells", stale_spec)
    assert list(mask) == [False, True, False, True]


def test_source_row_hidden_mask_is_none_for_a_shape_column_spec():
    spec = ShapeColumnColorSourceSpec(source_kind="shape_column", value_key="radius", value_kind="continuous")
    sdata = _circle_sdata(["A", "B"])
    assert source_row_hidden_mask(sdata, "circ", spec) is None


def test_hidden_categories_detects_alpha_zero():
    sdata = _polygon_sdata(["A", "A", "B", "B"])
    table = sdata.tables["table"]
    assert hidden_categories(table, "celltype") == frozenset()
    store_category_colors(table, "celltype", {"A": (1, 0, 0, 0.0), "B": (0, 0, 1, 1.0)})
    assert hidden_categories(table, "celltype") == frozenset({"A"})
    assert uns_colors_key("celltype") in table.uns
