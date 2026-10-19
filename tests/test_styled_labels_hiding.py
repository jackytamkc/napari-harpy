"""Hiding a category on styled LABELS drops those cells to background in the DISPLAY (not the zarr).

Regression for the labels analog of the shapes hiding bug: a category hidden via the colour editor
(palette alpha 0) reappeared under napari's ``opaque`` blending, which ignores the DirectLabelColormap
alpha. The styled labels layer now maps hidden-category label IDs to background (0) in its in-memory
data, so hiding is blending-independent; the underlying element / on-disk zarr is never modified.
"""

from __future__ import annotations

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel

from napari_harpy import Interactive
from napari_harpy.core._color_source import TableColorSourceSpec
from napari_harpy.viewer.category_colors import reset_category_colors, store_category_colors
from napari_harpy.viewer.labels_styling import masked_label_ids

SPEC = TableColorSourceSpec(
    table_name="table", source_kind="obs_column", value_key="celltype", value_kind="categorical"
)


def _table(celltypes: list[str], region: str) -> ad.AnnData:
    n = len(celltypes)
    obs = pd.DataFrame(
        {
            "region": pd.Categorical([region] * n),
            "instance_id": list(range(1, n + 1)),  # label ids 1..n (0 is background)
            "celltype": pd.Categorical(celltypes),
        }
    )
    return TableModel.parse(
        ad.AnnData(X=np.zeros((n, 1), np.float32), obs=obs),
        region=region,
        region_key="region",
        instance_key="instance_id",
    )


def _labels_sdata(celltypes: list[str]) -> SpatialData:
    n = len(celltypes)
    labels = np.zeros((3, n), dtype=np.int32)
    for i in range(n):
        labels[:, i] = i + 1  # column i holds label id i+1
    return SpatialData(
        labels={"cells": Labels2DModel.parse(labels)},
        tables={"table": _table(celltypes, "cells")},
    )


def _dask_labels_sdata(celltypes: list[str], scale_factors: list[int] | None = None) -> SpatialData:
    """Dask-backed labels (optionally pyramided) -- the real segmentation shape (zarr-backed, multiscale)."""
    n = len(celltypes)
    labels = np.zeros((8, 4 * n), dtype=np.int32)
    for i in range(n):
        labels[:, 4 * i : 4 * (i + 1)] = i + 1  # each cell is a 4-wide block that survives a /2 downscale
    dask_labels = da.from_array(labels, chunks=(8, 4))
    parsed = Labels2DModel.parse(dask_labels, scale_factors=scale_factors)
    return SpatialData(labels={"cells": parsed}, tables={"table": _table(celltypes, "cells")})


def _interactive(make_napari_viewer, sdata):
    inter = Interactive(sdata, viewer=make_napari_viewer(show=False), headless=True, autoload=False)
    inter.app_state.set_coordinate_system("global")
    return inter


def _style(inter, sdata):
    return inter.app_state.viewer_adapter.ensure_styled_labels_loaded(sdata, "cells", "global", SPEC).layer


def _hide(sdata, hidden: set[str], categories: list[str]) -> None:
    colors = {c: ((1, 0, 0, 0.0) if c in hidden else (0, 0, 1, 1.0)) for c in categories}
    store_category_colors(sdata.tables["table"], "celltype", colors)


def _display(layer) -> np.ndarray:
    data = layer.data[0] if isinstance(layer.data, (list, tuple)) else layer.data
    return np.asarray(data)


def _element(sdata) -> np.ndarray:
    element = sdata["cells"]
    return np.asarray(element.data if hasattr(element, "data") else element)


def test_masked_label_ids_returns_hidden_category_instances(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A", "B"])  # label ids 1,2,3,4, all typed
    _hide(sdata, {"B"}, ["A", "B"])
    assert masked_label_ids(sdata, "cells", SPEC) == frozenset({2, 4})  # just the hidden B cells


def test_masked_label_ids_empty_when_nothing_hidden_and_all_typed(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A"])
    assert masked_label_ids(sdata, "cells", SPEC) == frozenset()


def test_masked_label_ids_excludes_continuous_obs_sources(make_napari_viewer):
    # A continuous obs column with NaN cells must NOT be masked by this path (scoped to categorical
    # rendering). The gate keys off the render dtype, not value_kind.
    import numpy as _np
    import pandas as _pd
    from napari_harpy.core._color_source import TableColorSourceSpec as _Spec

    sdata = _labels_sdata(["A", "B", "A", "B"])
    obs = sdata.tables["table"].obs
    obs["score"] = _pd.Series([0.5, _np.nan, 0.9, _np.nan], index=obs.index)  # numeric, some NaN
    cont_spec = _Spec(table_name="table", source_kind="obs_column", value_key="score", value_kind="continuous")
    assert masked_label_ids(sdata, "cells", cont_spec) == frozenset()  # continuous NaN cells NOT masked


def test_masked_label_ids_uses_render_dtype_not_value_kind(make_napari_viewer):
    # An all-1 (or all-0) nullable-Int64 column renders CATEGORICALLY (greying its NaN cells), even though
    # the value_kind classifier calls it "continuous". Keying off the render dtype keeps those grey cells
    # maskable; keying off value_kind (the old bug) would have left them stuck.
    from napari_harpy.core._color_source import TableColorSourceSpec as _Spec

    sdata = _labels_sdata(["A", "B", "A", "B"])  # ids 1..4; the column below is what matters
    sdata.tables["table"].obs["marker"] = pd.array([1, pd.NA, 1, pd.NA], dtype="Int64")  # ids 2,4 NaN
    wrong = _Spec(table_name="table", source_kind="obs_column", value_key="marker", value_kind="continuous")
    assert masked_label_ids(sdata, "cells", wrong) == frozenset({2, 4})


def test_unassigned_cells_are_masked_on_a_multiscale_layer(make_napari_viewer):
    sdata = _dask_labels_sdata(["A", None, "B", None], scale_factors=[2])  # ids 2,4 unassigned
    inter = _interactive(make_napari_viewer, sdata)
    layer = _style(inter, sdata)
    assert layer.multiscale
    for level in layer.data:  # unassigned cells gone from every pyramid level
        assert not np.isin(np.asarray(level), [2, 4]).any()


def test_nothing_hidden_leaves_display_equal_to_element(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A"])
    inter = _interactive(make_napari_viewer, sdata)
    layer = _style(inter, sdata)
    assert np.array_equal(_display(layer), _element(sdata))


def test_hiding_drops_labels_from_display_not_the_element(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A", "B"])  # ids 1(A), 2(B), 3(A), 4(B)
    inter = _interactive(make_napari_viewer, sdata)
    element_before = _element(sdata).copy()
    _hide(sdata, {"B"}, ["A", "B"])
    display = _display(_style(inter, sdata))
    assert not np.isin(display, [2, 4]).any()  # B cells gone from the display
    assert np.isin([1, 3], display).all()  # A cells kept
    assert np.array_equal(_element(sdata), element_before)  # element / zarr untouched
    assert np.isin(_element(sdata), [2, 4]).any()  # hidden ids still on disk


def test_hidden_labels_stay_gone_under_opaque_blending(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"B"}, ["A", "B"])
    layer = _style(inter, sdata)
    layer.blending = "opaque"  # opaque ignores the colormap alpha; the data masking is what hides B
    assert not np.isin(_display(layer), [2, 4]).any()


def test_unhide_restores_the_full_display(make_napari_viewer):
    sdata = _labels_sdata(["A", "B", "A", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    element = _element(sdata).copy()
    _hide(sdata, {"B"}, ["A", "B"])
    assert not np.isin(_display(_style(inter, sdata)), [2, 4]).any()
    reset_category_colors(sdata.tables["table"], "celltype")
    assert np.array_equal(_display(_style(inter, sdata)), element)


def test_hide_all_yields_all_background(make_napari_viewer):
    sdata = _labels_sdata(["A", "B"])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A", "B"}, ["A", "B"])
    assert int(_display(_style(inter, sdata)).max()) == 0  # every cell hidden -> all background


def test_unassigned_cells_are_masked_even_with_nothing_hidden(make_napari_viewer):
    # Cells with no cell type (NaN) render flat grey (#808080 missing colour). They must be masked to
    # background too -- they are "not a visible cell type" -- so hiding all real types leaves a clean
    # overlay. This holds even when no category is explicitly hidden.
    sdata = _labels_sdata(["A", None, "B", None])  # ids 1(A), 2(NaN), 3(B), 4(NaN)
    inter = _interactive(make_napari_viewer, sdata)
    element_before = _element(sdata).copy()
    display = _display(_style(inter, sdata))
    assert not np.isin(display, [2, 4]).any()  # unassigned cells gone from the display
    assert np.isin([1, 3], display).all()  # typed cells still shown
    assert np.array_equal(_element(sdata), element_before)  # element/zarr untouched (display-only)


def test_masked_label_ids_combines_hidden_and_unassigned(make_napari_viewer):
    sdata = _labels_sdata(["A", None, "B", None])  # ids 1(A), 2(NaN), 3(B), 4(NaN)
    _hide(sdata, {"B"}, ["A", "B"])  # hide B (id 3)
    assert masked_label_ids(sdata, "cells", SPEC) == frozenset({2, 3, 4})  # NaN(2,4) + hidden B(3)


def test_hide_all_types_with_unassigned_gives_a_clean_display(make_napari_viewer):
    sdata = _labels_sdata(["A", None, "B", None])
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"A", "B"}, ["A", "B"])  # hide every real type; NaN cells already masked
    assert int(_display(_style(inter, sdata)).max()) == 0  # nothing left on the overlay


def test_reassigning_a_cell_into_a_hidden_class_masks_it_on_restyle(make_napari_viewer):
    # The change key must track obs cell->category assignments, not only the hidden category NAMES:
    # moving a cell into an already-hidden class (hidden-name set unchanged) must mask it next restyle.
    sdata = _labels_sdata(["A", "B", "A", "A"])  # ids 1(A), 2(B), 3(A), 4(A)
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"B"}, ["A", "B"])
    display = _display(_style(inter, sdata))
    assert not np.isin(display, [2]).any()  # B (id 2) hidden
    assert np.isin([1, 3, 4], display).all()  # A cells shown

    # Reassign cell id 1 from A -> B. B is still the only hidden *name*, but id 1 must now be masked.
    sdata.tables["table"].obs["celltype"] = pd.Categorical(["B", "B", "A", "A"], categories=["A", "B"])
    display = _display(_style(inter, sdata))
    assert not np.isin(display, [1, 2]).any()  # id 1 now hidden too (regression guard for the stale mask)
    assert np.isin([3, 4], display).all()


def test_multiscale_hiding_masks_every_level_without_crashing(make_napari_viewer):
    # napari multiscale layer.data is a MultiScaleData (not a list) -- the fix must mask each pyramid
    # level, not skip the branch and assign a single coarse array (which raised ValueError on load).
    sdata = _dask_labels_sdata(["A", "B", "A", "B"], scale_factors=[2])  # ids 1(A),2(B),3(A),4(B)
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"B"}, ["A", "B"])
    layer = _style(inter, sdata)
    assert layer.multiscale
    assert len(layer.data) >= 2
    for level in layer.data:  # B (ids 2,4) removed from EVERY level, not just the base
        assert not np.isin(np.asarray(level), [2, 4]).any()


def test_dask_backed_labels_stay_lazy_after_hiding(make_napari_viewer):
    # The mask must run on the underlying dask array (lazy), not eagerly materialise the full raster.
    sdata = _dask_labels_sdata(["A", "B", "A", "B"])  # single-scale, dask-backed
    inter = _interactive(make_napari_viewer, sdata)
    _hide(sdata, {"B"}, ["A", "B"])
    data = _style(inter, sdata).data
    underlying = data.data if hasattr(data, "dims") else data  # unwrap the xarray DataArray
    assert isinstance(underlying, da.Array)  # still lazy, not a fully-realised ndarray
    assert not np.isin(np.asarray(data), [2, 4]).any()
