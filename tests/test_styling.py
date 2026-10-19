from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from napari_harpy.viewer._styling import (
    _umap_neighbor_operator,
    categorical_colors_for_values,
    categorical_rgba_for_values,
    continuous_colors_for_values,
    continuous_rgba_for_values,
    module_score_values,
    resolve_expression_matrix,
    var_expression_values,
)
from napari_harpy.viewer.labels_styling import _build_categorical_color_dict, _build_continuous_color_dict


def _expression_table() -> ad.AnnData:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    table = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    table.layers["log1p"] = np.log1p(matrix)
    return table


def test_var_expression_values_reads_X_by_default() -> None:
    table = _expression_table()
    out = var_expression_values(table, var_name="GeneB", obs_positions=np.array([0, 1]), layer=None)
    np.testing.assert_allclose(out, [2.0, 4.0])


def test_var_expression_values_reads_named_layer() -> None:
    table = _expression_table()
    out = var_expression_values(table, var_name="GeneB", obs_positions=np.array([0, 1]), layer="log1p")
    np.testing.assert_allclose(out, np.log1p([2.0, 4.0]))


def test_var_expression_values_rejects_unknown_layer() -> None:
    table = _expression_table()
    with pytest.raises(ValueError, match="Expression layer `nope` is not available"):
        var_expression_values(table, var_name="GeneA", obs_positions=np.array([0]), layer="nope")


def test_var_expression_values_applies_obsp_smoother() -> None:
    import scipy.sparse as sp

    table = _expression_table()  # X = [[1, 2], [3, 4]]; GeneB column = [2, 4]
    # Row-stochastic smoother that averages the two cells -> each cell becomes the mean.
    table.obsp["smoother_k100"] = sp.csr_matrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float32"))

    smoothed = var_expression_values(
        table, var_name="GeneB", obs_positions=np.array([0, 1]), smoother_key="smoother_k100"
    )
    np.testing.assert_allclose(smoothed, [3.0, 3.0])  # mean of [2, 4]
    # Unsmoothed read is unchanged, and the smoother averages the FULL column before indexing.
    np.testing.assert_allclose(
        var_expression_values(table, var_name="GeneB", obs_positions=np.array([0, 1])), [2.0, 4.0]
    )
    np.testing.assert_allclose(
        var_expression_values(table, var_name="GeneB", obs_positions=np.array([1]), smoother_key="smoother_k100"), [3.0]
    )


def test_var_expression_values_rejects_unknown_smoother() -> None:
    table = _expression_table()
    with pytest.raises(ValueError, match="Smoother `nope` is not available"):
        var_expression_values(table, var_name="GeneA", obs_positions=np.array([0]), smoother_key="nope")


def test_var_expression_values_smoother_accepts_dense_operator() -> None:
    table = _expression_table()  # GeneB column = [2, 4]
    table.obsp["smoother_dense"] = np.array([[0.5, 0.5], [0.5, 0.5]], dtype="float64")
    out = var_expression_values(table, var_name="GeneB", obs_positions=np.array([0, 1]), smoother_key="smoother_dense")
    np.testing.assert_allclose(out, [3.0, 3.0])


def test_resolve_expression_matrix_x_aliases_and_layer() -> None:
    table = _expression_table()
    assert resolve_expression_matrix(table, None) is table.X
    assert resolve_expression_matrix(table, "X") is table.X
    np.testing.assert_allclose(resolve_expression_matrix(table, "log1p"), table.layers["log1p"])


def test_var_expression_values_rejects_unaligned_positions() -> None:
    table = _expression_table()
    with pytest.raises(ValueError, match="Could not align linked table rows"):
        var_expression_values(table, var_name="GeneA", obs_positions=np.array([-1, 0]), layer="log1p")


def test_module_score_values_is_mean_per_gene_population_z_score_and_subsets_last() -> None:
    matrix = np.asarray(
        [
            [1.0, 8.0, 5.0],
            [2.0, 4.0, 5.0],
            [4.0, 2.0, 5.0],
            [8.0, 1.0, 5.0],
        ]
    )
    table = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["GeneA", "GeneB", "Constant"]),
    )
    expected_full = np.column_stack(
        [
            (matrix[:, 0] - matrix[:, 0].mean()) / matrix[:, 0].std(ddof=0),
            (matrix[:, 1] - matrix[:, 1].mean()) / matrix[:, 1].std(ddof=0),
            np.zeros(table.n_obs),
        ]
    ).mean(axis=1)

    actual = module_score_values(
        table,
        var_names=("GeneA", "GeneB", "Constant"),
        obs_positions=np.asarray([3, 1]),
    )

    np.testing.assert_allclose(actual, expected_full[[3, 1]])


def test_module_score_values_reads_sparse_named_layer_without_additional_smoothing() -> None:
    import scipy.sparse as sp

    table = _expression_table()
    smoothed_layer = np.asarray([[2.0, 9.0], [6.0, 3.0]])
    table.layers["smoothed_k100"] = sp.csr_matrix(smoothed_layer)
    expected = ((smoothed_layer - smoothed_layer.mean(axis=0)) / smoothed_layer.std(axis=0)).mean(axis=1)

    actual = module_score_values(
        table,
        var_names=("GeneA", "GeneB"),
        obs_positions=np.asarray([0, 1]),
        layer="smoothed_k100",
    )

    np.testing.assert_allclose(actual, expected)


def test_module_score_values_rejects_duplicate_missing_and_unaligned_genes_or_rows() -> None:
    table = _expression_table()
    with pytest.raises(ValueError, match="duplicate module genes"):
        module_score_values(table, var_names=("GeneA", "GeneA"), obs_positions=np.asarray([0]))
    with pytest.raises(ValueError, match="not found in the selected table: Missing"):
        module_score_values(table, var_names=("GeneA", "Missing"), obs_positions=np.asarray([0]))
    with pytest.raises(ValueError, match="Could not align linked table rows"):
        module_score_values(table, var_names=("GeneA", "GeneB"), obs_positions=np.asarray([-1]))


def test_module_score_uses_umap_not_spatial_neighbors_and_standardizes_after_smoothing() -> None:
    matrix = np.asarray([[0.0, 7.0], [1.0, 1.0], [8.0, 0.0], [3.0, 2.0]])
    table = ad.AnnData(X=matrix, var=pd.DataFrame(index=["A", "B"]))
    # UMAP pairs (0, 2) and (1, 3); tissue-space pairs would instead be (0, 1), (2, 3).
    table.obsm["X_umap"] = np.array([[0, 0], [10, 0], [0.1, 0], [10.1, 0]])
    table.obsm["spatial"] = np.array([[0, 0], [0.1, 0], [10, 0], [10.1, 0]])
    table.layers["log1p"] = np.log1p(matrix)
    smoothed = table.layers["log1p"][[0, 1, 0, 1]] / 2 + table.layers["log1p"][[2, 3, 2, 3]] / 2
    expected = ((smoothed - smoothed.mean(axis=0)) / smoothed.std(axis=0)).mean(axis=1)

    actual = module_score_values(
        table, var_names=("A", "B"), obs_positions=np.array([3, 0, 2]), layer="log1p", umap_neighbors=2
    )

    np.testing.assert_allclose(actual, expected[[3, 0, 2]], atol=1e-14)
    assert set(table.obsp) == set()  # Plotting must not mutate the data/graph.
    np.testing.assert_array_equal(table.X, matrix)


def test_module_umap_smoothing_excludes_missing_coordinates_and_caps_k() -> None:
    table = ad.AnnData(
        X=np.array([[1.0, 4.0], [3.0, 2.0], [1000.0, 2000.0], [8.0, 0.0]]),
        var=pd.DataFrame(index=["A", "B"]),
    )
    table.obsm["X_umap"] = np.array([[0, 0], [1, 0], [np.nan, np.nan], [9, 0]])

    operator, valid = _umap_neighbor_operator(table, 100)
    np.testing.assert_array_equal(valid, [True, True, False, True])
    np.testing.assert_allclose(operator.sum(axis=1).A.ravel(), [1, 1, 0, 1])
    np.testing.assert_array_equal(operator[:, 2].toarray(), np.zeros((4, 1)))
    actual = module_score_values(table, var_names=("A", "B"), obs_positions=np.arange(4), umap_neighbors=100)

    assert np.isnan(actual[2])
    np.testing.assert_allclose(actual[valid], 0, atol=1e-14)


def test_module_umap_smoothing_handles_one_valid_cell_and_rejects_absent_or_all_missing_umap() -> None:
    table = _expression_table()
    with pytest.raises(ValueError, match="requires `X_umap`"):
        module_score_values(table, var_names=("GeneA", "GeneB"), obs_positions=np.arange(2), umap_neighbors=100)
    table.obsm["X_umap"] = np.array([[0, 0], [np.nan, np.nan]])
    actual = module_score_values(table, var_names=("GeneA", "GeneB"), obs_positions=np.arange(2), umap_neighbors=100)
    assert actual[0] == 0
    assert np.isnan(actual[1])
    table.obsm["X_umap"][:] = np.nan
    with pytest.raises(ValueError, match="no cells with finite coordinates"):
        module_score_values(table, var_names=("GeneA", "GeneB"), obs_positions=np.arange(2), umap_neighbors=100)


def test_module_umap_operator_includes_self_for_duplicate_coordinates() -> None:
    table = ad.AnnData(X=np.zeros((4, 2)))
    table.obsm["X_umap"] = np.zeros((4, 2))

    operator, _valid = _umap_neighbor_operator(table, 2)

    np.testing.assert_allclose(operator.diagonal(), 0.5)
    np.testing.assert_allclose(operator.sum(axis=1).A.ravel(), 1.0)


def test_module_umap_operator_rejects_excessive_neighbour_allocation(monkeypatch) -> None:
    from napari_harpy.viewer import _styling

    table = _expression_table()
    table.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr(_styling, "MAX_UMAP_NEIGHBOR_ENTRIES", 3)
    with pytest.raises(ValueError, match="neighbour-memory limit"):
        _umap_neighbor_operator(table, 2)


def test_build_styled_labels_layer_name_includes_layer() -> None:
    from napari_harpy.core._color_source import TableColorSourceSpec
    from napari_harpy.viewer.labels_styling import build_styled_labels_layer_name

    x_style = TableColorSourceSpec(table_name="t", source_kind="x_var", value_key="GeneA", value_kind="continuous")
    layer_style = TableColorSourceSpec(
        table_name="t", source_kind="x_var", value_key="GeneA", value_kind="continuous", layer="log1p"
    )
    obs_style = TableColorSourceSpec(table_name="t", source_kind="obs_column", value_key="ct", value_kind="categorical")
    smoothed_style = TableColorSourceSpec(
        table_name="t",
        source_kind="x_var",
        value_key="GeneA",
        value_kind="continuous",
        layer="log1p",
        smoother="smoother_k100",
    )
    module_style = TableColorSourceSpec(
        table_name="t",
        source_kind="x_module",
        value_key="GeneA + GeneB",
        value_kind="continuous",
        layer="smoothed_k100",
        module_genes=("GeneA", "GeneB"),
    )
    assert build_styled_labels_layer_name("cells", x_style) == "cells[X:GeneA]"
    assert build_styled_labels_layer_name("cells", layer_style) == "cells[log1p:GeneA]"
    assert build_styled_labels_layer_name("cells", obs_style) == "cells[obs:ct]"
    # A smoothed var is a distinct layer so smoothed and unsmoothed don't collide.
    assert build_styled_labels_layer_name("cells", smoothed_style) == "cells[log1p~smoother_k100:GeneA]"
    assert build_styled_labels_layer_name("cells", module_style) == "cells[smoothed_k100:module:GeneA + GeneB]"


def _colors_series_to_rgba(colors: pd.Series) -> np.ndarray:
    return np.asarray([to_rgba(color) for color in colors], dtype="float64")


def test_continuous_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    values = pd.Series([0.0, 1.0, np.nan, 0.5], index=["a", "b", "c", "d"])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_constant_values() -> None:
    values = pd.Series([2.0, 2.0, np.nan], index=[10, 11, 12])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_all_missing_values() -> None:
    values = pd.Series([np.nan, pd.NA, None], dtype="object")

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


def test_resolve_continuous_colormap_falls_back_for_none_and_invalid() -> None:
    from napari_harpy.viewer._styling import OVERLAY_CONTINUOUS_COLORMAP, resolve_continuous_colormap

    assert resolve_continuous_colormap("inferno") == "inferno"
    assert resolve_continuous_colormap(None) == OVERLAY_CONTINUOUS_COLORMAP  # unset -> viridis
    assert resolve_continuous_colormap("not_a_real_colormap") == OVERLAY_CONTINUOUS_COLORMAP  # stale preset
    assert resolve_continuous_colormap(123) == OVERLAY_CONTINUOUS_COLORMAP  # wrong type


def test_continuous_rgba_for_values_honours_the_chosen_colormap() -> None:
    from matplotlib import colormaps

    values = pd.Series([0.0, 1.0, 2.0, 3.0])
    viridis = continuous_rgba_for_values(values, colormap_name=None)  # default
    inferno = continuous_rgba_for_values(values, colormap_name="inferno")

    assert not np.allclose(viridis, inferno)  # a different colormap yields different colors
    # the max value maps to the top of the requested colormap
    np.testing.assert_allclose(inferno[-1], colormaps["inferno"](1.0))
    np.testing.assert_allclose(viridis[-1], colormaps["viridis"](1.0))
    # an unknown name degrades to viridis rather than raising
    np.testing.assert_allclose(continuous_rgba_for_values(values, colormap_name="bogus"), viridis)


def test_build_continuous_color_dict_honours_the_chosen_colormap() -> None:
    values = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    viridis = _build_continuous_color_dict(values)
    inferno = _build_continuous_color_dict(values, colormap="inferno")

    assert not np.array_equal(viridis[1], inferno[1])  # labels color_dict reflects the colormap choice
    assert viridis.keys() == inferno.keys()  # same instance ids, only colors differ


def test_categorical_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    categories = ["T", "B", "NK"]
    palette = ["#ff0000", "#00ff00", "#0000ff"]
    values = pd.Series(pd.Categorical(["T", "B", None, "NK"], categories=categories))

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_colors_for_unknown_values() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(["T", "unknown", None, "B"], dtype="object")

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_numpy_scalar_normalization() -> None:
    categories = [np.int64(1), np.int64(2)]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series([1, np.int64(2), 3, None], dtype="object")

    expected = _colors_series_to_rgba(categorical_colors_for_values(values, categories=categories, palette=palette))
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_build_continuous_label_color_dict_uses_vectorized_rgba_and_preserves_background() -> None:
    values = pd.Series([0.0, 10.0, np.nan], index=pd.Index([1, 2, 3], name="index"))

    color_dict = _build_continuous_color_dict(values)
    expected = continuous_rgba_for_values(values)

    assert color_dict[None] == "transparent"
    assert color_dict[0] == "transparent"
    np.testing.assert_allclose(color_dict[1], expected[0])
    np.testing.assert_allclose(color_dict[2], expected[1])
    np.testing.assert_allclose(color_dict[3], expected[2])


def test_build_categorical_label_color_dict_uses_vectorized_rgba_and_preserves_background() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(
        ["T", "unknown", None, "B"],
        index=pd.Index([1, 2, 3, 4], name="index"),
        dtype="object",
    )

    color_dict = _build_categorical_color_dict(values, categories=categories, palette=palette)
    expected = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert color_dict[None] == "transparent"
    assert color_dict[0] == "transparent"
    np.testing.assert_allclose(color_dict[1], expected[0])
    np.testing.assert_allclose(color_dict[2], expected[1])
    np.testing.assert_allclose(color_dict[3], expected[2])
    np.testing.assert_allclose(color_dict[4], expected[3])
