from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap, label_colormap

from napari_harpy.core._color_source import (
    TableColorSourceSpec,
    TableColorValueKind,
    validate_table_color_value_kind,
)
from napari_harpy.core.spatialdata import get_table, validate_table_binding
from napari_harpy.viewer._styling import (
    StyledPaletteSource,
    build_string_categorical_values,
    categorical_rgba_for_values,
    continuous_rgba_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    module_score_values,
    normalize_category_value,
    resolve_table_categorical_palette,
    validate_styled_palette_source,
    var_expression_values,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


@dataclass(frozen=True)
class LabelsStyleResult:
    """Describe labels styling metadata, if styling was applied."""

    value_kind: TableColorValueKind | None
    palette_source: StyledPaletteSource | None
    coercion_applied: bool

    def __post_init__(self) -> None:
        if self.value_kind is not None:
            validate_table_color_value_kind(self.value_kind)
        if self.palette_source is not None:
            validate_styled_palette_source(self.palette_source)


@dataclass(frozen=True)
class LabelsLoadResult(LabelsStyleResult):
    """Describe a primary or styled labels layer load/update result."""

    layer: Labels
    created: bool


def apply_table_color_source_to_labels_layer(
    layer: Labels,
    *,
    sdata: SpatialData,
    labels_name: str,
    style_spec: TableColorSourceSpec,
) -> LabelsStyleResult:
    """Apply one table-backed source to a styled labels layer."""
    table = get_table(sdata, style_spec.table_name)
    table_metadata = validate_table_binding(sdata, labels_name, style_spec.table_name)
    region_rows, obs_index = _get_region_rows_by_instance(table, table_metadata, labels_name)
    if style_spec.value_kind == "instance" or (
        style_spec.source_kind == "obs_column" and style_spec.value_key == table_metadata.instance_key
    ):
        if style_spec.source_kind != "obs_column" or style_spec.value_key != table_metadata.instance_key:
            raise ValueError(
                "Instance ID coloring must use the table instance key "
                f"`{table_metadata.instance_key}` as an observation source."
            )
        style_result, features = _build_instance_key_colormap(region_rows, instance_key=table_metadata.instance_key)
        _apply_instance_labels_colormap(layer)
    elif style_spec.source_kind == "obs_column":
        style_result, color_dict, features = _build_obs_column_colormap(
            table=table,
            region_rows=region_rows,
            column_name=style_spec.value_key,
            colormap=style_spec.colormap,
        )
        _apply_labels_colormap(layer, color_dict)
    elif style_spec.source_kind == "x_var":
        style_result, color_dict, features = _build_x_var_colormap(
            table=table,
            region_rows=region_rows,
            obs_index=obs_index,
            var_name=style_spec.value_key,
            layer=style_spec.layer,
            smoother=style_spec.smoother,
            colormap=style_spec.colormap,
        )
        _apply_labels_colormap(layer, color_dict)
    elif style_spec.source_kind == "x_module":
        style_result, color_dict, features = _build_x_module_colormap(
            table=table,
            region_rows=region_rows,
            obs_index=obs_index,
            module_genes=style_spec.module_genes,
            feature_name=style_spec.value_key,
            umap_neighbors=style_spec.umap_neighbors,
            layer=style_spec.layer,
            colormap=style_spec.colormap,
        )
        _apply_labels_colormap(layer, color_dict)
    else:  # pragma: no cover - TableColorSourceSpec validates this first.
        raise ValueError(f"Unsupported table color source kind `{style_spec.source_kind}`.")

    layer.features = _build_labels_features(features, instance_key=table_metadata.instance_key)
    return style_result


def build_styled_labels_layer_name(labels_name: str, style_spec: TableColorSourceSpec) -> str:
    """Return the user-facing layer name for one styled labels variant."""
    if style_spec.source_kind == "obs_column":
        return f"{labels_name}[obs:{style_spec.value_key}]"
    layer = style_spec.layer or "X"
    layer_label = f"{layer}~{style_spec.smoother}" if style_spec.smoother else layer
    if style_spec.source_kind == "x_module":
        if style_spec.umap_neighbors is not None:
            layer_label = f"{layer}~UMAP-k{style_spec.umap_neighbors}"
        return f"{labels_name}[{layer_label}:module:{style_spec.value_key}]"
    return f"{labels_name}[{layer_label}:{style_spec.value_key}]"


def _obs_column_renders_categorical(full_series: pd.Series) -> bool:
    """Whether an obs column renders CATEGORICALLY in the labels colormap (so its NaN cells get the flat
    grey missing colour, and hidden categories are meaningful).

    Mirrors the branch selection in :func:`_build_obs_column_colormap` EXACTLY -- keep the two in sync.
    We derive the mask's categoricity from the render path here rather than trusting
    ``style_spec.value_kind``: that is set by a slightly different classifier (and persisted verbatim in
    presets), so it can disagree with the live column dtype -- e.g. an all-0/all-1 nullable-int column is
    ``value_kind="continuous"`` but the renderer paints it categorically, greying its NaN cells.
    """
    return bool(
        _is_categorical_dtype(full_series)
        or pd.api.types.is_bool_dtype(full_series)
        or (
            pd.api.types.is_integer_dtype(full_series)
            and _has_exact_binary_zero_one_values(full_series.dropna().tolist())
        )
        or is_string_like_series(full_series)
    )


def masked_label_ids(sdata: SpatialData, labels_name: str, style_spec: TableColorSourceSpec) -> frozenset[int]:
    """Label IDs to drop to background in a styled-labels DISPLAY.

    Cells whose category is HIDDEN (alpha-0 palette entry) PLUS -- for a column that renders
    *categorically* -- cells that are UNASSIGNED (NaN category), which otherwise show the flat grey
    ``MISSING_CATEGORICAL_COLOR``. Both are "not a visible cell type", so hiding every type yields a clean
    overlay. The caller maps these IDs to background (0) in the *displayed* raster -- the only way to hide
    labels under napari's opaque blending, which ignores the colormap alpha. Display-only: the element /
    on-disk zarr is never touched. ``obs_column`` sources only; empty otherwise.
    """
    from napari_harpy.viewer.category_colors import hidden_categories

    if not isinstance(style_spec, TableColorSourceSpec) or style_spec.source_kind != "obs_column":
        return frozenset()
    table = get_table(sdata, style_spec.table_name)
    if style_spec.value_key not in getattr(table, "obs", {}):
        return frozenset()
    hidden = hidden_categories(table, style_spec.value_key)
    is_categorical = _obs_column_renders_categorical(table.obs[style_spec.value_key])
    if not hidden and not is_categorical:
        return frozenset()  # nothing hidden and no "unassigned" concept -> skip the obs scan
    table_metadata = validate_table_binding(sdata, labels_name, style_spec.table_name)
    if style_spec.value_key == table_metadata.instance_key:  # instance-id colouring has no categories
        return frozenset()
    region_rows, _ = _get_region_rows_by_instance(table, table_metadata, labels_name)
    if style_spec.value_key not in region_rows.columns:
        return frozenset()
    values = region_rows[style_spec.value_key]
    na = values.isna().to_numpy(dtype=bool)  # vectorized; runs on every categorical restyle
    masked = na.copy() if is_categorical else np.zeros(len(values), dtype=bool)  # unassigned cells
    if hidden:  # per-cell normalize only when a category is actually hidden
        masked |= np.array(
            [not is_na and normalize_category_value(value) in hidden for value, is_na in zip(values, na, strict=True)],
            dtype=bool,
        )
    return frozenset(int(instance_id) for instance_id in region_rows.index[masked])


def _build_obs_column_colormap(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    column_name: str,
    colormap: str | None = None,
) -> tuple[LabelsStyleResult, dict[int, Any], pd.DataFrame]:
    if column_name not in table.obs:
        raise ValueError(f"Observation column `{column_name}` is not available in the selected table.")

    full_series = table.obs[column_name]
    region_series = region_rows[column_name]

    if _is_categorical_dtype(full_series):
        categories = [normalize_category_value(value) for value in full_series.cat.categories]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(region_series, categories=categories, palette=palette)
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_bool_dtype(full_series):
        categories = [value for value in (False, True) if value in set(full_series.dropna().tolist())]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(region_series, categories=categories, palette=palette)
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_integer_dtype(full_series) and _has_exact_binary_zero_one_values(full_series.dropna().tolist()):
        categories = [
            value
            for value in (0, 1)
            if value in set(pd.to_numeric(full_series.dropna(), errors="coerce").astype("int64").tolist())
        ]
        numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("Int64")
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(numeric_region_series, categories=categories, palette=palette)
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)

    if is_string_like_series(full_series):
        string_region_values, categories = build_string_categorical_values(
            full_values=full_series,
            row_values=region_series,
            column_name=column_name,
        )
        palette = default_categorical_palette_for_categories(categories)
        color_dict = _build_categorical_color_dict(
            string_region_values,
            categories=categories,
            palette=palette,
        )
        # Plain string/object columns are rendered as viewer-only categorical
        # values. They do not use stored categorical palettes, so the caller
        # should report the default palette and mark that coercion happened.
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source="default_missing",
            coercion_applied=True,
        )
        return style_result, color_dict, pd.DataFrame({column_name: string_region_values}, index=region_rows.index)

    numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("float64")
    color_dict = _build_continuous_color_dict(numeric_region_series, colormap=colormap)
    style_result = LabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)


def _build_instance_key_colormap(
    region_rows: pd.DataFrame,
    *,
    instance_key: str,
) -> tuple[LabelsStyleResult, pd.DataFrame]:
    instance_values = pd.Series(
        region_rows.index.to_numpy(dtype=np.int64, copy=False),
        index=region_rows.index,
        name=instance_key,
    )
    style_result = LabelsStyleResult(
        value_kind="instance",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, pd.DataFrame({instance_key: instance_values}, index=region_rows.index)


def _build_x_var_colormap(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    obs_index: pd.Index,
    var_name: str,
    layer: str | None = None,
    smoother: str | None = None,
    colormap: str | None = None,
) -> tuple[LabelsStyleResult, dict[int, Any], pd.DataFrame]:
    numeric_values = var_expression_values(
        table,
        var_name=var_name,
        obs_positions=table.obs.index.get_indexer(obs_index),
        layer=layer,
        smoother_key=smoother,
    )
    numeric_region_series = pd.Series(
        pd.to_numeric(numeric_values, errors="coerce").astype("float64"),
        index=region_rows.index,
        name=var_name,
    )
    color_dict = _build_continuous_color_dict(numeric_region_series, colormap=colormap)
    style_result = LabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({var_name: numeric_region_series}, index=region_rows.index)


def _build_x_module_colormap(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    obs_index: pd.Index,
    module_genes: tuple[str, ...],
    feature_name: str,
    umap_neighbors: int | None = None,
    layer: str | None = None,
    colormap: str | None = None,
) -> tuple[LabelsStyleResult, dict[int, Any], pd.DataFrame]:
    numeric_values = module_score_values(
        table,
        var_names=module_genes,
        umap_neighbors=umap_neighbors,
        obs_positions=table.obs.index.get_indexer(obs_index),
        layer=layer,
    )
    numeric_region_series = pd.Series(
        numeric_values,
        index=region_rows.index,
        name=feature_name,
        dtype="float64",
    )
    color_dict = _build_continuous_color_dict(numeric_region_series, colormap=colormap)
    style_result = LabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({feature_name: numeric_region_series}, index=region_rows.index)


def _get_region_rows_by_instance(
    table: AnnData, table_metadata: Any, labels_name: str
) -> tuple[pd.DataFrame, pd.Index]:
    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == labels_name].copy()

    instance_ids = pd.to_numeric(region_rows[table_metadata.instance_key], errors="coerce")
    region_rows = region_rows.loc[instance_ids.notna()].copy()
    region_rows[table_metadata.instance_key] = instance_ids.loc[region_rows.index].astype("int64")
    region_rows = region_rows.loc[region_rows[table_metadata.instance_key] > 0].copy()
    duplicate_instance_ids = region_rows.loc[
        region_rows[table_metadata.instance_key].duplicated(keep=False),
        table_metadata.instance_key,
    ]
    if not duplicate_instance_ids.empty:
        preview = _format_instance_preview(duplicate_instance_ids.drop_duplicates().tolist())
        raise ValueError(
            f"Table `{table_metadata.table_name}` cannot be aligned to labels element `{labels_name}` because "
            f"`{table_metadata.instance_key}` contains duplicate positive label IDs after labels-specific "
            f"numeric coercion: {preview}."
        )

    aligned_obs_index = region_rows.index.copy()
    region_rows = region_rows.set_index(table_metadata.instance_key)
    region_rows.index = region_rows.index.astype("int64", copy=False)
    # napari Labels uses a feature column literally named `index` as the
    # hidden label-value -> feature-row mapping for hover/properties lookup.
    region_rows.index.name = "index"
    return region_rows, aligned_obs_index


def _build_labels_features(features: pd.DataFrame, *, instance_key: str) -> pd.DataFrame:
    features = features.copy()
    if instance_key not in features.columns:
        features.insert(0, instance_key, features.index.to_numpy(copy=False))
    return features.reset_index()


def _format_instance_preview(instance_ids: list[Any]) -> str:
    preview = ", ".join(str(instance_id) for instance_id in instance_ids[:5])
    if len(instance_ids) > 5:
        preview += ", ..."
    return preview


def _build_categorical_color_dict(
    values: pd.Series,
    *,
    categories: list[object],
    palette: list[str],
) -> dict[int | None, Any]:
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    colors = categorical_rgba_for_values(values, categories=categories, palette=palette)
    for instance_id, color in zip(values.index, colors, strict=True):
        color_dict[int(instance_id)] = color
    return color_dict


def _build_continuous_color_dict(values: pd.Series, *, colormap: str | None = None) -> dict[int | None, Any]:
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    colors = continuous_rgba_for_values(values, colormap_name=colormap)
    for instance_id, color in zip(values.index, colors, strict=True):
        color_dict[int(instance_id)] = color
    return color_dict


def _apply_labels_colormap(layer: Labels, layer_color_dict: dict[int | None, Any]) -> None:
    layer.colormap = DirectLabelColormap(color_dict=layer_color_dict, background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _apply_instance_labels_colormap(layer: Labels) -> None:
    layer.colormap = label_colormap(background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _has_exact_binary_zero_one_values(values: list[object]) -> bool:
    if not values:
        return False
    normalized_values = {int(value) for value in pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1} or normalized_values == {0} or normalized_values == {1}
