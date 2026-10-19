from __future__ import annotations

import pytest

from napari_harpy.core._color_source import (
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
    gene_module_value_key,
    parse_gene_module_text,
    validate_gene_module_genes,
)
from napari_harpy.viewer.labels_styling import LabelsStyleResult
from napari_harpy.viewer.shapes_styling import ShapesStyleResult


def test_table_color_source_spec_rejects_invalid_source_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color source kind"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="layer",
            value_key="cell_type",
            value_kind="categorical",
        )


def test_table_color_source_spec_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color value kind"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="ordinal",
        )


def test_gene_module_helpers_parse_validate_and_label_genes() -> None:
    genes = parse_gene_module_text("COL1A1, COL1A2;\nCOL3A1")

    assert genes == ("COL1A1", "COL1A2", "COL3A1")
    assert validate_gene_module_genes(genes, available_var_names=("COL1A1", "COL1A2", "COL3A1")) == genes
    assert gene_module_value_key(genes) == "COL1A1 + COL1A2 + COL3A1"


def test_gene_module_validation_rejects_too_few_duplicate_and_missing_genes() -> None:
    with pytest.raises(ValueError, match="at least two genes"):
        validate_gene_module_genes(("COL1A1",))
    with pytest.raises(ValueError, match="duplicate module genes: COL1A1"):
        validate_gene_module_genes(("COL1A1", "COL1A1"))
    with pytest.raises(ValueError, match="not found in the selected table: COL3A1"):
        validate_gene_module_genes(("COL1A1", "COL3A1"), available_var_names=("COL1A1", "COL1A2"))


def test_table_gene_module_spec_requires_continuous_score_and_two_unique_genes() -> None:
    spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_module",
        value_key="COL1A1 + COL1A2",
        value_kind="continuous",
        module_genes=["COL1A1", "COL1A2"],
    )
    assert spec.module_genes == ("COL1A1", "COL1A2")

    with pytest.raises(ValueError, match="value_kind='continuous'"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="x_module",
            value_key="COL1A1 + COL1A2",
            value_kind="categorical",
            module_genes=("COL1A1", "COL1A2"),
        )
    with pytest.raises(ValueError, match="at least two genes"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="x_module",
            value_key="COL1A1",
            value_kind="continuous",
            module_genes=("COL1A1",),
        )


def test_module_spec_canonicalizes_gene_label_and_rejects_unverified_smoother() -> None:
    spec = TableColorSourceSpec(
        table_name="t",
        source_kind="x_module",
        value_key="Wrong genes",
        value_kind="continuous",
        module_genes=("A", "B"),
        umap_neighbors=25,
    )
    assert spec.value_key == "A + B"
    with pytest.raises(ValueError, match="`smoother` is unsupported"):
        TableColorSourceSpec(
            table_name="t",
            source_kind="x_module",
            value_key="A + B",
            value_kind="continuous",
            module_genes=("A", "B"),
            smoother="smoother_spatial",
        )


def test_shape_color_source_spec_rejects_invalid_source_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color source kind"):
        ShapeColumnColorSourceSpec(
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="categorical",
        )


def test_shape_color_source_spec_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color value kind"):
        ShapeColumnColorSourceSpec(
            source_kind="shape_column",
            value_key="cell_type",
            value_kind="instance",
        )


def test_explicit_shape_source_union_covers_shape_columns_and_table_sources() -> None:
    sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = [
        ShapeColumnColorSourceSpec(
            source_kind="shape_column",
            value_key="cell_type",
            value_kind="categorical",
        ),
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="categorical",
        ),
    ]

    assert [source.value_key for source in sources] == ["cell_type", "cell_type"]


def test_labels_style_result_accepts_no_style_metadata() -> None:
    result = LabelsStyleResult(
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )

    assert result.value_kind is None


def test_labels_style_result_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color value kind"):
        LabelsStyleResult(
            value_kind="ordinal",
            palette_source=None,
            coercion_applied=False,
        )


def test_labels_style_result_rejects_invalid_palette_source() -> None:
    with pytest.raises(ValueError, match="Invalid styled palette source"):
        LabelsStyleResult(
            value_kind="categorical",
            palette_source="generated",
            coercion_applied=False,
        )


def test_shapes_style_result_rejects_invalid_palette_source() -> None:
    with pytest.raises(ValueError, match="Invalid styled palette source"):
        ShapesStyleResult(
            value_kind="categorical",
            palette_source="generated",
            coercion_applied=False,
        )


def test_shapes_style_result_accepts_no_style_metadata() -> None:
    result = ShapesStyleResult(
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )

    assert result.value_kind is None


def test_shapes_style_result_accepts_table_backed_instance_coloring_metadata() -> None:
    result = ShapesStyleResult(
        value_kind="instance",
        palette_source=None,
        coercion_applied=False,
        unannotated_source_shape_count=2,
        unannotated_rendered_shape_count=3,
    )

    assert result.value_kind == "instance"
    assert result.unannotated_source_shape_count == 2
    assert result.unannotated_rendered_shape_count == 3


def test_shapes_style_result_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color value kind"):
        ShapesStyleResult(
            value_kind="ordinal",
            palette_source=None,
            coercion_applied=False,
        )
