"""Safe partition loading utilities enforcing Tier 3 holdout isolation.

Exposes ``load_dev_partition`` to load only Tier 1 (warm-up) and Tier 2
(development) rows from ``partition_masks.parquet``, asserting exact
partition counts (400 / 750 / 5 / 300) and preventing accidental Tier 3
holdout access unless ``allow_holdout=True`` is explicitly passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

EXPECTED_TIER_COUNTS = {
    "tier1_warmup": 400,
    "tier2_dev": 750,
    "embargo_boundary": 5,
    "tier3_holdout": 300,
}
REQUIRED_PARTITION_COLUMNS = tuple(EXPECTED_TIER_COUNTS.keys())


def validate_partition_masks(masks_df: pd.DataFrame) -> None:
    """Validate structural invariants of a 3-Tier partition mask DataFrame."""
    missing_cols = [c for c in REQUIRED_PARTITION_COLUMNS if c not in masks_df.columns]
    if missing_cols:
        raise ValueError(f"partition_masks missing required columns: {missing_cols}")

    if not masks_df.index.is_monotonic_increasing:
        raise ValueError("partition_masks index must be monotonically increasing.")
    if masks_df.index.has_duplicates:
        raise ValueError("partition_masks index must not contain duplicates.")

    for col, expected_count in EXPECTED_TIER_COUNTS.items():
        actual_count = int(masks_df[col].astype(bool).sum())
        if actual_count != expected_count:
            raise AssertionError(
                f"Partition '{col}' count mismatch: expected {expected_count}, got {actual_count}."
            )

    tier_sum = sum(masks_df[col].astype(int) for col in REQUIRED_PARTITION_COLUMNS)
    if not (tier_sum == 1).all():
        raise AssertionError("Partition masks must be mutually exclusive and cover every row exactly once.")


def load_dev_partition(
    masks_source: Union[Path, str, pd.DataFrame],
    features_source: Optional[Union[Path, str, pd.DataFrame]] = None,
    *,
    include_tier1: bool = True,
    include_tier2: bool = True,
    include_tier3: bool = False,
    allow_holdout: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load Tier 1 / Tier 2 partition rows while guarding Tier 3 holdout access.

    Parameters
    ----------
    masks_source : Path | str | pd.DataFrame
        Path to ``partition_masks.parquet`` or an in-memory masks DataFrame.
    features_source : Optional[Path | str | pd.DataFrame]
        Optional path to ``market_features_5y.parquet`` or an in-memory DataFrame.
    include_tier1 : bool
        Include Tier 1 warm-up rows (default True).
    include_tier2 : bool
        Include Tier 2 development rows (default True).
    include_tier3 : bool
        Request Tier 3 holdout rows (default False). Raises ``PermissionError``
        unless ``allow_holdout=True`` is explicitly provided.
    allow_holdout : bool
        Explicit safety override required when ``include_tier3=True``.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        Filtered ``masks_df`` if ``features_source`` is None, otherwise
        ``(features_df_filtered, masks_df_filtered)``.
    """
    if include_tier3 and not allow_holdout:
        raise PermissionError(
            "Access to Tier 3 holdout partition is blocked by default. "
            "Pass allow_holdout=True explicitly only for authorized one-shot holdout evaluation."
        )

    masks_df = (
        masks_source.copy()
        if isinstance(masks_source, pd.DataFrame)
        else pd.read_parquet(masks_source)
    )
    validate_partition_masks(masks_df)

    mask = pd.Series(False, index=masks_df.index)
    if include_tier1:
        mask = mask | masks_df["tier1_warmup"].astype(bool)
    if include_tier2:
        mask = mask | masks_df["tier2_dev"].astype(bool)
    if include_tier3 and allow_holdout:
        mask = mask | masks_df["tier3_holdout"].astype(bool)

    filtered_masks = masks_df.loc[mask].copy()

    if features_source is None:
        return filtered_masks

    features_df = (
        features_source.copy()
        if isinstance(features_source, pd.DataFrame)
        else pd.read_parquet(features_source)
    )
    if not features_df.index.equals(masks_df.index):
        raise ValueError("features_df index does not match partition_masks index.")

    return features_df.loc[mask].copy(), filtered_masks
