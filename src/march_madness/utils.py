"""Module utilities."""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
from pydantic import BaseModel

from march_madness.settings import OUTPUT_DIR


class SummaryStats(BaseModel):
    mean: list[float] | list[list[float]]
    std: list[float] | list[list[float]]
    percentile_025: list[float] | list[list[float]]
    percentile_500: list[float] | list[list[float]]
    percentile_975: list[float] | list[list[float]]

    def get_index(self, index: int) -> "SummaryStats":
        """Get summary statistics for a specific index."""
        return SummaryStats(
            mean=self.mean[index],
            std=self.std[index],
            percentile_025=self.percentile_025[index],
            percentile_500=self.percentile_500[index],
            percentile_975=self.percentile_975[index],
        )


def summarize_samples(site_samples: dict[str, Any]) -> SummaryStats:
    """Generates summary statistics for the samples for 0/1-dimensional sites."""
    if np.ndim(site_samples) == 1:
        site_samples = site_samples.reshape(-1, 1)  # Ensure arr is 2D for quantile calculation
    return SummaryStats(
        mean=np.mean(site_samples, axis=0).tolist(),
        std=np.std(site_samples, axis=0).tolist(),
        percentile_025=np.percentile(site_samples, 2.5, axis=0).tolist(),
        percentile_500=np.percentile(site_samples, 50, axis=0).tolist(),
        percentile_975=np.percentile(site_samples, 97.5, axis=0).tolist(),
    )


def get_win_probs(predictive: dict[str, Any], team_str: str, opp_team_str: str) -> pl.DataFrame:
    return pl.DataFrame(
        data=np.array((predictive[f"{team_str}_score"] > predictive[f"{opp_team_str}_score"]).mean(axis=0)),
        schema=[f"{team_str}_win_prob"],
    )


def get_quantiles(arr: jnp.ndarray, col_name: str = "value") -> pl.DataFrame:
    if np.ndim(arr) == 1:
        arr = arr.reshape(-1, 1)  # Ensure arr is 2D for quantile calculation
    return pl.DataFrame(
        data=np.quantile(arr, q=[0.025, 0.5, 0.975], axis=0),
        schema=[f"{col_name}_025", f"{col_name}_50", f"{col_name}_975"],
    )


def load_best_params(league: str, model: str) -> dict:
    with Path(OUTPUT_DIR / f"{league}/{model}/best_params.json").open("r") as f:
        return json.load(f)
