"""Elo rating calculation module."""

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import TypedDict

import optuna
import optuna.visualization as vis
import polars as pl
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss

from march_madness.loader import DataLoader
from march_madness.settings import OUTPUT_DIR


@dataclass(frozen=True, slots=True)
class Game:
    """Data container representing a single game."""

    game_id: int
    team1_id: int
    team2_id: int
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    team1_home: bool
    team2_home: bool
    season: int


class EloData(TypedDict):
    """Data container representing the result of an Elo calculation."""

    game_id: int
    season: int
    team1_id: int
    team2_id: int
    team1_elo_before_match: float
    team2_elo_before_match: float
    team1_adjusted_elo_before_match: float
    team2_adjusted_elo_before_match: float
    team1_expected: float
    team2_expected: float
    team1_elo_after_match: float
    team2_elo_after_match: float
    team1_hca: float
    team2_hca: float
    elo_advantage: float


@dataclass(frozen=True, slots=True)
class EloResults:
    """Data container representing the results of Elo calculations for a season."""

    game_ratings: pl.DataFrame
    final_ratings: pl.DataFrame


def expected_score(
    *,
    team_rating: float,
    opponent_rating: float,
    scale_factor: float,
    hca: float = 0,
) -> float:
    """Returns the expected score for a team given its rating and the opponent's rating."""
    team_rating += hca
    return 1 / (1 + 10 ** ((opponent_rating - team_rating) / scale_factor))


def generate_games(frame: pl.DataFrame) -> Generator[Game, None, None]:
    """Generates Game objects from a DataFrame containing game information."""
    for row in frame.sort("ID").iter_rows(named=True):
        yield Game(
            game_id=row["ID"],
            team1_id=row["team1_id"],
            team2_id=row["team2_id"],
            team1_name=row["team1_name"],
            team2_name=row["team2_name"],
            team1_score=row["team1_score"],
            team2_score=row["team2_score"],
            team1_home=bool(row["team1_home"]),
            team2_home=bool(row["team2_home"]),
            season=row["season"],
        )


def calculate_margin_multiplier(
    margin: int,
    *,
    decay: float = 0.8,
    shift: float = 3.0,
    denominator_baseline: float = 7.5,
    advantage_scale: float = 0.006,
    elo_advantage: float = 0,
    max_upset_boost: float = 3.0,
) -> float:
    """Returns a multiplier based on the margin of victory.

    Parameters
    ----------
    margin : int
        Margin of victory for the game.
    decay : float, optional
        Controls diminishing returns.
        An exponent of 1 makes the numerator grow linearly with the margin.
        An exponent of less than 1 makes it grow sublinearly with the margin.
        An exponent of greater than 1 makes it grow superlinearly with the margin.
    shift : float, optional
        Shifts the margin to influence how aggressively the multiplier responds
        to small margins relative to larger ones.
    denominator_baseline : float, optional
        A scaling factor in the denominator to control the overall magnitude of the multiplier.
    advantage_scale : float, optional
        Scales the impact of the Elo advantage in the denominator.
        This value should include the home-court advantage if applicable.
    max_upset_boost : float, optional
        Maximum factor by which the denominator can be reduced to boost the impact of upsets.

    Returns
    -------
    float
        Multiplier based on the margin of victory.
    """
    numerator = (abs(margin) + shift) ** decay
    denominator = max(
        denominator_baseline + advantage_scale * elo_advantage,
        denominator_baseline / max_upset_boost,
    )
    return numerator / denominator


def apply_season_carryover(
    elo_ratings: dict[int, float],
    season_carryover: float,
    starting_rating: float,
) -> dict[int, float]:
    """Applies season carryover to Elo ratings, adjusting them towards the baseline rating."""
    baseline_rating = sum(elo_ratings.values()) / len(elo_ratings) if elo_ratings else starting_rating
    new_elo_ratings: dict[int, float] = {}
    for team, rating in elo_ratings.items():
        new_elo_ratings[team] = baseline_rating + season_carryover * (rating - baseline_rating)
    return new_elo_ratings


def objective(
    trial: optuna.trial.Trial,
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
) -> None:
    # previous best loss - 0.5492583457814224
    params = {
        "starting_rating": 0,  # hard coded - free parameter
        "scale_factor": 400,  # hard coded - establishes rating units
        "denominator_baseline": 7.5,  # hard coded - establishes baseline for denominator
        "k_factor": trial.suggest_float("k_factor", 8.0, 64.0, log=True),  # result sensitivity to match outcome
        "season_carryover": trial.suggest_float("season_carryover", 0.5, 1.0),
        "hca": trial.suggest_float("hca", 0.0, 150),  # 0 is ~50%, 150 is ~70%
        "decay": trial.suggest_float("decay", 0.5, 1.0),  # explore sublinear growth
        "shift": trial.suggest_float("shift", 2.0, 8.0),  # 0 makes close wins more impactful
        "advantage_scale": trial.suggest_float("advantage_scale", 0.0, 0.01),  # 0 disables the correction
        "max_upset_boost": trial.suggest_float("max_upset_boost", 1.0, 5.0),  # no upset to a big ceiling
    }
    elo_results = get_elo_ratings(train_frame, **params)
    final_ratings = elo_results.final_ratings
    predictions = (
        test_frame.join(
            final_ratings.select(
                team1_id="team_id",
                team1_elo="elo",
                season="season",
            ),
            how="left",
            on=["team1_id", "season"],
        )
        .join(
            final_ratings.select(
                team2_id="team_id",
                team2_elo="elo",
                season="season",
            ),
            how="left",
            on=["team2_id", "season"],
        )
        .with_columns(
            # Assume no home court advantage occurs in the tournament
            # TODO: this assumption isn't true for women - need to make sure data captures this though
            team1_win_prob=1 / (1 + 10 ** ((pl.col("team2_elo") - (pl.col("team1_elo"))) / params["scale_factor"]))
        )
    )
    return log_loss(
        y_true=predictions["team1_win"],
        y_proba=predictions["team1_win_prob"],
        labels=[0, 1],
    )


def run_elo(
    league: str,
    *,
    tune: bool = True,
    n_trials: int = 100,
    starting_rating: int = 0,
    scale_factor: float = 400,
    k_factor: float = 32,
    season_carryover: float = 0.50,
    hca: float = 0.0,
    decay: float = 0.8,
    shift: float = 3.0,
    denominator_baseline: float = 7.5,
    advantage_scale: float = 0.006,
    max_upset_boost: float = 3.0,
    save: bool = True,
):
    data_constructor = DataLoader(league=league)
    frame = data_constructor.load_game_box_scores().filter(
        # Exclude secondary tournament games from training and prediction
        pl.col("secondary_tourney").is_null(),
    )
    train_frame = frame.filter(pl.col("is_ncaa_tourney").eq(0))
    test_frame = frame.filter(pl.col("is_ncaa_tourney").eq(1))

    save_dir = OUTPUT_DIR / league / "elo"
    save_dir.mkdir(parents=True, exist_ok=True)

    if tune:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"elo_{league}",
            storage="sqlite:///data/optuna.sqlite3",
            load_if_exists=True,
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )
        study.optimize(
            lambda trial: objective(trial, train_frame, test_frame),
            n_trials=n_trials,
        )

        best_params = dict(study.best_params)

        # Objective value over successive trials
        vis.plot_optimization_history(study).show()

        # Estimated importance of each hyperparameter
        vis.plot_param_importances(study).show()

        # Objective value vs. individual hyperparameters
        vis.plot_slice(study).show()

        # TODO: add logic/check to only overwrite best parameters if they improve performance
        with (save_dir / "best_params.json").open("w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)

        # Get Elo ratings using the best parameters from tuning
        elo_results = get_elo_ratings(train_frame, **best_params)

    else:
        elo_results = get_elo_ratings(
            train_frame,
            starting_rating=starting_rating,
            scale_factor=scale_factor,
            k_factor=k_factor,
            season_carryover=season_carryover,
            hca=hca,
            decay=decay,
            shift=shift,
            denominator_baseline=denominator_baseline,
            advantage_scale=advantage_scale,
            max_upset_boost=max_upset_boost,
        )

    if save:
        elo_results.game_ratings.write_csv(save_dir / "game_ratings.csv")
        elo_results.final_ratings.write_csv(save_dir / "final_ratings.csv")


def get_elo_ratings(
    frame: pl.DataFrame,
    *,
    starting_rating: int = 0,
    scale_factor: float = 400,
    k_factor: float = 32,
    season_carryover: float = 0.50,
    hca: float = 0.0,
    decay: float = 0.8,
    shift: float = 3.0,
    denominator_baseline: float = 7.5,
    advantage_scale: float = 0.006,
    max_upset_boost: float = 3.0,
) -> None:
    all_team_ids = set(frame["team1_id"].to_list() + frame["team2_id"].to_list())
    elo_ratings = dict.fromkeys(all_team_ids, starting_rating)
    elo_results: list[EloData] = []
    most_recent_season: int | None = None

    for game in generate_games(frame):
        current_season = game.season

        if most_recent_season is not None and current_season != most_recent_season:
            elo_ratings = apply_season_carryover(
                elo_ratings=elo_ratings,
                season_carryover=season_carryover,
                starting_rating=starting_rating,
            )

        most_recent_season = current_season
        team1_rating = elo_ratings[game.team1_id]
        team2_rating = elo_ratings[game.team2_id]

        if game.team1_home:
            team1_hca = hca
            team2_hca = 0

        elif game.team2_home:
            team1_hca = 0
            team2_hca = hca
        else:
            team1_hca = 0
            team2_hca = 0

        # Use these adjustments for the elo advantage calculation for margin multiplier
        # Let the expected score calculations use the original ratings with HCA adjustments separately
        team1_adjusted_rating = team1_rating + team1_hca
        team2_adjusted_rating = team2_rating + team2_hca

        expected_team1 = expected_score(
            team_rating=team1_adjusted_rating,
            opponent_rating=team2_adjusted_rating,
            scale_factor=scale_factor,
        )
        expected_team2 = 1 - expected_team1

        # Positive value indicates team1 won
        margin = game.team1_score - game.team2_score

        if margin > 0:
            actual_team1 = 1
            actual_team2 = 0
            winner_adjusted_rating = team1_adjusted_rating
            loser_adjusted_rating = team2_adjusted_rating
        elif margin < 0:
            actual_team1 = 0
            actual_team2 = 1
            winner_adjusted_rating = team2_adjusted_rating
            loser_adjusted_rating = team1_adjusted_rating
        else:
            actual_team1 = 0.5
            actual_team2 = 0.5
            winner_adjusted_rating = 0
            loser_adjusted_rating = 0

        elo_advantage = winner_adjusted_rating - loser_adjusted_rating
        margin_multiplier = calculate_margin_multiplier(
            margin=margin,
            decay=decay,
            shift=shift,
            denominator_baseline=denominator_baseline,
            advantage_scale=advantage_scale,
            elo_advantage=elo_advantage,
            max_upset_boost=max_upset_boost,
        )

        elo_ratings[game.team1_id] += k_factor * (actual_team1 - expected_team1) * margin_multiplier
        elo_ratings[game.team2_id] += k_factor * (actual_team2 - expected_team2) * margin_multiplier

        elo_results.append(
            EloData(
                game_id=game.game_id,
                season=game.season,
                team1_result=actual_team1,
                team2_result=actual_team2,
                team1_id=game.team1_id,
                team2_id=game.team2_id,
                team1_name=game.team1_name,
                team2_name=game.team2_name,
                team1_elo_before_match=team1_rating,
                team2_elo_before_match=team2_rating,
                team1_adjusted_elo_before_match=team1_adjusted_rating,
                team2_adjusted_elo_before_match=team2_adjusted_rating,
                team1_expected=expected_team1,
                team2_expected=expected_team2,
                team1_elo_after_match=elo_ratings[game.team1_id],
                team2_elo_after_match=elo_ratings[game.team2_id],
                team1_hca=team1_hca,
                team2_hca=team2_hca,
                elo_advantage=elo_advantage,
            )
        )

    game_ratings = pl.DataFrame(
        elo_results,
        schema={
            "game_id": pl.Int64,
            "season": pl.Int64,
            "team1_id": pl.Int64,
            "team2_id": pl.Int64,
            "team1_name": pl.Utf8,
            "team2_name": pl.Utf8,
            "team1_result": pl.Float64,
            "team2_result": pl.Float64,
            "team1_elo_before_match": pl.Float64,
            "team2_elo_before_match": pl.Float64,
            "team1_expected": pl.Float64,
            "team2_expected": pl.Float64,
            "team1_elo_after_match": pl.Float64,
            "team2_elo_after_match": pl.Float64,
            "team1_adjusted_elo_before_match": pl.Float64,
            "team2_adjusted_elo_before_match": pl.Float64,
            "team1_hca": pl.Float64,
            "team2_hca": pl.Float64,
            "elo_advantage": pl.Float64,
        },
    )
    final_ratings = (
        pl.concat(
            [
                game_ratings.select(
                    game_id="game_id",
                    season="season",
                    team_id="team1_id",
                    team_name="team1_name",
                    elo="team1_elo_after_match",
                ),
                game_ratings.select(
                    game_id="game_id",
                    season="season",
                    team_id="team2_id",
                    team_name="team2_name",
                    elo="team2_elo_after_match",
                ),
            ]
        )
        .sort("game_id")
        .group_by("season", "team_id", "team_name")
        .agg(pl.col("elo").last())
        .sort("season", "team_id")
    )
    return EloResults(
        game_ratings=game_ratings,
        final_ratings=final_ratings,
    )
