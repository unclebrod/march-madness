import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl

from march_madness.settings import (
    DATA_DIR,
    FINAL_FOUR_REGION_SETTINGS,
    OUTPUT_DIR,
    WITHIN_REGION_STANDARD_SEED_SETTINGS,
)

logger = logging.getLogger(__name__)

pd.set_option("future.no_silent_downcasting", True)  # noqa

RECENCY_PADDING = 40
EARTH_RADIUS_MILES = 3959.0

GAME_RENAMER = {
    "Season": "season",
    "DayNum": "day_num",
    "NumOT": "num_ot",
    "WTeamID": "team1_id",
    "WScore": "team1_score",
    "WLoc": "team1_loc",
    "WFGM": "team1_fgm",
    "WFGA": "team1_fga",
    "WFGM3": "team1_fg3m",
    "WFGA3": "team1_fg3a",
    "WFTM": "team1_ftm",
    "WFTA": "team1_fta",
    "WOR": "team1_orb",
    "WDR": "team1_drb",
    "WAst": "team1_ast",
    "WTO": "team1_tov",
    "WStl": "team1_stl",
    "WBlk": "team1_blk",
    "WPF": "team1_pf",
    "LTeamID": "team2_id",
    "LScore": "team2_score",
    "LFGM": "team2_fgm",
    "LFGA": "team2_fga",
    "LFGM3": "team2_fg3m",
    "LFGA3": "team2_fg3a",
    "LFTM": "team2_ftm",
    "LFTA": "team2_fta",
    "LOR": "team2_orb",
    "LDR": "team2_drb",
    "LAst": "team2_ast",
    "LTO": "team2_tov",
    "LStl": "team2_stl",
    "LBlk": "team2_blk",
    "LPF": "team2_pf",
    "SecondaryTourney": "secondary_tourney",
}


def _poss_expr(team_str: str, multiplier: float = 0.475) -> pl.Expr:
    """Expression for calculating an estimated number of possessions."""
    return (
        pl.col(f"{team_str}_fga")
        .sub(pl.col(f"{team_str}_orb"))
        .add(pl.col(f"{team_str}_tov"))
        .add(pl.col(f"{team_str}_fta").mul(pl.lit(multiplier)))
    )


def _fill_null_location(
    null_col: str,
    team1_loc_col: str = "team1_loc",
) -> pl.Expr:
    """Expression to fill in null location data on home/away status."""
    return (
        pl.when(pl.col(null_col).is_null())
        .then(
            pl.when(pl.col(team1_loc_col).eq("H"))
            .then(pl.col(f"team1_{null_col}"))
            .when(pl.col(team1_loc_col).eq("A"))
            .then(pl.col(f"team2_{null_col}"))
        )
        .otherwise(pl.col(null_col))
    )


def haversine(
    lat1: str,
    lon1: str,
    lat2: str,
    lon2: str,
) -> pl.Expr:
    """Vectorized Haversine distance (miles) as a Polars expression."""
    lat1_rad = pl.col(lat1).radians()
    lon1_rad = pl.col(lon1).radians()
    lat2_rad = pl.col(lat2).radians()
    lon2_rad = pl.col(lon2).radians()

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (dlat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)

    c = 2 * pl.arctan2(a.sqrt(), (1 - a).sqrt())

    return EARTH_RADIUS_MILES * c


def derive_team_locations(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive team locations based on game locations."""
    team_city_frame = pl.concat(
        [
            frame.select(
                team_id=pl.col("team1_id"),
                team_loc=pl.col("team1_loc"),
                season=pl.col("season"),
                city=pl.col("city"),
                state=pl.col("state"),
            ),
            frame.select(
                team_id=pl.col("team2_id"),
                team_loc=pl.col("team2_loc"),
                season=pl.col("season"),
                city=pl.col("city"),
                state=pl.col("state"),
            ),
        ]
    )
    # These are teams where we can determine the location based on home games
    existing_team_locations = (
        team_city_frame.filter(
            pl.col("team_loc").eq("H"),
            pl.col("city").is_not_null(),
        )
        .group_by("team_id", "season")
        .agg(pl.struct("city", "state").mode().sort().first().alias("location"))
    )
    # There are some teams for which we cannot determine the location within a season
    # We identify them first
    missing_team_locations = (
        team_city_frame.select(["season", "team_id"])
        .unique()
        .join(
            existing_team_locations.select(["season", "team_id"]).unique(),
            on=["season", "team_id"],
            how="anti",
        )
    )
    # We combine them, and then backfill the past team years with future ones
    # Helps pick up more teams than you would initially htink
    team_locations = (
        pl.concat([existing_team_locations, missing_team_locations], how="diagonal_relaxed")
        .sort(["season", "team_id"])
        .with_columns(pl.col("location").backward_fill().over(["team_id"]))
        .unnest("location")
    )
    cities = frame.select(["city", "state", "lat", "lng", "elevation"]).unique().drop_nulls()
    return team_locations.join(cities, on=["city", "state"], how="left")


def derive_rest_days(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive rest days for teams based on game dates."""
    return (
        pl.concat(
            [
                frame.select(
                    ID=pl.col("ID"),
                    season=pl.col("season"),
                    team_id=pl.col("team1_id"),
                    days_into_season=pl.col("days_into_season"),
                ),
                frame.select(
                    ID=pl.col("ID"),
                    season=pl.col("season"),
                    team_id=pl.col("team2_id"),
                    days_into_season=pl.col("days_into_season"),
                ),
            ]
        )
        .sort(["season", "team_id", "days_into_season"])
        .with_columns(
            **{f"days_rest_{x}": pl.col("days_into_season").diff(x).over(["team_id", "season"]) for x in range(1, 4)}
        )
        .with_columns(
            pl.col("days_rest_1").lt(1.5).alias("b2b").cast(pl.Int8).fill_null(0),
            pl.col("days_rest_1").lt(2.5).alias("2in3").cast(pl.Int8).fill_null(0),
            pl.col("days_rest_2").lt(3.5).alias("3in4").cast(pl.Int8).fill_null(0),
            pl.col("days_rest_3").lt(4.5).alias("4in5").cast(pl.Int8).fill_null(0),
        )
        .drop("days_into_season", *[f"days_rest_{x}" for x in range(1, 4)])
        .sort(["ID"])
    )


def build_team_game_box_scores(frame: pl.DataFrame) -> pl.DataFrame:
    """Create both team perspectives, swapping team columns and reversing signed features."""
    rename_dict = {
        col: f"team2_{col.removeprefix('team1_')}"
        if col.startswith("team1_")
        else f"team1_{col.removeprefix('team2_')}"
        for col in frame.columns
        if col.startswith(("team1_", "team2_"))
    }
    # Paired team features swap above; these unpaired features are relative to team 1.
    signed_columns = {
        "spread",
        "b2b_diff",
        "2in3_diff",
        "3in4_diff",
        "4in5_diff",
        "fatigue_score_diff",
        "fatigue_effect_1",
        "fatigue_effect_2",
        "fatigue_effect_3",
        "fatigue_effect_4",
    }
    swapped = (
        frame.with_columns(-pl.col(col) for col in frame.columns if col in signed_columns)
        .rename(rename_dict)
        .select(frame.columns)
    )
    return pl.concat([frame, swapped], how="vertical").sort("season", "ID", "day_num", "team1_id")


def generate_ncaaw_homecourt() -> pl.DataFrame:
    """Generate NCAAW tournament home court advantage dataframe."""
    regions = ["W", "X", "Y", "Z"]
    seeds = list(range(1, 17))  # Seeds 1-16

    matchups = []

    # First round matchups
    first_round_winners = {}  # Store possible winners for R2 matchups
    for region in regions:
        first_round_winners[region] = []
        for i in range(8):  # 1 vs 16, 2 vs 15, ..., 8 vs 9
            high_seed = seeds[i]
            low_seed = seeds[-(i + 1)]
            slot = f"R1{region}{i + 1}"
            team_a = f"{region}{high_seed:02d}"
            team_b = f"{region}{low_seed:02d}"
            has_home_court = int(high_seed <= 4)  # Top 4 seeds host

            matchups.append([slot, team_a, team_b, has_home_court])
            first_round_winners[region].append((team_a, team_b))  # Save for R2

    # Mapping from R2 slot -> host seed
    # (1v16 vs 8v9 hosted by 1, etc.)
    r2_host_seeds = [1, 2, 3, 4]

    # Second round matchups (explicitly listing all combinations)
    for region in regions:
        for i in range(4):  # Each R2 slot is winner of (1 vs 16) vs winner of (8 vs 9), etc.
            slot = f"R2{region}{i + 1}"

            team_a_options = first_round_winners[region][i]  # Winner of high-seed game
            team_b_options = first_round_winners[region][8 - i - 1]  # Winner of low-seed game

            host_seed = r2_host_seeds[i]

            for team_a in team_a_options:
                for team_b in team_b_options:
                    seed_a = int(team_a[1:])
                    seed_b = int(team_b[1:])

                    # Home court only exists if the host seed survives
                    has_home_court = int(host_seed in {seed_a, seed_b})

                    matchups.append([slot, team_a, team_b, has_home_court])

    return pl.DataFrame(
        matchups,
        schema=["Slot", "StrongSeed", "WeakSeed", "team1_home"],
        orient="row",
    )


def current_season() -> int:
    """Get the current NCAA season based on today's date."""
    today = datetime.now(ZoneInfo("America/New_York"))
    year = today.year
    if today.month >= 11:  # November or later
        return year + 1
    return year


@dataclass(frozen=True, slots=True)
class FileConfig:
    """File configuration convenience class."""

    name: str
    requires_league: bool
    men_only: bool = False


class Files:
    """Container for keeping track of file loading configurations."""

    cities = FileConfig(name="Cities", requires_league=False)
    cities_geocoded = FileConfig(name="CitiesGeocoded", requires_league=False)
    conferences = FileConfig(name="Conferences", requires_league=False)
    conference_tourney_games = FileConfig(name="ConferenceTourneyGames", requires_league=True)
    game_cities = FileConfig(name="GameCities", requires_league=True)
    massey_ordinals = FileConfig(name="MasseyOrdinals", requires_league=True, men_only=True)
    ncaa_tourney_compact_results = FileConfig(name="NCAATourneyCompactResults", requires_league=True)
    ncaa_tourney_detailed_results = FileConfig(name="NCAATourneyDetailedResults", requires_league=True)
    ncaa_tourney_seed_round_slots = FileConfig(name="NCAATourneySeedRoundSlots", requires_league=True, men_only=True)
    ncaa_tourney_seeds = FileConfig(name="NCAATourneySeeds", requires_league=True)
    ncaa_tourney_slots = FileConfig(name="NCAATourneySlots", requires_league=True)
    regular_season_compact_results = FileConfig(name="RegularSeasonCompactResults", requires_league=True)
    regular_season_detailed_results = FileConfig(name="RegularSeasonDetailedResults", requires_league=True)
    seasons = FileConfig(name="Seasons", requires_league=True)
    secondary_tourney_compact_results = FileConfig(name="SecondaryTourneyCompactResults", requires_league=True)
    secondary_tourney_teams = FileConfig(name="SecondaryTourneyTeams", requires_league=True)
    team_coaches = FileConfig(name="TeamCoaches", requires_league=True, men_only=True)
    team_conferences = FileConfig(name="TeamConferences", requires_league=True)
    teams = FileConfig(name="Teams", requires_league=True)
    team_spellings = FileConfig(name="TeamSpellings", requires_league=True)
    sample_submissions_stage_1 = FileConfig(name="SampleSubmissionStage1", requires_league=False)
    sample_submissions_stage_2 = FileConfig(name="SampleSubmissionStage2", requires_league=False)
    seed_benchmark_stage_1 = FileConfig(name="SeedBenchmarkStage1", requires_league=False)


@dataclass(frozen=True, slots=True)
class DataLoader:
    """Class responsible for loading and constructing data for use downstream."""

    league: Literal["M", "W"] = "M"
    data_dir: Path = DATA_DIR

    def read(self, config: FileConfig) -> pl.DataFrame:
        if self.league != "M" and config.men_only:
            raise ValueError("The selected data is only available for men.")
        prefix = self.league if config.requires_league else ""
        loc = self.data_dir / f"{prefix}{config.name}.csv"
        logger.info("Reading data from %s...", loc)
        return pl.read_csv(loc)

    def load_game_box_scores(self) -> pl.DataFrame:
        """Return box score and other game-level information with one row per game."""
        logger.info("Creating game team box scores...")
        # Load the files up front to make easy references during joins
        teams = self.read(Files.teams)
        regular_season_results = self.read(Files.regular_season_detailed_results)
        playoff_results = self.read(Files.ncaa_tourney_detailed_results)
        secondary_results = self.read(Files.secondary_tourney_compact_results)
        conferences = self.read(Files.team_conferences)
        conference_tourney_games = self.read(Files.conference_tourney_games)
        seasons = self.read(Files.seasons)
        game_cities = self.read(Files.game_cities)
        # Geocoded city information is created via `geocoding.py`
        cities = self.read(Files.cities_geocoded)

        # Join and rename columns for personal preference and consistency
        frame = (
            pl.concat(
                [
                    regular_season_results.with_columns(
                        is_ncaa_tourney=pl.lit(value=False),
                        is_secondary_tourney=pl.lit(value=False),
                    ),
                    playoff_results.with_columns(
                        is_ncaa_tourney=pl.lit(value=True),
                        is_secondary_tourney=pl.lit(value=False),
                    ),
                    # The secondary tournament is not fully detailed like the other game sources
                    secondary_results.with_columns(
                        is_ncaa_tourney=pl.lit(value=False),
                        is_secondary_tourney=pl.lit(value=True),
                    ),
                ],
                how="diagonal_relaxed",
            )
            .rename(GAME_RENAMER)
            .join(
                conferences.rename(
                    {
                        "ConfAbbrev": "team1_conf",
                        "TeamID": "team1_id",
                        "Season": "season",
                    }
                ),
                how="left",
                on=["season", "team1_id"],
            )
            .join(
                conferences.rename(
                    {
                        "ConfAbbrev": "team2_conf",
                        "TeamID": "team2_id",
                        "Season": "season",
                    }
                ),
                how="left",
                on=["season", "team2_id"],
            )
            .join(
                teams.rename(
                    {
                        "TeamID": "team1_id",
                        "TeamName": "team1_name",
                        "FirstD1Season": "team1_first_d1_season",
                        "LastD1Season": "team1_last_d1_season",
                    },
                    # Women's doesn't have first/last D1 season data
                    strict=False,
                ),
                how="left",
                on=["team1_id"],
            )
            .join(
                teams.rename(
                    {
                        "TeamID": "team2_id",
                        "TeamName": "team2_name",
                        "FirstD1Season": "team2_first_d1_season",
                        "LastD1Season": "team2_last_d1_season",
                    },
                    # Women's doesn't have first/last D1 season data
                    strict=False,
                ),
                how="left",
                on=["team2_id"],
            )
            .join(
                conference_tourney_games.rename(
                    {
                        "ConfAbbrev": "conf_tourney_abbr",
                        "Season": "season",
                        "DayNum": "day_num",
                        "WTeamID": "team1_id",
                        "LTeamID": "team2_id",
                    }
                ),
                how="left",
                on=["season", "day_num", "team1_id", "team2_id"],
            )
            .join(
                seasons.rename(
                    {
                        "Season": "season",
                        "DayZero": "day_zero",
                        "RegionW": "region_w",
                        "RegionX": "region_x",
                        "RegionY": "region_y",
                        "RegionZ": "region_z",
                    }
                ),
                on=["season"],
                how="left",
            )
            .join(
                game_cities.rename(
                    {
                        "CRType": "cr_type",
                        "CityID": "city_id",
                        "Season": "season",
                        "DayNum": "day_num",
                        "WTeamID": "team1_id",
                        "LTeamID": "team2_id",
                    }
                ),
                on=["season", "day_num", "team1_id", "team2_id"],
                how="left",
            )
            .join(
                cities,
                on=["city_id"],
                how="left",
            )
            .sort(["season", "day_num", "team1_id"])
            .with_row_index("ID")
            .with_columns(
                team2_loc=pl.col("team1_loc").replace({"H": "A", "A": "H"}),
                is_conf_tourney=pl.col("conf_tourney_abbr").is_not_null(),
                day_zero=pl.col("day_zero").str.to_date("%m/%d/%Y"),
                minutes=pl.lit(40).add(pl.col("num_ot").mul(pl.lit(5))),
                spread=pl.col("team1_score").sub(pl.col("team2_score")),
                total=pl.col("team1_score").add(pl.col("team2_score")),
                team1_win=pl.col("team1_score").gt(pl.col("team2_score")),
                team2_win=pl.col("team2_score").gt(pl.col("team1_score")),
                team1_poss=_poss_expr("team1"),
                team2_poss=_poss_expr("team2"),
                # Before 2003, there's only secondary results
                # `days_into_season` won't fully make sense for these early seasons
                days_into_season=pl.col("day_num").sub(pl.col("day_num").min().over("season")),
            )
            .with_columns(
                days_into_season_norm=pl.col("days_into_season").truediv(
                    pl.col("days_into_season").max().over("season")
                ),
                days_into_season_sq=pl.col("days_into_season").pow(2),
                recency_weight=pl.col("days_into_season")
                .add(pl.lit(RECENCY_PADDING))
                .truediv(pl.col("days_into_season").add(pl.lit(RECENCY_PADDING)).max().over("season")),
                date=pl.col("day_zero").add(pl.duration(days=pl.col("day_num"))),
                team1_home=pl.col("team1_loc").eq("H"),
                team2_home=pl.col("team2_loc").eq("H"),
                is_neutral=pl.col("team1_loc").eq("N"),
                avg_poss=pl.col("team1_poss").add(pl.col("team2_poss")).truediv(2),
                team1_pace=pl.col("team1_poss").truediv(pl.col("minutes")),
                team2_pace=pl.col("team2_poss").truediv(pl.col("minutes")),
            )
            .with_columns(
                days_into_season_norm_sq=pl.col("days_into_season_norm").pow(2),
                # Use the average possessions for PPP calculations - smoother, more stable
                team1_ppp=pl.col("team1_score").truediv(pl.col("avg_poss")),
                team2_ppp=pl.col("team2_score").truediv(pl.col("avg_poss")),
                avg_pace=pl.col("avg_poss").truediv(pl.col("minutes")),
            )
        )
        team_locations = derive_team_locations(frame)
        team_rest = derive_rest_days(frame)
        frame = (
            frame.join(
                team_locations.rename(
                    {
                        "team_id": "team1_id",
                        "city": "team1_city",
                        "state": "team1_state",
                        "lat": "team1_lat",
                        "lng": "team1_lng",
                        "elevation": "team1_elevation",
                    }
                ),
                on=["season", "team1_id"],
                how="left",
            )
            .join(
                team_locations.rename(
                    {
                        "team_id": "team2_id",
                        "city": "team2_city",
                        "state": "team2_state",
                        "lat": "team2_lat",
                        "lng": "team2_lng",
                        "elevation": "team2_elevation",
                    }
                ),
                on=["season", "team2_id"],
                how="left",
            )
            .join(
                team_rest.rename(
                    {
                        "team_id": "team1_id",
                        "b2b": "team1_b2b",
                        "2in3": "team1_2in3",
                        "3in4": "team1_3in4",
                        "4in5": "team1_4in5",
                    }
                ),
                on=["season", "team1_id", "ID"],
                how="left",
            )
            .join(
                team_rest.rename(
                    {
                        "team_id": "team2_id",
                        "b2b": "team2_b2b",
                        "2in3": "team2_2in3",
                        "3in4": "team2_3in4",
                        "4in5": "team2_4in5",
                    }
                ),
                on=["season", "team2_id", "ID"],
                how="left",
            )
            .with_columns(
                city=_fill_null_location(null_col="city"),
                state=_fill_null_location(null_col="state"),
                lat=_fill_null_location(null_col="lat"),
                lng=_fill_null_location(null_col="lng"),
                elevation=_fill_null_location(null_col="elevation"),
            )
            .with_columns(
                team1_travel_distance=pl.when(pl.col("team1_loc").eq("H"))
                .then(0)
                .otherwise(haversine(lat1="team1_lat", lon1="team1_lng", lat2="lat", lon2="lng")),
                team2_travel_distance=pl.when(pl.col("team2_loc").eq("H"))
                .then(0)
                .otherwise(haversine(lat1="team2_lat", lon1="team2_lng", lat2="lat", lon2="lng")),
            )
            .with_columns(
                # Will use convention of team1 - team2 for consistency with spread
                **{
                    "team1_travel_distance_diff": pl.col("team1_travel_distance").sub(pl.col("team2_travel_distance")),
                    "team1_elevation_diff": pl.col("team1_elevation").sub(pl.col("team2_elevation")),
                    "team2_travel_distance_diff": pl.col("team2_travel_distance").sub(pl.col("team1_travel_distance")),
                    "team2_elevation_diff": pl.col("team2_elevation").sub(pl.col("team1_elevation")),
                    "b2b_diff": pl.col("team1_b2b").sub(pl.col("team2_b2b")),
                    "2in3_diff": pl.col("team1_2in3").sub(pl.col("team2_2in3")),
                    "3in4_diff": pl.col("team1_3in4").sub(pl.col("team2_3in4")),
                    "4in5_diff": pl.col("team1_4in5").sub(pl.col("team2_4in5")),
                },
            )
            .with_columns(
                team1_log_travel_distance_diff=pl.when(pl.col("team1_travel_distance_diff").gt(0))
                .then(pl.col("team1_travel_distance_diff").log1p())
                .otherwise(pl.col("team1_travel_distance_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                team1_log_elevation_diff=pl.when(pl.col("team1_elevation_diff").gt(0))
                .then(pl.col("team1_elevation_diff").log1p())
                .otherwise(pl.col("team1_elevation_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                team2_log_travel_distance_diff=pl.when(pl.col("team2_travel_distance_diff").gt(0))
                .then(pl.col("team2_travel_distance_diff").log1p())
                .otherwise(pl.col("team2_travel_distance_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                team2_log_elevation_diff=pl.when(pl.col("team2_elevation_diff").gt(0))
                .then(pl.col("team2_elevation_diff").log1p())
                .otherwise(pl.col("team2_elevation_diff").mul(pl.lit(-1)).log1p().mul(pl.lit(-1))),
                fatigue_effect_1=pl.col("b2b_diff"),
                fatigue_score_diff=pl.sum_horizontal(
                    [
                        pl.col("b2b_diff"),
                        pl.col("2in3_diff"),
                        pl.col("3in4_diff"),
                        pl.col("4in5_diff"),
                    ]
                ),
            )
            .with_columns(
                fatigue_effect_2=pl.col("2in3_diff").add(pl.col("fatigue_effect_1")),
            )
            .with_columns(
                fatigue_effect_3=pl.col("3in4_diff").add(pl.col("fatigue_effect_2")),
            )
            .with_columns(
                fatigue_effect_4=pl.col("4in5_diff").add(pl.col("fatigue_effect_3")),
            )
        )
        if self.league == "M":
            # TODO: These are men's only data, but check if this changes in the future
            coaches = self.read(Files.team_coaches)
            massey_ordinals = (
                self.read(Files.massey_ordinals)
                .with_columns(SystemName=pl.format("ranking_{}", pl.col("SystemName").str.to_lowercase()))
                .pivot(
                    on=["SystemName"],
                    index=["Season", "RankingDayNum", "TeamID"],
                    values=["OrdinalRank"],
                )
                .rename(
                    {
                        "Season": "season",
                        "TeamID": "team_id",
                        "RankingDayNum": "day_num",
                    }
                )
            )
            ranking_cols = [x for x in massey_ordinals.columns if x.startswith("ranking_")]
            massey_ordinals = massey_ordinals.sort(["season", "team_id", "day_num"]).with_columns(
                *[pl.col(x).forward_fill().over(["season", "team_id"]) for x in ranking_cols]
            )

            frame = (
                frame.join_where(
                    coaches,
                    pl.col("season").eq(pl.col("Season")),
                    pl.col("team1_id").eq(pl.col("TeamID")),
                    pl.col("day_num").ge(pl.col("FirstDayNum")),
                    pl.col("day_num").le(pl.col("LastDayNum")),
                    how="left",
                )
                .drop(["Season", "TeamID", "FirstDayNum", "LastDayNum"])
                .rename({"CoachName": "team1_coach"})
                .join_where(
                    coaches,
                    pl.col("season").eq(pl.col("Season")),
                    pl.col("team2_id").eq(pl.col("TeamID")),
                    pl.col("day_num").ge(pl.col("FirstDayNum")),
                    pl.col("day_num").le(pl.col("LastDayNum")),
                    how="left",
                )
                .drop(["Season", "TeamID", "FirstDayNum", "LastDayNum"])
                .rename({"CoachName": "team2_coach"})
                .sort(["season", "team1_id", "day_num"])
                .join_asof(
                    massey_ordinals.rename({x: f"team1_{x}" for x in ranking_cols} | {"team_id": "team1_id"}),
                    by=["season", "team1_id"],
                    on="day_num",
                    strategy="backward",
                    check_sortedness=False,
                )
                .sort(["season", "team2_id", "day_num"])
                .join_asof(
                    massey_ordinals.rename({x: f"team2_{x}" for x in ranking_cols} | {"team_id": "team2_id"}),
                    by=["season", "team2_id"],
                    on="day_num",
                    strategy="backward",
                    check_sortedness=False,
                )
                .sort(["season", "day_num", "team1_id"])
            )

        return frame

    def load_game_team_box_scores(self) -> pl.DataFrame:
        """Return box score and other game-level information with one row per team-game (two rows per game)."""
        logger.info("Creating game team box scores...")
        return build_team_game_box_scores(self.load_game_box_scores())

    def load_test_data(self, season: int | None = None) -> pl.DataFrame:
        season = season or current_season()
        teams = self.read(Files.teams)
        test_df = (
            self.read(Files.sample_submissions_stage_2)
            .with_columns(
                id_split=pl.col("ID").str.split("_"),
            )
            .with_columns(
                season=pl.col("id_split").list.get(0).cast(int),
                team1_id=pl.col("id_split").list.get(1).cast(int),
                team2_id=pl.col("id_split").list.get(2).cast(int),
                minutes=pl.lit(40),
                loc=pl.lit("N"),
                is_neutral=pl.lit(1),
                team1_home=pl.lit(0),
                team2_home=pl.lit(0),
                is_conf_tourney=pl.lit(0),
                is_ncaa_tourney=pl.lit(1),
                is_secondary_tourney=pl.lit(0),
                team1_loc=pl.lit("N"),
                team2_loc=pl.lit("N"),
                team1_b2b=pl.lit(0),
                team1_2in3=pl.lit(0),
                team1_3in4=pl.lit(0),
                team1_4in5=pl.lit(0),
                team2_b2b=pl.lit(0),
                team2_2in3=pl.lit(0),
                team2_3in4=pl.lit(0),
                team2_4in5=pl.lit(0),
                days_into_season_norm=pl.lit(1),
                days_into_season_norm_sq=pl.lit(1),
                team1_log_travel_distance_diff=pl.lit(0),
                team1_log_elevation_diff=pl.lit(0),
                team2_log_travel_distance_diff=pl.lit(0),
                team2_log_elevation_diff=pl.lit(0),
            )
            .filter(pl.col("team1_id").cast(str).str.starts_with("1" if self.league == "M" else "3"))
            .join(
                teams.rename({"TeamID": "team1_id", "TeamName": "team1_name"}),
                how="left",
                on=["team1_id"],
            )
            .join(
                teams.rename({"TeamID": "team2_id", "TeamName": "team2_name"}),
                how="left",
                on=["team2_id"],
            )
            .drop("id_split")
        )

        if self.league == "W":
            # TODO: fill in null team IDs (result of last four play-in games)
            # In the women's tourney, higher seeds have home court advantage in the first two rounds
            tourney_seeds = self.read(Files.ncaa_tourney_seeds).filter(pl.col("Season").eq(season)).drop("Season")
            homecourt_df = (
                generate_ncaaw_homecourt()
                .join(
                    tourney_seeds.rename({"Seed": "StrongSeed", "TeamID": "team1_id"}),
                    how="left",
                    on="StrongSeed",
                )
                .join(
                    tourney_seeds.rename({"Seed": "WeakSeed", "TeamID": "team2_id"}),
                    how="left",
                    on="WeakSeed",
                )
            )
            hca_df = (
                pl.concat(
                    [
                        homecourt_df,
                        homecourt_df.with_columns(
                            team2_id=pl.col("team1_id"),
                            team1_id=pl.col("team2_id"),
                            team2_home=pl.col("team1_home"),
                        ).drop("team1_home"),
                    ],
                    how="diagonal_relaxed",
                )
                .with_columns(
                    pl.col("team1_home").fill_null(0),
                    pl.col("team2_home").fill_null(0),
                )
                .with_columns(
                    is_neutral=pl.lit(0),
                    team1_loc=pl.when(pl.col("team1_home").eq(1)).then(pl.lit("H")).otherwise(pl.lit("N")),
                    team2_loc=pl.when(pl.col("team2_home").eq(1)).then(pl.lit("H")).otherwise(pl.lit("N")),
                )
                .with_columns(
                    team1_loc=pl.when(pl.col("team2_loc").eq("H")).then(pl.lit("A")).otherwise(pl.col("team1_loc")),
                    team2_loc=pl.when(pl.col("team1_loc").eq("H")).then(pl.lit("A")).otherwise(pl.col("team2_loc")),
                )
            )
            test_df = test_df.update(
                hca_df.drop(["Slot", "StrongSeed", "WeakSeed"]),
                on=["team1_id", "team2_id"],
                how="left",
            )
        return test_df

    def create_bracket(self, season: int | None = None, *, save: bool = True) -> None:
        """Generate a bracket for the NCAA tournament."""
        season = season or current_season()
        teams = self.read(Files.teams)
        tourney_seeds = (
            self.read(Files.ncaa_tourney_seeds)
            .filter(pl.col("Season").eq(current_season()))
            .join(
                teams.select("TeamID", "TeamName"),
                how="left",
                on="TeamID",
            )
            .with_columns(pl.col("Seed").str.slice(0, 3))
            .with_columns(
                Region=pl.col("Seed").str.slice(0, 1),  # Extract region from seed (e.g., "W01" -> "W")
                Seed=pl.col("Seed").str.slice(1, 3).cast(int),  # Extract seed number (e.g., "W01" -> 1)
            )
        )
        predictions = pl.read_csv(OUTPUT_DIR / "final_submission.csv")
        if self.league == "M":
            exclude = [
                1420,  # UMBC
                # 1224,  # Howard
                1400,  # Texas
                # 1301,  # NC State
                1341,  # Prairie View A&M
                # 1250,  # Lehigh
                1275,  # Miami (OH)
                # 1374,  # SMU
            ]
        else:
            exclude = [
                3283,  # Missouri State
                # 3372,  # Stephen F. Austin
                3304,  # Nebraska
                # 3350,  # Richmond
                3380,  # Southern U.
                # 3359,  # Samford
                3438,  # Virginia
                # 3113,  # Arizona State
            ]
        tourney_seeds = tourney_seeds.filter(~pl.col("TeamID").is_in(exclude))
        team_list = tourney_seeds["TeamName"].to_list()
        team_id_list = tourney_seeds["TeamID"].to_list()
        team_map = dict(zip(team_list, team_id_list, strict=False))

        def _get_win_prob(team: str, opp_team: str) -> float:
            team_id = team_map[team]
            opp_team_id = team_map[opp_team]
            if team_id < opp_team_id:
                game_id = f"{season}_{team_id}_{opp_team_id}"
                # lower id means we take prediction directly
                win_prob = predictions.filter(pl.col("ID").eq(game_id))["Pred"].item()
            else:
                game_id = f"{season}_{opp_team_id}_{team_id}"
                # higher id means we take 1 - prediction
                win_prob = 1 - predictions.filter(pl.col("ID").eq(game_id))["Pred"].item()
            return win_prob

        matchup_dict = defaultdict(list)
        for team in team_list:
            for opp_team in team_list:
                if team == opp_team:
                    prob = None
                else:
                    prob = _get_win_prob(team, opp_team)
                matchup_dict[team].append(prob)

        matchup_df = pd.DataFrame(matchup_dict, index=team_list).T

        round_df = pd.DataFrame(
            data=np.tile(np.nan, (len(team_list), len(team_list))),
            index=team_list,
            columns=team_list,
        )
        base_df = pd.DataFrame(tourney_seeds, columns=tourney_seeds.columns)
        for i, row in base_df.iterrows():
            region_round_map = WITHIN_REGION_STANDARD_SEED_SETTINGS[row["Seed"]]
            region_round_map[row["Seed"]] = np.nan
            base_rounds = base_df["Seed"].replace(region_round_map)

            # Set self to null, necessary since sometimes will have the same seed as a play-in
            base_rounds.loc[i] = np.nan

            # Set Final Four matchups
            final_four_rounds = base_df["Region"] == FINAL_FOUR_REGION_SETTINGS[row["Region"]]
            base_rounds.loc[final_four_rounds] = base_rounds.max() + 1
            # In standard 64 bracket (no play ins) this should sum to 16 (all teams from paired region)

            # Set Championship matchups
            championship_rounds = (base_df["Region"] != row["Region"]) & (~final_four_rounds)
            base_rounds.loc[championship_rounds] = base_rounds.max() + 1  # only +1 since now this has some Final Four
            # In standard 64 bracket (no play ins) this should sum to 32 (all teams from other side of bracket)

            round_df[row["TeamName"]] = pd.Series(base_rounds.values, index=team_list)

        round_df = round_df.T

        advance_df = pd.DataFrame(index=round_df.index)
        # TODO: will need to change R0 and range if first four
        # For now, this essentially skips the round=0 created by some of the round generators
        advance_df["R0"] = 1.0
        for round_num in range(1, 6 + 1):
            advance_series = pd.Series(dtype=float)
            for team_name, _ in round_df.iterrows():
                team_round_row = round_df.loc[team_name, :]
                round_opponents = team_round_row.loc[team_round_row == round_num].index.tolist()
                advance_pr = (
                    advance_df.loc[round_opponents, f"R{round_num - 1}"] * matchup_df.loc[team_name, round_opponents]
                ).sum()
                advance_series[team_name] = advance_df.loc[team_name, f"R{round_num - 1}"] * advance_pr
            advance_df[f"R{round_num}"] = advance_series

        advance_df = advance_df.sort_values("R6", ascending=False)
        advance_df.to_csv(OUTPUT_DIR / f"{self.league}/advance_{self.league}_{season}.csv")

    def analysis(self) -> None:
        tourney_slots = self.read(Files.ncaa_tourney_slots)
        tourney_seeds = self.read(Files.ncaa_tourney_seeds)
        results = self.read(Files.ncaa_tourney_compact_results).drop("DayNum", "WLoc", "NumOT")
        teams = self.read(Files.teams).drop("FirstD1Season", "LastD1Season")

        df = (
            tourney_slots.join(
                tourney_seeds.rename({"Seed": "StrongSeed", "TeamID": "StrongTeamID"}),
                on=["Season", "StrongSeed"],
                how="left",
            )
            .join(
                tourney_seeds.rename({"Seed": "WeakSeed", "TeamID": "WeakTeamID"}),
                on=["Season", "WeakSeed"],
                how="left",
            )
            .join(
                teams.rename({"TeamID": "StrongTeamID", "TeamName": "StrongTeamName"}),
                on=["StrongTeamID"],
                how="left",
            )
            .join(
                teams.rename({"TeamID": "WeakTeamID", "TeamName": "WeakTeamName"}),
                on=["WeakTeamID"],
                how="left",
            )
            .join(
                results.rename(
                    {
                        "WTeamID": "StrongTeamID",
                        "LTeamID": "WeakTeamID",
                        "WScore": "StrongScore",
                        "LScore": "WeakScore",
                    }
                ),
                on=["Season", "StrongTeamID", "WeakTeamID"],
                how="left",
            )
            .update(
                results.rename(
                    {
                        "WTeamID": "WeakTeamID",
                        "LTeamID": "StrongTeamID",
                        "WScore": "WeakScore",
                        "LScore": "StrongScore",
                    }
                ),
                on=["Season", "StrongTeamID", "WeakTeamID"],
                how="left",
            )
            .filter(
                pl.col("StrongTeamID").is_not_null(),
                pl.col("WeakTeamID").is_not_null(),
            )
            .with_columns(
                spread=pl.col("StrongScore").sub(pl.col("WeakScore")),
                strong_win=pl.col("StrongScore").gt(pl.col("WeakScore")),
                weak_win=pl.col("WeakScore").gt(pl.col("StrongScore")),
                matchup=pl.format("{} vs {}", pl.col("StrongSeed").str.slice(1, 2), pl.col("WeakSeed").str.slice(1, 2)),
            )
        )
        group_df = (
            df.group_by("matchup")
            .agg(
                total_games=pl.count(),
                strong_wins=pl.col("strong_win").sum(),
                weak_wins=pl.col("weak_win").sum(),
                avg_spread=pl.col("spread").mean(),
            )
            .with_columns(
                strong_win_pct=pl.col("strong_wins").truediv(pl.col("total_games")),
                weak_win_pct=pl.col("weak_wins").truediv(pl.col("total_games")),
            )
            .sort("matchup")
        )
