"""Main entry point for March Madness modeling and analysis."""

import numpyro
import plotly.io as pio
import polars as pl
from cyclopts import App
from dotenv import find_dotenv, load_dotenv

from march_madness import geocoding, odds
from march_madness.elo import run_elo
from march_madness.loader import DataLoader
from march_madness.log import configure_logging
from march_madness.models.base import McmcParams, SviParams
from march_madness.models.ppp import PointsPerPossessionTrainer
from march_madness.settings import OUTPUT_DIR
from march_madness.trainer import Trainer
from march_madness.tuner import Tuner

TRAINER_MAP: dict[str, type[Trainer]] = {
    "ppp": PointsPerPossessionTrainer,
}


app = App()


@app.command
def train(
    league: str = "M",
    model: str = "ppp",
    inference: str = "svi",
    num_samples: int = 2_000,
    mcmc_params: McmcParams | None = None,
    svi_params: SviParams | None = None,
    *,
    save: bool = True,
) -> None:
    logger.info(f"Training {model} model for {league} league with {inference} inference.")
    data_constructor = DataLoader(league=league)
    trainer = TRAINER_MAP[model](league=league)
    df = data_constructor.load_game_team_box_scores()
    trainer.train(
        df=df,
        inference=inference,
        num_samples=num_samples,
        mcmc_params=mcmc_params,
        svi_params=svi_params,
    )
    # preds = trainer.predict(df=df)  # Generate predictions on training data for evaluation
    if save:
        trainer.save(path=league)
    logger.info("Training complete.")


@app.command
def infer(
    league: str = "M",
    model: str = "ppp",
    *,
    save: bool = True,
) -> None:
    logger.info(f"Running inference for {model} model in {league} league.")
    trainer = TRAINER_MAP[model].load(path=league)
    trainer.infer(save=save)
    logger.info("Inference complete.")


@app.command
def tune(
    league: str = "M",
    model: str = "ppp",
    inference: str = "svi",
    num_samples: int = 1_000,
    mcmc_params: McmcParams | None = None,
    svi_params: SviParams | None = None,
    n_trials: int = 50,
) -> None:
    logger.info(f"Tuning {model} model for {league} league.")
    data_constructor = DataLoader(league=league)
    df = data_constructor.load_game_team_box_scores()
    trainer_cls = TRAINER_MAP[model]
    tuner = Tuner(
        df=df,
        trainer_cls=trainer_cls,
        league=league,
    )
    tuner.tune(
        inference=inference,
        num_samples=num_samples,
        n_trials=n_trials,
        mcmc_params=mcmc_params,
        svi_params=svi_params,
    )
    logger.info("Tuning complete.")


@app.command
def submit(
    season: int,
    model: str = "ppp",
    suffix: str = "",
) -> None:
    logger.info(f"Generating submission for {model} model for season {season}.")
    sfx = f"_{suffix}" if suffix else ""
    df_list: list[pl.DataFrame] = []
    for league in ["M", "W"]:
        data_constructor = DataLoader(league=league)
        df = data_constructor.load_test_data(season=season)
        trainer = TRAINER_MAP[model].load(path=league)
        preds = trainer.predict(df=df)
        df_list.append(preds)
    submission = pl.concat(df_list, how="vertical_relaxed")
    submission.select(
        pl.col("ID"),
        pl.col("team1_win_prob").alias("Pred"),
    ).write_csv(OUTPUT_DIR / f"final_submission{sfx}.csv")
    submission.select(
        pl.col("ID"),
        (pl.col("spread") * -1).alias("Pred"),
    ).write_csv(OUTPUT_DIR / f"final_submission_spread{sfx}.csv")
    logger.info("Submission files generated.")


@app.command
def bracket(
    season: int | None = None,
    league: str = "M",
    *,
    save: bool = True,
) -> None:
    logger.info(f"Creating bracket for {league} league for season {season}.")
    data_constructor = DataLoader(league=league)
    data_constructor.create_bracket(season=season, save=save)
    logger.info("Bracket creation complete.")


@app.command
def analysis(
    league: str = "M",
) -> None:
    logger.info(f"Running analysis for {league} league.")
    data_constructor = DataLoader(league=league)
    data_constructor.analysis()
    logger.info("Analysis complete.")


@app.command
def geocode() -> None:
    logger.info("Starting geocoding process.")
    geocoding.geocode_cities()
    logger.info("Geocoding process complete.")


@app.command
def espn() -> None:
    logger.info("Fetching odds from ESPN.")
    odds.main()
    logger.info("Odds fetching complete.")


@app.command
def elo(
    league: str = "M",
    *,
    tune: bool = True,
    n_trials: int = 100,
    starting_rating: int = 0,
    scale_factor: float = 400,
    denominator_baseline: float = 7.5,
    k_factor: float = 32,
    season_carryover: float = 0.880909198124617,
    hca: float = 0.4791259918853443,
    decay: float = 0.8770430595292348,
    shift: float = 5.011022004050616,
    advantage_scale: float = 0.00032279515387965106,
    max_upset_boost: float = 3.745746081635934,
) -> None:
    logger.info(f"Running Elo rating calculation for {league} league.")
    run_elo(
        league=league,
        tune=tune,
        n_trials=n_trials,
        starting_rating=starting_rating,
        scale_factor=scale_factor,
        season_carryover=season_carryover,
        hca=hca,
        decay=decay,
        shift=shift,
        denominator_baseline=denominator_baseline,
        k_factor=k_factor,
        advantage_scale=advantage_scale,
        max_upset_boost=max_upset_boost,
    )
    logger.info("Elo rating calculation complete.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    configure_logging()

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(4)

    pio.renderers.default = "browser"

    app()
