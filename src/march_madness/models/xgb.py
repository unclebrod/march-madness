"""Skeleton file for later implementation of XGBoost models


OUTLINE OF IDEAS:
- Get out of sample predictions for the "main" ppp model
    - including both raw prediction values, plus hierarchical team effects
- Incorporate elo as a feature
- Incorporate tournament contextual features that the model can't, like competitor seeding
- Additional features such as aggregated teams statistics
     - Consider raw average, last N games, weighted tuned exponentially weighted statistics
     - Rate stats (four factors for example)
- Probably need this frame to be game/team level, with each row representing a team's performance in a specific game
    - Need the model to understand games from both ways
    - Probably need to consider averaging predictions from both team perspectives
- Could also include raw seeding as a feature

CHATGPT IDEAS:
    - Offense/defense interactions?
    - Could even incorporate the uncertainty from the posterior predictions (i like dis)
    - ALSO predicted totals, possessions
    - Standard deviation of predicted scoring margin
    - Absolute seed difference as well as directional
    - Conference?
    - Turnover rate
    - Offensive rebounding rate
    - Free throw rate
    - 3PA/3P%
    - Recent shooting/turnover/efficiency trends (last N games or exponentially weighted)
    - Try to prioritize per possession or rate-based metrics
    - Season-long statistics vs. recent statistics using a fixed window vs exponentially weighted statistics
    - Recent vs season long difference


TUNING:
    - Tournament games since features are derived from the regular season
    - Consider cross-validation strategies that respect the temporal nature of the data
    - OR could k-fold where each left out fold represents a different season
        - Think this is higher values since features won't be temporal across seasons
    - Have to be intentional about number of features given relatively small sample size of tournament games
    - Shallow trees with strong regularization, early stopping based on validation Brier score
    - Conservative learning rates
    - Compare against logistic regression baseline
"""
