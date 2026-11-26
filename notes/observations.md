## Data quality observations
- Rating scale: 0.5-5.0 in half steps, mean = 3.50
- Duplicate user-movie pairs: 0
- Movies with no genre listed: 34 (encoded as text, not NaN)
- Sparsity: 98.3% (610 users x 9,724 movies)
- Movies with only 1 rating: 3,446 (about 35% of movies)
- Heavy long tail: one user has 2,698 ratings, the median user has 70
- Time range of data: 1996-03-29 to 2018-09-24

## Experiment observations
- v1 (leave-one-out encoding): gradient boosting overfit badly
  (regression RMSE 0.7163 train vs 0.8960 test; classification AUC 0.9288 vs 0.7548).
  Cause: leave-one-out target encoding leaks the row's own label to tree models.
- Linear/logistic regression were unaffected (train ~ test).
- Fix (v2): 5-fold out-of-fold target encoding.
- Class balance: 48.2% "liked" (rating >= 4), so near-balanced; baseline accuracy 52%.