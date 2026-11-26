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

- v2 (out-of-fold encoding) results: GB regression test RMSE 0.8620 (baseline 1.0488, linear 0.8799);
  GB classification test acc 0.7157 / F1 0.7085 / AUC 0.7914 (logistic: 0.7067 / 0.6940 / 0.7787).
- Fixing the leakage improved GB test AUC from 0.7548 to 0.7914 and test RMSE from 0.8960 to 0.8620.
- Best model for both tasks: HistGradientBoosting.

## Recommender observations
- SVD test RMSE: 0.9210 (10 factors) -> 0.9405 (100 factors); more factors overfit.
- Precision@10: popularity 0.1225, best SVD (20 factors) 0.1339, SVD-100 0.0903.
- SVD beats the user-mean baseline (0.9497) but not gradient boosting regression (0.8620).
- Similar-movie lists are noisy for movies with few ratings (e.g. Toy Story neighbours are mixed genres).
- Saved model: 10 factors (lowest RMSE).

## Clustering / PCA observations
- User profile = share of ratings per genre (19) + avg rating (20 features), standardised.
- PCA: 12 of 20 components needed for 90% variance; PC1 22.4%, PC2 19.3% (PC1: Crime/Drama/Thriller vs Fantasy/Animation/Children).
- K-Means silhouette is low everywhere (best 0.1151 at k=3, 0.0878 at k=10): weak cluster structure, tastes are a continuum.
- k searched from 3 to 10 on purpose (k=2 only splits users crudely).
- Clusters: 0 = crime/thriller (227 users, avg 3.76); 1 = action/adventure/sci-fi (137, 3.75); 2 = comedy/romance/drama (246, 3.51).