from pydantic import BaseModel, Field


class RatingRequest(BaseModel):
    user_id: int = Field(..., ge=1, examples=[1])
    movie_id: int = Field(..., ge=1, examples=[318])


class RatingResponse(BaseModel):
    user_id: int
    movie_id: int
    title: str | None
    predicted_rating: float
    liked_probability: float
    liked: bool
    model_version: str


class FeaturesRequest(BaseModel):
    user_mean: float = Field(..., ge=0.5, le=5.0)
    user_count: int = Field(..., ge=0)
    movie_mean: float = Field(..., ge=0.5, le=5.0)
    movie_count: int = Field(..., ge=0)
    genres: list[str] = Field(default_factory=list)


class FeaturesResponse(BaseModel):
    predicted_rating: float
    liked_probability: float
    liked: bool
    unknown_genres: list[str]
    model_version: str


class Recommendation(BaseModel):
    movie_id: int
    title: str


class RecommendResponse(BaseModel):
    user_id: int
    known_user: bool
    recommendations: list[Recommendation]
    model_version: str


class BatchRow(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float
    liked_probability: float


class BatchResponse(BaseModel):
    n_rows: int
    model_version: str
    predictions: list[BatchRow]