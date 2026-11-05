"""Validated scalar parameters for optimizers and experiments."""

from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Config(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, validate_default=True
    )

    def with_updates(self, **updates: object) -> Self:
        """Validate a replacement before changing the caller's configuration."""
        return cast(Self, type(self).model_validate({**self.model_dump(), **updates}))


class ExhaustiveSearchConfig(_Config):
    max_items: int = Field(default=20, ge=0)


class HillClimbingConfig(_Config):
    neighborhood_size: int = Field(default=3, ge=0)


class SimulatedAnnealingConfig(_Config):
    initial_temperature: float = Field(default=100.0, gt=0, allow_inf_nan=False)
    minimum_temperature: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    cooling_rate: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)
    iterations_per_temperature: int = Field(default=100, gt=0)
    max_iterations: int = Field(default=20_000, gt=0)
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_temperatures(self) -> Self:
        if self.initial_temperature <= self.minimum_temperature:
            raise ValueError("initial_temperature must exceed minimum_temperature")
        return self


class ExperimentConfig(_Config):
    n_repeats: int = Field(default=30, gt=0)
