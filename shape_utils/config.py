from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FunctionalMapConfig(BaseModel):
    w_descr: float = Field(default=1e0, ge=0, description="Descriptor preservation weight.")
    w_lap: float = Field(default=1e-2, ge=0, description="Laplacian commutativity weight.")
    w_dcomm: float = Field(default=1e-1, ge=0, description="Descriptor commutativity weight.")
    w_orient: float = Field(default=0.0, ge=0, description="Orientation preservation weight.")

    n_cpus: int = Field(default=1, ge=1, description="Number of CPU workers to use.")
    refine: Optional[Literal["icp", "zoomout"]] = Field(default=None, description="Optional refinement method.")
    verbose: bool = Field(default=True, description="Pass verbose output to pyFM routines.")

    zoomout_nit: int = Field(default=11, ge=1, description="Number of ZoomOut refinement iterations.")
    zoomout_step: int = Field(default=1, ge=1, description="ZoomOut refinement step size.")

    def fit_params(self) -> dict[str, float]:
        return {
            "w_descr": self.w_descr,
            "w_lap": self.w_lap,
            "w_dcomm": self.w_dcomm,
            "w_orient": self.w_orient,
        }

    @classmethod
    def from_value(
        cls,
        value: "FunctionalMapConfig | dict[str, Any] | int | None" = None,
        *,
        n_cpus: int | None = None,
        refine: str | None = None,
    ) -> "FunctionalMapConfig":
        if isinstance(value, cls):
            data = value.dict()
        elif isinstance(value, dict):
            data = dict(value)
        elif isinstance(value, int):
            data = {"n_cpus": value}
        elif value is None:
            data = {}
        else:
            raise TypeError("config must be FunctionalMapConfig, dict, int, or None")

        if n_cpus is not None:
            data["n_cpus"] = n_cpus
        if refine is not None:
            data["refine"] = refine

        return cls(**data)


class DenseMeshConfig(BaseModel):
    dist_ratio: float = Field(default=3.0, gt=0, description="Distance radius multiplier.")
    self_weight_limit: float = Field(default=0.25, ge=0, description="Minimum self weight.")
    correct_dist: bool = Field(default=False, description="Correct distance matrix values.")
    interpolation: str = Field(default="poly", description="Dense mesh interpolation method.")
    return_dist: bool = Field(default=True, description="Return distance matrix from dense mesh processing.")
    adapt_radius: bool = Field(default=True, description="Adapt geodesic radius during dense mesh processing.")
    update_sample: bool = Field(default=True, description="Update sampling during dense mesh processing.")
    force_n_samples: bool = Field(default=False, description="Force exactly the requested number of samples.")
    verbose: bool = Field(default=True, description="Pass verbose output to dense mesh routines.")

    def process_params(self, n_cpus: int) -> dict[str, Any]:
        data = self.dict()
        data["n_jobs"] = n_cpus
        return data
