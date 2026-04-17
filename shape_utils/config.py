# config.py
from typing import Optional
from pydantic import BaseModel, Field

class FunctionalMapConfig(BaseModel):
    # Algorithmic hyperparameters
    w_descr: float = Field(1e0, ge=0)
    w_lap: float = Field(1e-2, ge=0)
    w_dcomm: float = Field(1e-1, ge=0)
    w_orient: float = Field(0.0, ge=0)

    # Runtime options (can come from CLI flags)
    n_cpus: int = Field(1, ge=1)
    refine: Optional[str] = None
    verbose: bool = True

    # Refinement options
    zoomout_nit: int = Field(11, ge=1)
    zoomout_step: int = Field(1, ge=1)