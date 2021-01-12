from pydantic import BaseModel
from typing import List, Optional

class predictAPIParameter(BaseModel):
    target: str
    startdate: Optional[str] = None
    enddate: Optional[str] = None
    horizon: Optional[int] = None

class modelConfiguration(BaseModel):
    target_name: str
    table_name: str
    target_column: str
    features_column: str
    lag_features: int
    forecasting_horizon: int
    model_file: str
    anomaly_model: str

    class Config:
        orm_mode = True