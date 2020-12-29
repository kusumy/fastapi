from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base

class ModelConfiguration(Base):
    __tablename__ = "model_configuration"

    id = Column(Integer, primary_key=True, index=True)
    target_name = Column(String)
    table_name = Column(String)
    target_column = Column(String)
    features_column = Column(String)
    lag_features = Column(Integer)
    forecasting_horizon = Column(Integer)
    model_file = Column(String)
    forecasting_table = Column(String)
    metric_table = Column(String)