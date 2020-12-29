import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse
from sqlalchemy.orm import Session

from mlapi import crud, schemas, models
from mlapi.database import SessionLocal, engine

import typing

models.Base.metadata.create_all(bind=engine)
app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#@app.post('/predict_api')
#async def predict(item: schemas.predictAPIParameter, db: Session=Depends(get_db)):
    #model_conf = crud.get_model_configuration(db, target=item.target, startdate=item.startdate, enddate=item.enddate, horizon=item.horizon)
#    return

@app.post('/get_model_config', response_model=schemas.modelConfiguration)
async def model_config(item: schemas.predictAPIParameter, db: Session=Depends(get_db)):
    model_conf = crud.get_model_configuration(db, target=item.target)
    return model_conf

@app.get('/get_allmodel_config')
async def allmodel_config(db: Session=Depends(get_db)):
    model_conf = crud.get_allmodel_configuration(db)
    return model_conf

@app.post('/forecast')
async def forecast(item: schemas.predictAPIParameter, db: Session=Depends(get_db)):
    model_forecast = crud.get_forecast(db, target=item.target, startdate=item.startdate, enddate=item.enddate, horizon=item.horizon)
    return model_forecast

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)