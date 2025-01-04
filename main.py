import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open('iris_model.pkl','rb') as f:
    model = pickle.load(f)

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petl_length:float
    petal_width:float

@app.get("/")
def read_root():
    return(f"predicted output is ")

@app.post("/predict/")
def predict(data: IrisInput):
    # Prepare input data for prediction
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Map prediction to target names
    #target_names = ["setosa", "versicolor", "virginica"]
    #predicted_class = target_names[prediction]

    return {'prediction':int(prediction)}
