from http.client import HTTPException
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from pydantic import BaseModel
import io
# Initialize instance
app = FastAPI()


def pre_processing(file: bytes):
    try:
        df = pd.read_csv(io.BytesIO(file))
        numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
        print("The number of numerical features is",len(numerical_features),"and they are : \n",numerical_features)
        categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
        print("The number of categorical features is",len(categorical_features),"and they are : \n",categorical_features)

        #discrete numerical features 
        discrete_feature = [feature for feature in numerical_features if df[feature].nunique()<=15 and feature != 'label']
        print("The number of discrete features is",len(discrete_feature),"and they are : \n",discrete_feature)
        continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + ['label']]
        print("The number of continuous_feature features is",len(continuous_feature),"and they are : \n",continuous_feature)

        df = pd.get_dummies(df, columns=categorical_features,drop_first=True)
        print("This Dataframe has {} rows and {} columns after encoding".format(df.shape[0], df.shape[1]))
        x = df.drop(['label'], axis=1)
        y = df['label']
        ms = MinMaxScaler()

    # Bước 2: Fit scaler trên dữ liệu
        ms.fit(x)

    # Bước 3: Transform dữ liệu
        x = ms.transform(x)
        return x, y
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")


# pre_processing("/home/tung/Downloads/MLE K1/Project/data/dataset_sdn.csv")

def get_label(classes):
    y_pred = []
    for i in classes:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

# Load model
model = joblib.load("./models/model.pkl")


# Create an endpoint to check api work or not
@app.get("/")
def check_health():
    return {"status": "Oke"}


# Initialize cache
cache = {}


# Create an endpoint to make prediction
@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()

    x,y = pre_processing(content)

    classes = model.predict(x)
    return get_label(classes=classes)

# @app.post("/predict_cache")
# def predict_cache(data: Diabetes_measures):
#     if str(data) in cache:
#         logger.info("Getting result from cache!")
#         return cache[str(data)]
#     else:
#         logger.info("Making predictions...")
#         logger.info(data)
#         logger.info(jsonable_encoder(data))
#         logger.info(pd.DataFrame(jsonable_encoder(data), index=[0]))
#         result = model.predict(pd.DataFrame(jsonable_encoder(data), index=[0]))[0]
#         cache[str(data)] = ["Normal", "Diabetes"][result]

#         return {"result": ["Normal", "Diabetes"][result]}