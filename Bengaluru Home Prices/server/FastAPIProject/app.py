from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import util

app = FastAPI()

# Allow CORS (Cross-Origin Resource Sharing) - lets other websites access your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/get_location_names")
async def get_location_names():
    return {"locations": util.get_location_names()}

@app.post("/predict_home_price")
async def predict_home_price(request: Request):
    form_data = await request.form()
    total_sqft = float(form_data["total_sqft"])
    location = form_data["location"]
    bhk = int(form_data["bhk"])
    bath = int(form_data["bath"])

    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
    return {"estimated_price": estimated_price}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Server For Home Price Prediction...")
    util.load_saved_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=8000)