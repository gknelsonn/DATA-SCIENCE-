import json
import pickle

import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(bedrooms, bathrooms, guestroom, airconditioning):
    loc_index = np.where(__locations == guestroom)[0]
    x = np.zeros(len(__locations))






def get_location_name():
    return __locations


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[7:]

    with open("./artifacts/random_forest_model.pkl", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_name())