from pandas import read_csv
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from miceforest import ImputationKernel

from sklearn.preprocessing import StandardScaler
import pickle
import joblib


def load_data_frame():
    # load dataset
    dataframe = read_csv("housing_data.txt", delim_whitespace=True, header=None)
    return dataframe


def cleanData(dataframe):
    # remove observations or features with missing values
    dataframe.info()
    dataframe = dataframe.dropna()
    # dataframe.dropna(how="all") # rows where all the column values are missing
    # dataframe.dropna(thresh=0.6) # This strategy sets a minimum number of missing values required to preserve the rows

    # Fill with mean value of the column
    mean_value = dataframe.mean()
    dataframe = dataframe.fillna(mean_value)

    # Multiple imputation prediction
    mice_kernel = ImputationKernel(
        data=dataframe,
        save_all_iterations=True,
        random_state=2023
    )
    mice_kernel.mice(2)
    dataframe = mice_kernel.complete_data()
    dataframe.head()

    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    Y = dataset[:, 13]

    # scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, "data_transformer.joblib")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


# define wider model
def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(13, activation="relu"))
    model.add(layers.Dense(3, activation="relu"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def trainAndSave(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=20, batch_size=5)
    model.save("house_prediction_model.h5")
    return model


def evaluate(model, X_test, Y_test):
    # with open('scaler.pkl','rb') as f:
    #     sc = pickle.load(f)
    # X_test = sc.fit_transform(X_test)
    model.load_weights("house_prediction_model.h5")
    e = model.evaluate(X_test, Y_test)
    return e


if __name__ == "__main__":
    dataframe = load_data_frame()
    X_train, X_test, Y_train, Y_test = cleanData(dataframe)

    model = create_model()
    model = trainAndSave(model, X_train, Y_train)
    score = evaluate(model, X_test, Y_test)

    print("score", score)
