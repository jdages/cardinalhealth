import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from keras.models import load_model

class Predictor:

    def __init__(self):
        # import data
        self.training_data_url = "https://storage.googleapis.com/cardinalmldata/data.csv"
        self.testing_data_url = "https://storage.googleapis.com/cardinalmldata/datatest.csv"

    def _get_training_data(self):
        # training data
        training_data = pd.read_csv(self.training_data_url)
        training_dataset = training_data.values

        # split into input (X) and output (Y) variables
        X_train = training_dataset[:,1:10].astype(float)
        Y_train = training_dataset[:,10]

        return X_train, Y_train

    def _get_encoder(self, _data):
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(_data)
        return encoder

    # baseline model
    def _create_baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=9, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        X_train, Y_train = self._get_training_data()
        Y_encoder = self._get_encoder(Y_train)
        encoded_Y_train = Y_encoder.transform(Y_train)

        # evaluate model with standardized dataset
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('estimator', KerasClassifier(build_fn=self._create_baseline_model, epochs=5, batch_size=5, verbose=1)))
        pipeline = Pipeline(estimators)
        pipeline.fit(X_train, encoded_Y_train)
        self._save_model(pipeline)        

    def _save_model(self, _pipeline):
        # Save the Keras model first:
        _pipeline.named_steps['estimator'].model.save('keras_model.h5')
        # This hack allows us to save the sklearn pipeline:
        _pipeline.named_steps['estimator'].model = None
        # Finally, save the pipeline:
        joblib.dump(_pipeline, 'sklearn_pipeline.pkl')
        # delete the pipeline object
        del _pipeline

    def _load_model(self):
        # Load the pipeline first:
        _pipeline = joblib.load('sklearn_pipeline.pkl')
        # Then, load the Keras model:
        _pipeline.named_steps['estimator'].model = load_model('keras_model.h5')
        return _pipeline

    def predict(self, features):
        x_data = features.values
        model_input = x_data[:,1:10].astype(float)
        _pipeline = self._load_model()
        y_pred = _pipeline.predict(model_input)
        return y_pred

