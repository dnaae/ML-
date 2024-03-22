import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical, np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from keras.regularizers import l1
import numpy as np

# Load the dataset
data = pd.read_csv("merged.csv")

# Drop unnecessary columns
data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'TBL','fraction_genome_altered','aneuploidy_score','TMB'], axis=1, inplace=True)

# Define target variable
target_variable = 'msi_status'
y = data[target_variable]
X = data.drop(target_variable, axis=1)

# Instantiate the encoder
le = LabelEncoder()
encoder = LabelEncoder()

# Fit and transform the features
X = X.apply(le.fit_transform)

# Fit and transform the labels
y_encoded = encoder.fit_transform(y)

# Convert labels to categorical
y_cat = to_categorical(y_encoded)

# Define the model in a function
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer 1
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(32, activation='relu', activity_regularizer=l1(0.001)))  # Hidden layer 2 with L1 regularization
    model.add(Dense(16, activation='relu'))  # Hidden layer 3
    model.add(Dense(y_cat.shape[1], activation='softmax'))  # Output layer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)

# Create a dictionary to set the class weight
class_weights = dict(enumerate(class_weights))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to the training data only
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Create model with KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=50, verbose=0)

# Perform cross-validation
scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)

print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))

# Fit the model on the training data
model.fit(X_train_smote, np_utils.to_categorical(y_train_smote), epochs=100, batch_size=50, class_weight=class_weights)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))