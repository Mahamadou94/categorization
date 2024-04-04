import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder


def main():
    
    # Load the dataset
    df = pd.read_csv('aac_intakes_outcomes.csv')
    # Prepocess the data and handle NAN values
    df = df.dropna(subset=['outcome_type'])
    df = df.dropna(subset=['sex_upon_intake'])
    most_frequent_sex = df['sex_upon_outcome'].mode()[0]
    df['sex_upon_outcome'] = df['sex_upon_outcome'].fillna(most_frequent_sex)
    
    # Convert date field to datetime
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])

    # Calcul the duration in shelter days
    df['duration_in_shelter_days'] = (df['outcome_datetime'] - df['intake_datetime']).dt.total_seconds() / (24 * 60 * 60)

    # Encode categorical
    df = pd.get_dummies(df, columns=['animal_type', 'breed', 'color', 'intake_condition', 'sex_upon_intake'])
    
    # Drop 'outcome_subtype' due to NaN values 
    df = df.drop('outcome_subtype', axis=1).dropna(subset=['outcome_type'])
    
    # Define features and target variable
    X = df.drop('outcome_type', axis=1)
    y = df['outcome_type']

    # Identify categorical and numerical columns (excluding the target variable 'outcome_type')
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Define preprocessing for numerical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical columns 
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Define the model with RFE
    rfe = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42), step=1, cv=5, scoring='accuracy')
    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor)
                        ('feature_selection', rfe),
                        ('model', model)])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, preds))

    # model file name
    model_filename = 'model_file.pkl'  

    # Save the model to a .pkl file
    joblib.dump(clf, model_filename)

    print(f'Model saved to {model_filename}')
    
    
if __name__ =="__main__":
    main()

