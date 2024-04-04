
import pandas as pd
import numpy as np
from langdetect import detect
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def main():
    # Load data
    jobs_data = pd.read_csv("jobs.csv")
    soc_data_path = "soc2020volume1structureanddescriptionofunitgroupsexcel180523.xlsx"
    soc_descriptions_df = pd.read_excel(soc_data_path, sheet_name='SOC2020 descriptions', header=4)
    soc_framework_df = pd.read_excel(soc_data_path, sheet_name='SOC2020 Framework', header=0)

    # Adjust column names
    soc_descriptions_df.columns = [
        'SOC 2020 Major Group',
        'SOC 2020 Sub-Major Group',
        'SOC 2020 Minor Group',
        'SOC 2020 Unit Group',
        'SOC 2020 Group Title',
        'Groups Classified Within Sub-Groups',
        'Group Description',
        'Typical Entry Routes And Associated Qualifications',
        'Tasks',
        'Related Job Titles'
    ]

    # Define language detection function
    def detect_language(text):
        try:
            return detect(text)
        except Exception as e:
            return "Error detecting language"
    # Apply lang detect function on jobs_data 
    jobs_data["language"] = jobs_data["job_description"].apply(detect_language)

    # Expand 'Related_Job_Titles' into separate rows
    expanded_rows = []
    for _, row in soc_descriptions_df.iterrows():
        if pd.notna(row['Related Job Titles']):
            job_titles = row['Related Job Titles'].split('\n\n')
            for job_title in job_titles:
                job_title = job_title.strip('~')
                expanded_row = row.to_dict()
                expanded_row['Job Titles'] = job_title
                expanded_rows.append(expanded_row)
    # Convert to dataframe
    expanded_df = pd.DataFrame(expanded_rows)
    final_df = expanded_df[['SOC 2020 Group Title', 'Group Description', 'Groups Classified Within Sub-Groups', 'Related Job Titles', 'Job Titles']]
    # Combine relevent columns
    final_df_1 = final_df.copy()
    final_df_1['combined_text'] = final_df['SOC 2020 Group Title'] + " " + final_df['Group Description']
    final_df['combined_text'] = final_df["Job Titles"]  + " - " + final_df['SOC 2020 Group Title'] + " : " + "\n" + final_df['Group Description']

    # Preprocess text
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    jobs_data['combined_text'] = jobs_data['job_description'].apply(preprocess_text)
    final_df['combined_text'] = final_df['combined_text'].apply(preprocess_text)

    # Vectorize and compute similarity
    tfidf_vectorizer = TfidfVectorizer()
    soc_tfidf_matrix = tfidf_vectorizer.fit_transform(final_df['combined_text'])
    jobs_tfidf_matrix = tfidf_vectorizer.transform(jobs_data['combined_text'])
    # Cosine similarity calculation 
    cosine_sim = cosine_similarity(jobs_tfidf_matrix, soc_tfidf_matrix)
    best_matches = np.argmax(cosine_sim, axis=1)
    jobs_data['SOC Standardized Title'] = final_df.iloc[best_matches]['SOC 2020 Group Title'].values

    # Save the standardized job titles to a new CSV file
    jobs_data.to_csv('standardized_jobs.csv', index=False)

    ## Let's train and predict for new entry

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(jobs_data['job_description'], jobs_data['SOC Standardized Title'], test_size=0.2)

    # Define and train the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)

    # Example prediction
    predicted_titles = pipeline.predict(["Data Analyst"])
    print(predicted_titles)
if __name__ == "__main__":
    main()
