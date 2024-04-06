import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('spotify_songs.csv')

df.info()
df.head()
df.dtypes
df.describe()

genre_counts = df['playlist_genre'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Genre Distribution in Playlists')
plt.show()

sns.pairplot(df[['track_popularity', 'danceability', 'energy', 'loudness', 'valence']])
plt.show()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Create a correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='playlist_genre', y='track_popularity', data=df)
plt.title('Box Plot of Track Popularity by Playlist Genre')
plt.xticks(rotation=45)
plt.show()

average_popularity_by_genre = df.groupby('playlist_genre')['track_popularity'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='playlist_genre', y='track_popularity', data=average_popularity_by_genre, palette='viridis')
plt.title('Average Track Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Track Popularity')
plt.xticks(rotation=45)
plt.show()

sns.pairplot(df[['energy', 'danceability', 'valence', 'playlist_genre']], hue='playlist_genre', palette='viridis')
plt.show()

numeric_cols = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'duration_ms', 'track_popularity']
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='playlist_genre', y='energy', data=df, palette='pastel')
plt.title('Distribution of Energy by Genre')
plt.xlabel('Playlist Genre')
plt.ylabel('Energy')
plt.xticks(rotation=45)
plt.show()

import plotly.express as px
dfcpy = df.copy()
dfcpy['track_album_release_date'] = pd.to_datetime(dfcpy['track_album_release_date'], format='mixed')
dfcpy['year'] =pd.DatetimeIndex(dfcpy.track_album_release_date).year
fig=px.bar(dfcpy['year'].value_counts()[:10],orientation='h')
fig.show()

sns.lineplot(x='year', y='acousticness', data=dfcpy)
plt.title("Music Trends Based On Accousticness")
plt.tight_layout()
plt.show()

sns.lineplot(x='year', y='liveness', data=dfcpy)
plt.title("Music Trends Based On Liveness")
plt.tight_layout()
plt.show()

sns.lineplot(x='year', y='danceability', data=dfcpy)
plt.title("Music Trends Based On danceability")
plt.tight_layout()
plt.show()

sns.lineplot(x='year', y='loudness', data=dfcpy)
plt.title("Music Trends Based On loudness")
plt.tight_layout()
plt.show()

sns.lineplot(x='year', y='speechiness', data=dfcpy)
plt.title("Music Trends Based On speechiness")
plt.tight_layout()
plt.show()

# prompt: get the total no of artists from track_artist column by using unique values
df['track_artist'].nunique()

df['track_artist'].fillna('N/A', inplace=True)  # Fill missing values in a specific column
df['track_album_name'].fillna('N/A', inplace=True)  # Fill missing values in a specific column
df['track_name'].fillna('N/A', inplace=True)  # Fill missing values in a specific column
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder  # Import the LabelEncoder class
# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
# Create a label encoder object
le = LabelEncoder()
# Apply label encoding to each categorical column
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

#dependent variable
y=df['playlist_genre']
y

#independent variable
X=df.drop(columns=['playlist_genre'],axis=1)
X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#0.3 indicates 30% test dataset and remaining 70% training dataset which is ideal size of dataset for ml algorithms training and testing

X_train

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier  # Import the DecisionTreeClassifier class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Create a decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)  # Set random_state for reproducibility

# Train the model on the training set
dt_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt_model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')  # For multi-class classification
f1 = f1_score(y_test, y_pred, average='weighted')  # For multi-class classification


# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.tree import plot_tree  # Import plot_tree
# Create a confusion matrix:
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualize the decision tree:
dt_model.classes_ = dt_model.classes_.astype(str)
# Create the plot with adjusted figure size
fig, ax = plt.subplots(figsize=(20, 20))  # Set desired width and height (in inches)

plot_tree(dt_model, filled=True, feature_names=X_train.columns, class_names=dt_model.classes_, ax=ax)
plt.show()

# Create a larger figure with tighter layout
fig, ax = plt.subplots(figsize=(20, 20))  # Increase figure size

# Customize plot appearance for better readability
plot_tree(
    dt_model,
    filled=True,
    feature_names=X_train.columns,
    class_names=dt_model.classes_,
    ax=ax,
    fontsize=10,  # Adjust font size for nodes
    rounded=True,  # Use rounded edges for nodes
    precision=2,  # Limit displayed values to 2 decimal places
)

# Improve text alignment for better clarity
plt.title("Decision Tree Visualization", fontsize=14, y=1.05)  # Adjust title position
plt.xlabel("Feature Name", fontsize=12)
plt.ylabel("Class", fontsize=12)

# Rotate x-axis labels to prevent overlapping
plt.xticks(rotation=45, ha="right")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_predic = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predic)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)





















