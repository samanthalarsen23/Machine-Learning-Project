
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('drug_consumption.data')

# giving the columns meaningful names
df.columns = [
    "ID", "Age", "Gender", "Education", "Country", "Ethnicity",
    "Nscore", "Escore", "Oscore", "Ascore", "Cscore",
    "Impulsive", "SS", "Alcohol",
    "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc",
    "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine",
    "Semer", "VSA"
]

remove_columns = [
    "Alcohol","Amphet", "Amyl", "Benzos", "Caff",  "Choc",
    "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine",
    "Semer", "VSA"
]

df = df.drop(columns = remove_columns)
df = df.drop(columns = ["ID"])

# Counts of each use class
print(df["Cannabis"].value_counts().sort_index())

# Display the percentages as well
print(df["Cannabis"].value_counts(normalize=True).sort_index() * 100)

# Visualize the imbalance in the target variable
plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Cannabis")

plt.title("Cannabis Consumption Class Distribution")
plt.xlabel("Cannabis Consumption Class")
plt.ylabel("Number of People")
plt.show()

# Separate the input features from the target

X = df.drop(columns=["Cannabis"])
y = df["Cannabis"]

# These are the continuous personality measurements
numeric_features = [
    "Nscore", "Escore", "Oscore", "Ascore",
    "Cscore", "Impulsive", "SS"
]

# Plot distributions to look for unusual values or extreme observations
X[numeric_features].hist(figsize=(12,8), bins=20)

plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.show()

# one hot encode demographics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Age", "Gender", "Country", "Education"]),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

ohEncoded = ct.fit_transform(X)
X = pd.DataFrame(ohEncoded, columns=ct.get_feature_names_out(), index=df.index)

categorical_features = [
    "Age",
    "Gender",
    "Education",
    "Country",
    "Ethnicity"
]

numeric_features = [
    "Nscore",
    "Escore",
    "Oscore",
    "Ascore",
    "Cscore",
    "Impulsive",
    "SS" ]

# Correlations are most useful for the continuous personality variables
correlation_matrix = X[numeric_features].corr()

plt.figure(figsize=(9,7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Personality Features")
plt.show()

# remove the target before clustering
X_cluster = df.drop(columns=["Cannabis"])


X_cluster = X_cluster.drop(columns=[])

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

# Compare numbers of clusters using inertia
from sklearn.cluster import KMeans
inertias = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(2,9), inertias, marker="o")

plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# compare numbers of clusters with silhouette score
from sklearn.metrics import silhouette_score

silhouette_scores = []
for n_clusters in range(2, 9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(range(2, 9), silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


features = [
    "Age", "Gender", "Education", "Country", "Ethnicity",
    "Nscore", "Escore", "Oscore", "Ascore", "Cscore",
    "Impulsive", "SS"
]

X_cluster = df[features]

# Standardize the features
X_scaled = StandardScaler().fit_transform(X_cluster)

# Group participants into two clusters.
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Reduce the features to two dimensions for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters.
plt.figure(figsize=(9, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    alpha=0.7
)

plt.title("K-Means Clusters Without Cannabis")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

df_c = df[features].copy()
df_c["cluster"] = clusters

# Mean of each feature per cluster, in standardized units
profile = (pd.DataFrame(X_scaled, columns=features)
             .groupby(clusters).mean().round(2))
print(profile.T)

# Which features differ most between clusters
spread = (profile.max() - profile.min()).sort_values(ascending=False)
print(spread)

import seaborn as sns
plt.figure(figsize=(7, 6))
sns.heatmap(profile.T, annot=True, cmap="RdBu_r", center=0)
plt.title("Cluster profiles (standardized means)")
plt.show()

# Separate the input features from target features
X = df.drop(columns=["Cannabis", "Cluster"])
y = df["Cannabis"]



# Split the data into training and testing sets.
# Stratify keeps the proportions of use classes similar in both sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Random Forest Classifier
n_estimators=200
rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion matric
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]
)

fig, ax = plt.subplots(figsize=(9, 8))
disp.plot(ax=ax)

plt.title("Random Forest: Cannabis Consumption Predictions")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

rf_classifier.feature_importances_

rf_classifier.feature_names_in_

# Chart feature importances
plt.barh(rf_classifier.feature_names_in_, rf_classifier.feature_importances_)
plt.title("Random Forest: Feature Importances")