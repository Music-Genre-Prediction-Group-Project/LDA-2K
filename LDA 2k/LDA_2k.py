import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import os

# File path
file_path = r"C:\Users\yugio\source\repos\LDA 2k\genly_dataset_2k.csv"

# Step 1: Load and Process Data
print("Loading and processing data...")
data = []
with open(file_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        track_id = row[0]
        genre = row[1]
        word_data = row[2:]
        word_freq = {int(item.split(':')[0]): int(item.split(':')[1]) for item in word_data if item}
        data.append((track_id, genre, word_freq))

# Convert to DataFrame
columns = ['track_id', 'genre'] + [f'word_{i}' for i in range(1, 5001)]
rows = []
for track_id, genre, word_freq in data:
    row = {'track_id': track_id, 'genre': genre}
    row.update({f'word_{i}': word_freq.get(i, 0) for i in range(1, 5001)})
    rows.append(row)

df = pd.DataFrame(rows, columns=columns)

# Step 2: LDA Topic Modeling
print("Performing LDA topic modeling...")
df['text'] = df.iloc[:, 2:].apply(
    lambda row: ' '.join([f"{i+1}:{freq}" for i, freq in enumerate(row) if freq > 0]), axis=1
)

vectorizer = CountVectorizer(max_features=10000)
dtm = vectorizer.fit_transform(df['text'])

lda_model = LatentDirichletAllocation(n_components=8, random_state=42)
lda_model.fit(dtm)

topic_distributions = lda_model.transform(dtm)

# Step 3: Visualize LDA Topics
print("Visualizing LDA topic distributions...")
topic_weights = np.sum(topic_distributions, axis=0)
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(topic_weights) + 1), topic_weights, color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Topic Weight')
plt.title('Topic Distribution Across Dataset')
plt.show()

# Step 4: Prepare Data for Classification
print("Preparing data for classification...")

# Define the genre labels manually
genres = ['Country', 'Electronic', 'Metal', 'Pop', 'Punk', 'Rap', 'RnB', 'Rock']

label_encoder = LabelEncoder()
label_encoder.fit(genres)  # Fit label encoder to predefined genre list

encoded_genres = label_encoder.transform(df['genre'])  # Encode genres as integers

# Split data: Make sure the indices align
X_train, X_test, y_train, y_test = train_test_split(
    topic_distributions, encoded_genres, test_size=0.2, random_state=42
)

# Simple Classification: Assign Each Test Instance to the Dominant Topic
print("Classifying using LDA topic dominance...")

# Use topic_distributions directly, apply argmax to get the dominant topic per sample
y_pred = np.argmax(topic_distributions, axis=1)

# Ensure y_pred matches the length of y_test
y_pred = y_pred[:len(y_test)]  # Adjust if necessary to match test size

# Step 5: Generate Confusion Matrix
print("Generating confusion matrix and computing metrics...")
conf_matrix = confusion_matrix(y_test, y_pred)

# Remove rows and columns that are completely zero
non_zero_indices = ~np.all(conf_matrix == 0, axis=1)
filtered_conf_matrix = conf_matrix[non_zero_indices][:, non_zero_indices]

# Update the labels to match the filtered matrix
categories = ['Country', 'Electronic', 'Metal', 'Pop', 'Punk', 'Rap', 'RnB', 'Rock']
filtered_labels = [categories[i] for i in range(len(categories)) if non_zero_indices[i]]

# Normalize the filtered confusion matrix
normalized_conf_matrix = filtered_conf_matrix.astype('float') / filtered_conf_matrix.sum(axis=1, keepdims=True)

# Step 6: Compute Metrics
category_metrics = []

for i, label in enumerate(filtered_labels):
    # True Positives (TP): Diagonal value
    tp = filtered_conf_matrix[i, i]
    
    # False Positives (FP): Sum of the column, excluding diagonal
    fp = filtered_conf_matrix[:, i].sum() - tp
    
    # False Negatives (FN): Sum of the row, excluding diagonal
    fn = filtered_conf_matrix[i, :].sum() - tp
    
    # True Negatives (TN): Total samples - TP - FP - FN
    tn = filtered_conf_matrix.sum() - (tp + fp + fn)
    
    # Precision for this category
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Accuracy for this category
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Error rate for this category
    error_rate = 1 - accuracy
    
    category_metrics.append({
        "label": label,
        "precision": precision,
        "accuracy": accuracy,
        "error_rate": error_rate
    })

# Compute Averages
avg_precision = np.mean([m["precision"] for m in category_metrics])
avg_accuracy = np.mean([m["accuracy"] for m in category_metrics])
avg_error_rate = np.mean([m["error_rate"] for m in category_metrics])

# Step 7: Plot the Confusion Matrix and Add Metrics
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=normalized_conf_matrix, display_labels=filtered_labels)
disp.plot(cmap=plt.cm.Reds, ax=ax, colorbar=False, values_format=None)  # Suppress default text


# Add metrics text to the plot
text_x = 8.0  # Place metrics outside the confusion matrix on the left
text_y = normalized_conf_matrix.shape[0] - 8.0

# Format the per-category metrics
metrics_text = "Per-Category Metrics:\n"
for metric in category_metrics:
    metrics_text += (
        f"{metric['label']} - "
        f"Precision: {metric['precision']:.4f}, "
        f"Accuracy: {metric['accuracy']:.4f}, "
        f"Error Rate: {metric['error_rate']:.4f}\n"
    )

# Add average metrics
metrics_text += (
    f"\nAverages:\n"
    f"Precision: {avg_precision:.4f}, "
    f"Accuracy: {avg_accuracy:.4f}, "
    f"Error Rate: {avg_error_rate:.4f}"
)

# Display the metrics on the plot
ax.text(
    text_x, text_y, metrics_text,
    fontsize=12, color="black", va="top", ha="left",
    bbox=dict(boxstyle="round", edgecolor="black", facecolor="white", alpha=0.8)
)

# Customize axis labels and title
plt.title('Normalized Confusion Matrix with Metrics', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.tight_layout()