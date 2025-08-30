get_ipython().run_cell_magic('capture', '', '!gdown 1CMcTsxSzz6vzvrq_OIwOJFtowI3KHPOa\n!gdown 1JnaVPaqDA60zpT7gkq6VtIPf8nlBGi87\n!unzip Timeline20250531.zip\n')
import os
len(os.listdir('Timeline20250531/cliplets/'))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image_folder = 'Timeline20250531/cliplets/'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
num_images_to_display = min(5, len(image_files))
plt.figure(figsize=(15, 5))
for i in range(num_images_to_display):
    img_path = image_files[i]
    img = mpimg.imread(img_path)
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import cv2
def preprocess_image(image_path, size=(64, 64)):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    img_resized = cv2.resize(img_bin, size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized.flatten()
image_data = []
for img_file in image_files:
    try:
        processed_img = preprocess_image(img_file)
        image_data.append(processed_img)
    except Exception as e:
        print(f"Error processing image {img_file}: {e}")
image_data = np.array(image_data)
pca = PCA(n_components=400)
image_data_pca = pca.fit_transform(image_data)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()
print(f"Cumulative explained variance with {pca.n_components} components: {cumulative_explained_variance[-1]:.4f}")
import pandas as pd
filenames = os.listdir('Timeline20250531/cliplets/')
data = pd.DataFrame({'filename': filenames})
data['letter'] = data.filename.apply(lambda x: x.split('_')[0])
data['TM'] = data.filename.apply(lambda x: int(x.split('_')[1]))
data['number'] = data.filename.apply(lambda x: x.split('_')[2].split('.')[0])
print(data.shape[0])
data.sample(5)
metadata = pd.read_csv('metadata.csv')
print(metadata.shape[0])
metadata.head()
data['year'] = data.TM.apply(lambda x: metadata.loc[metadata['TM'] == x]['Year ante quem'].values[0])
data['region'] = data.TM.apply(lambda x: metadata.loc[metadata['TM'] == x]['Production Nome (supposed)'].values[0])
data.sample(5)
from sklearn.cluster import KMeans
n_clusters = 24
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(image_data_pca)
pd.Series(clusters).value_counts().plot.bar(figsize=(10,2));
sns.despine(left=True, bottom=True)
for cluster_id in range(n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    num_images_in_cluster = len(cluster_indices)
    print(f"Cluster {cluster_id} ({num_images_in_cluster} images):")
    images_to_display = cluster_indices[:min(10, num_images_in_cluster)]
    if images_to_display.size > 0:
        plt.figure(figsize=(5, 10))
        for i, img_index in enumerate(images_to_display):
            img_path = image_files[img_index]
            img = Image.open(img_path).convert('RGB').resize((12,12))
            plt.subplot(1, len(images_to_display), i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No images in this cluster to display.")
def create_clustered_image_grid(image_files, clusters, n_clusters, images_per_cluster_row=5):
    cluster_image_indices = [np.where(clusters == i)[0] for i in range(n_clusters)]
    small_img_size = (64, 64)
    display_img_size = (64, 64)
    margin = 10
    row_width = images_per_cluster_row * display_img_size[0] + (images_per_cluster_row - 1) * margin
    row_height = display_img_size[1]
    total_height = n_clusters * row_height + (n_clusters - 1) * margin
    total_width = row_width
    final_image = Image.new('RGB', (total_width, total_height), color='white')
    y_offset = 0
    for cluster_id in range(n_clusters):
        cluster_indices = cluster_image_indices[cluster_id]
        images_to_display_indices = cluster_indices[:images_per_cluster_row]
        x_offset = 0
        for img_index in images_to_display_indices:
            try:
                img_path = image_files[img_index]
                img = Image.open(img_path).convert('RGB').resize(display_img_size)
                final_image.paste(img, (x_offset, y_offset))
                x_offset += display_img_size[0] + margin
            except Exception as e:
                print(f"Could not paste image {img_path}: {e}")
                x_offset += display_img_size[0] + margin
        y_offset += row_height + margin
    actual_height = y_offset - margin if y_offset > margin else 0
    actual_width = total_width
    max_actual_row_width = 0
    for cluster_id in range(n_clusters):
        images_in_row = min(len(cluster_image_indices[cluster_id]), images_per_cluster_row)
        actual_row_width = images_in_row * display_img_size[0] + (images_in_row - 1) * margin
        max_actual_row_width = max(max_actual_row_width, actual_row_width)
    actual_width = max_actual_row_width
    final_image_cropped = final_image.crop((0, 0, actual_width, actual_height))
    return final_image_cropped
clustered_image_grid = create_clustered_image_grid(image_files, clusters, n_clusters, images_per_cluster_row=7)
clustered_image_grid.save("clustered_images_grid.png")
print("Clustered image grid saved as clustered_images_grid.png")
plt.figure(figsize=(15, 12))
scatter = plt.scatter(image_data_pca[:, 0], image_data_pca[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.5)
plt.grid(True)
cluster_centers = kmeans.cluster_centers_
for i in range(n_clusters):
    indices_in_cluster = np.where(clusters == i)[0]
    if len(indices_in_cluster) > 0:
        cluster_points_2d = image_data_pca[indices_in_cluster]
        center_2d = cluster_centers[i]
        distances = np.linalg.norm(cluster_points_2d - center_2d, axis=1)
        closest_image_index_in_cluster = indices_in_cluster[np.argmin(distances)]
        closest_image_path = image_files[closest_image_index_in_cluster]
        try:
            img_thumb = Image.open(closest_image_path).convert('RGB')
            img_thumb = img_thumb.resize((32, 32))
            img_thumb_np = np.array(img_thumb)
            center_x, center_y = cluster_centers[i][:2]
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            imagebox = OffsetImage(img_thumb_np, zoom=1.0)
            ab = AnnotationBbox(imagebox, (center_x, center_y), frameon=False, pad=0.1)
            plt.gca().add_artist(ab)
        except Exception as e:
            print(f"Error processing image for annotation {closest_image_path}: {e}")
sns.despine(left=True, bottom=True)
plt.savefig('image_clusters.pdf', dpi=300, format='PDF')
plt.show()
cluster_df = pd.DataFrame({'filename': [os.path.basename(f) for f in image_files], 'cluster_label': clusters})
cluster_df = pd.merge(data, cluster_df, on='filename', how='left')
cluster_df.sample(3)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
ground_truth_labels = data['letter'].values
nmi = normalized_mutual_info_score(ground_truth_labels, clusters)
print(f"Normalized Mutual Information (NMI): {nmi}")
ari = adjusted_rand_score(ground_truth_labels, clusters)
print(f"Adjusted Rand Index (ARI): {ari}")
from sklearn.metrics import silhouette_score, silhouette_samples
sample_silhouette_values = silhouette_samples(image_data_pca, clusters)
average_silhouette = silhouette_score(image_data_pca, clusters)
print(f"Average Silhouette Score (Micro): {average_silhouette}")
ground_truth_labels = data['letter'].values
nmi = normalized_mutual_info_score(ground_truth_labels, clusters)
print(f"Normalized Mutual Information (NMI): {nmi}")
ari = adjusted_rand_score(ground_truth_labels, clusters)
print(f"Adjusted Rand Index (ARI): {ari}")
if len(np.unique(clusters)) > 1:
    average_silhouette = silhouette_score(image_data_pca, clusters)
    print(f"Average Silhouette Score: {average_silhouette}")
else:
    print("Silhouette Score not applicable for a single cluster.")
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
homogeneity = homogeneity_score(ground_truth_labels, clusters)
completeness = completeness_score(ground_truth_labels, clusters)
v_measure = v_measure_score(ground_truth_labels, clusters)
print(f"Homogeneity: {homogeneity}")
print(f"Completeness: {completeness}")
print(f"V-measure: {v_measure}")
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
import numpy as np
def hungarian_accuracy(ground_truth, predictions):
  """
    Computes the Hungarian accuracy for clustering.
    Parameters:
    ground_truth (array-like): True labels (numeric).
    predictions (array-like): Predicted cluster labels (numeric).
    Returns:
    float: Hungarian accuracy.
  """
  ground_truth = np.asarray(ground_truth)
  predictions = np.asarray(predictions)
  cm = confusion_matrix(ground_truth, predictions)
  cost_matrix = -cm
  row_ind, col_ind = linear_sum_assignment(cost_matrix)
  optimal_assignment_sum = -cost_matrix[row_ind, col_ind].sum()
  total_samples = cm.sum()
  accuracy = optimal_assignment_sum / total_samples
  return accuracy
label_encoder_clustering = LabelEncoder()
ground_truth_labels_encoded = label_encoder_clustering.fit_transform(data['letter'].values)
hungarian_acc = hungarian_accuracy(ground_truth_labels_encoded, clusters)
print(f"Hungarian Accuracy: {hungarian_acc}")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(ground_truth_labels_encoded, clusters)
true_labels_unique_encoded = np.unique(ground_truth_labels_encoded)
cluster_labels_unique = np.unique(clusters)
if cm.shape[0] == len(true_labels_unique_encoded) and cm.shape[1] == len(cluster_labels_unique):
  cm_reordered = cm[:, col_ind]
  encoded_to_letter_map = {i: label for i, label in enumerate(label_encoder_clustering.classes_)}
  cluster_to_true_label_map = {cluster_labels_unique[col_ind[i]]: encoded_to_letter_map[true_labels_unique_encoded[row_ind[i]]] for i in range(len(row_ind))}
  reordered_cluster_labels = [f'Cluster {cluster_labels_unique[col_ind[j]]} (Matched to: {encoded_to_letter_map[true_labels_unique_encoded[row_ind[j]]]})' for j in range(cm_reordered.shape[1])]
  true_label_names = label_encoder_clustering.classes_.tolist()
  true_label_names_ordered = [encoded_to_letter_map[i] for i in true_labels_unique_encoded]
  sns.heatmap(cm_reordered, annot=True, fmt='d', cmap='Blues',
              xticklabels=reordered_cluster_labels, yticklabels=true_label_names_ordered)
  plt.xlabel('Predicted Cluster (Optimally Matched to True Label)')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix for Clustering Evaluation (Columns Reordered by Hungarian Assignment)')
  plt.tight_layout()
  plt.show()
else:
  print("Error: Shape of confusion matrix does not match the number of unique labels.")
  print("Confusion Matrix Shape:", cm.shape)
  print("Number of unique true labels (encoded):", len(true_labels_unique_encoded))
  print("Number of unique cluster labels:", len(cluster_labels_unique))
  plt.figure(figsize=(12, 10))
  true_label_names = label_encoder_clustering.classes_.tolist()
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=cluster_labels_unique, yticklabels=true_label_names)
  plt.xlabel('Predicted Cluster')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix for Clustering Evaluation')
  plt.tight_layout()
  plt.savefig('clustering_confusion_matrix.pdf', dpi=300, format='PDF')
  plt.show()
plt.figure(figsize=(12, 10))
true_label_names = label_encoder_clustering.classes_.tolist()
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cluster_labels_unique, yticklabels=true_label_names)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.tight_layout()
plt.savefig('clustering_confusion_matrix.pdf', dpi=300, format='PDF')
plt.show()
cluster_silhouette_scores = []
for i in range(n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    if len(cluster_indices) > 1:
        cluster_silhouette_scores.append(np.mean(sample_silhouette_values[cluster_indices]).item())
macro_silhouette = np.mean(cluster_silhouette_scores) if cluster_silhouette_scores else 0
print(f"Macro Silhouette Score (Average of Cluster Averages): {macro_silhouette}")
dict(zip(range(n_clusters), cluster_silhouette_scores))
cluster_purity = {}
cluster_most_represented_letter = {}
for cluster_id in range(n_clusters):
    indices_in_cluster = np.where(clusters == cluster_id)[0]
    if len(indices_in_cluster) > 0:
        letters_in_cluster = ground_truth_labels[indices_in_cluster]
        letter_counts = pd.Series(letters_in_cluster).value_counts()
        most_frequent_letter = letter_counts.index[0]
        most_frequent_count = letter_counts.iloc[0]
        purity = most_frequent_count / len(indices_in_cluster)
        cluster_purity[cluster_id] = purity
        cluster_most_represented_letter[cluster_id] = most_frequent_letter
purity_df = pd.DataFrame({
    'Cluster': list(cluster_purity.keys()),
    'Purity': [cluster_purity[k] for k in cluster_purity.keys()],
    'Letter': [cluster_most_represented_letter[k] for k in cluster_purity.keys()]
})
purity_df.sort_values(by='Purity', ascending=False, inplace=True)
purity_df.head()
plt.figure(figsize=(10, 8))
sns.barplot(x='Purity', y='Cluster', data=purity_df, orient='h', color='skyblue')
plt.yticks(ticks=purity_df.index, fontsize=18, labels=[f'{c} ({l})' for c, l in zip(purity_df['Cluster'], purity_df['Letter'])])
plt.xlabel('Purity', fontsize=14)
plt.ylabel('Cluster (Majority Letter)', fontsize=18)
plt.grid(axis='x')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('cluster_purity.pdf', dpi=300, format='PDF')
from sklearn.cluster import SpectralClustering
n_clusters_spectral = 24
spectral = SpectralClustering(n_clusters=n_clusters_spectral,
                              assign_labels='discretize',
                              affinity='nearest_neighbors',
                              n_neighbors=15,
                              random_state=42,
                              n_init=10
                              )
clusters_spectral = spectral.fit_predict(image_data_pca)
print("Spectral Clustering completed.")
pd.Series(clusters_spectral).value_counts().plot.bar(figsize=(10,2));
sns.despine(left=True, bottom=True)
for cluster_id in range(n_clusters_spectral):
    cluster_indices = np.where(clusters_spectral == cluster_id)[0]
    num_images_in_cluster = len(cluster_indices)
    print(f"Spectral Cluster {cluster_id} ({num_images_in_cluster} images):")
    images_to_display = cluster_indices[:min(10, num_images_in_cluster)]
    if images_to_display.size > 0:
        plt.figure(figsize=(5, 10))
        for i, img_index in enumerate(images_to_display):
            img_path = image_files[img_index]
            img = Image.open(img_path).convert('RGB').resize((12,12))
            plt.subplot(1, len(images_to_display), i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No images in this cluster to display.")
clustered_image_grid_spectral = create_clustered_image_grid(image_files, clusters_spectral, n_clusters_spectral, images_per_cluster_row=7)
clustered_image_grid_spectral.save("clustered_images_grid_spectral.png")
print("Clustered image grid for Spectral Clustering saved as clustered_images_grid_spectral.png")
plt.figure(figsize=(15, 12))
scatter = plt.scatter(image_data_pca[:, 0], image_data_pca[:, 1], c=clusters_spectral, cmap='viridis', s=10, alpha=0.5)
plt.title('Image Clusters Visualization with Spectral Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
sns.despine(left=True, bottom=True)
plt.savefig('image_clusters_spectral.pdf', dpi=300, format='PDF')
plt.show()
cluster_df_spectral = pd.DataFrame({'filename': [os.path.basename(f) for f in image_files], 'cluster_label_spectral': clusters_spectral})
data_with_spectral_clusters = pd.merge(data, cluster_df_spectral, on='filename', how='left')
nmi_spectral = normalized_mutual_info_score(ground_truth_labels, clusters_spectral)
print(f"Spectral Clustering Normalized Mutual Information (NMI): {nmi_spectral}")
ari_spectral = adjusted_rand_score(ground_truth_labels, clusters_spectral)
print(f"Spectral Clustering Adjusted Rand Index (ARI): {ari_spectral}")
if len(np.unique(clusters_spectral)) > 1:
    average_silhouette_spectral = silhouette_score(image_data_pca, clusters_spectral)
    print(f"Spectral Clustering Average Silhouette Score: {average_silhouette_spectral}")
    sample_silhouette_values_spectral = silhouette_samples(image_data_pca, clusters_spectral)
else:
    print("Spectral Clustering Silhouette Score not applicable for a single cluster.")
    sample_silhouette_values_spectral = None
homogeneity_spectral = homogeneity_score(ground_truth_labels, clusters_spectral)
completeness_spectral = completeness_score(ground_truth_labels, clusters_spectral)
v_measure_spectral = v_measure_score(ground_truth_labels, clusters_spectral)
print(f"Spectral Clustering Homogeneity: {homogeneity_spectral}")
print(f"Spectral Clustering Completeness: {completeness_spectral}")
print(f"Spectral Clustering V-measure: {v_measure_spectral}")
hungarian_acc_spectral = hungarian_accuracy(ground_truth_labels_encoded, clusters_spectral)
print(f"Spectral Clustering Hungarian Accuracy: {hungarian_acc_spectral}")
plt.figure(figsize=(12, 10))
cm_spectral = confusion_matrix(ground_truth_labels_encoded, clusters_spectral)
true_labels_unique_encoded_spectral = np.unique(ground_truth_labels_encoded)
cluster_labels_unique_spectral = np.unique(clusters_spectral)
if cm_spectral.shape[0] == len(true_labels_unique_encoded_spectral) and cm_spectral.shape[1] == len(cluster_labels_unique_spectral):
  cost_matrix_spectral = -cm_spectral
  row_ind_spectral, col_ind_spectral = linear_sum_assignment(cost_matrix_spectral)
  cm_reordered_spectral = cm_spectral[:, col_ind_spectral]
  encoded_to_letter_map_spectral = {i: label for i, label in enumerate(label_encoder_clustering.classes_)}
  reordered_cluster_labels_spectral = [f'Cluster {cluster_labels_unique_spectral[col_ind_spectral[j]]} (Matched to: {encoded_to_letter_map_spectral[true_labels_unique_encoded_spectral[row_ind_spectral[j]]]})' for j in range(cm_reordered_spectral.shape[1])]
  true_label_names_spectral = label_encoder_clustering.classes_.tolist()
  true_label_names_ordered_spectral = [encoded_to_letter_map_spectral[i] for i in true_labels_unique_encoded_spectral]
  sns.heatmap(cm_reordered_spectral, annot=True, fmt='d', cmap='Blues',
              xticklabels=reordered_cluster_labels_spectral, yticklabels=true_label_names_ordered_spectral)
  plt.xlabel('Predicted Cluster (Optimally Matched to True Label)')
  plt.ylabel('True Label')
  plt.title('Spectral Clustering Confusion Matrix (Columns Reordered)')
  plt.tight_layout()
  plt.show()
else:
  print("Error: Shape of spectral confusion matrix does not match the number of unique labels.")
  plt.figure(figsize=(12, 10))
  true_label_names_spectral = label_encoder_clustering.classes_.tolist()
  sns.heatmap(cm_spectral, annot=True, fmt='d', cmap='Blues',
              xticklabels=cluster_labels_unique_spectral, yticklabels=true_label_names_spectral)
  plt.xlabel('Predicted Cluster')
  plt.ylabel('True Label')
  plt.title('Spectral Clustering Confusion Matrix')
  plt.tight_layout()
  plt.show()
cluster_silhouette_scores_spectral = []
if sample_silhouette_values_spectral is not None:
    for i in range(n_clusters_spectral):
        cluster_indices = np.where(clusters_spectral == i)[0]
        if len(cluster_indices) > 1:
            cluster_silhouette_scores_spectral.append(np.mean(sample_silhouette_values_spectral[cluster_indices]).item())
    macro_silhouette_spectral = np.mean(cluster_silhouette_scores_spectral) if cluster_silhouette_scores_spectral else 0
    print(f"Spectral Clustering Macro Silhouette Score: {macro_silhouette_spectral}")
else:
    print("Spectral Clustering Per cluster Silhouette scores not applicable.")
cluster_purity_spectral = {}
cluster_most_represented_letter_spectral = {}
for cluster_id in range(n_clusters_spectral):
    indices_in_cluster = np.where(clusters_spectral == cluster_id)[0]
    if len(indices_in_cluster) > 0:
        letters_in_cluster = ground_truth_labels[indices_in_cluster]
        letter_counts = pd.Series(letters_in_cluster).value_counts()
        most_frequent_letter = letter_counts.index[0]
        most_frequent_count = letter_counts.iloc[0]
        purity = most_frequent_count / len(indices_in_cluster)
        cluster_purity_spectral[cluster_id] = purity
        cluster_most_represented_letter_spectral[cluster_id] = most_frequent_letter
purity_df_spectral = pd.DataFrame({
    'Cluster': list(cluster_purity_spectral.keys()),
    'Purity': [cluster_purity_spectral[k] for k in cluster_purity_spectral.keys()],
    'Letter': [cluster_most_represented_letter_spectral[k] for k in cluster_purity_spectral.keys()]
})
purity_df_spectral.sort_values(by='Purity', ascending=False, inplace=True)
print("\nSpectral Clustering Cluster Purity:")
print(purity_df_spectral.head())
plt.figure(figsize=(10, 8))
sns.barplot(x='Purity', y='Cluster', data=purity_df_spectral, orient='h', color='lightcoral')
plt.yticks(ticks=purity_df_spectral.index, fontsize=18, labels=[f'{c} ({l})' for c, l in zip(purity_df_spectral['Cluster'], purity_df_spectral['Letter'])])
plt.xlabel('Purity', fontsize=14)
plt.ylabel('Cluster (Majority Letter)', fontsize=18)
plt.grid(axis='x')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('cluster_purity_spectral.pdf', dpi=300, format='PDF')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
X = image_data
train_data = data[data['letter'] != 'Other'].copy()
train_X = X[data['letter'] != 'Other']
train_y = train_data['letter']
unk_data = data[data['letter'] == 'Other'].copy()
unk_X = X[data['letter'] == 'Other']
unk_y = unk_data['letter']
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y_encoded, test_size=0.2, random_state=42, stratify=train_y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_unk_tensor = torch.tensor(unk_X, dtype=torch.float32)
X_train.shape
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
class CNN1DDeep(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DDeep, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            dummy_output = self.pool3(self.relu(self.conv3(self.pool2(self.relu(self.conv2(self.pool1(self.relu(self.conv1(dummy_input)))))))))
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
class CNN2D(nn.Module):
    def __init__(self, num_classes, image_size=(64, 64)):
        super(CNN2D, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, image_size[0], image_size[1])
            dummy_output = self.pool3(self.relu(self.conv3(self.pool2(self.relu(self.conv2(self.pool1(self.relu(self.conv1(dummy_input)))))))))
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    def get_embeddings(self, x):
        """Return embeddings before the final classification layer"""
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x
data_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.5, value=0.5)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def preprocess_image_2d(image_path, size=(64, 64)):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    img_resized = cv2.resize(img_bin, size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized
image_data_2d = []
for img_file in image_files:
    try:
        processed_img_2d = preprocess_image_2d(img_file)
        image_data_2d.append(processed_img_2d)
    except Exception as e:
        print(f"Error processing image {img_file} for 2D CNN: {e}")
image_data_2d = np.array(image_data_2d)
train_data_split = data[data['letter'] != 'Other'].copy()
unk_data_split = data[data['letter'] == 'Other'].copy()
train_indices_2d = train_data_split.index.tolist()
unk_indices_2d = unk_data_split.index.tolist()
label_encoder = LabelEncoder()
train_indices_2d, test_indices_2d, y_train_encoded_2d, y_test_encoded_2d = train_test_split(
    train_indices_2d,
    label_encoder.fit_transform(train_data_split['letter']),
    test_size=0.2,
    random_state=42,
    stratify=label_encoder.fit_transform(train_data_split['letter'])
)
train_indices_2d, val_indices_2d, y_train_encoded_2d, y_val_encoded_2d = train_test_split(
    train_indices_2d,
    y_train_encoded_2d,
    test_size=0.1,
    random_state=42,
    stratify=y_train_encoded_2d
)
X_train_2d = image_data_2d[train_indices_2d]
y_train_2d = y_train_encoded_2d
X_val_2d = image_data_2d[val_indices_2d]
y_val_2d = y_val_encoded_2d
X_test_2d = image_data_2d[test_indices_2d]
y_test_2d = y_test_encoded_2d
X_unk_2d = image_data_2d[unk_indices_2d]
class ImageDatasetAugmented(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = self.X[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        if self.y is not None:
            return image, self.y[idx]
        else:
            return image
batch_size = 16
train_loader_2d_aug = DataLoader(ImageDatasetAugmented(X_train_2d, y_train_2d, transform=data_transform), batch_size=batch_size, shuffle=True)
val_loader_2d = DataLoader(ImageDatasetAugmented(X_val_2d, y_val_2d, transform=test_transform), batch_size=batch_size)
test_loader_2d = DataLoader(ImageDatasetAugmented(X_test_2d, y_test_2d, transform=test_transform), batch_size=batch_size)
unk_loader_2d = DataLoader(ImageDatasetAugmented(X_unk_2d, transform=test_transform), batch_size=batch_size)
num_classes_2d = len(label_encoder.classes_)
image_input_size = (64, 64)
model = CNN2D(num_classes=num_classes_2d, image_size=image_input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
num_epochs = 100
train_losses = []
val_losses = []
val_accuracies = []
print("Starting 2D CNN Training with Augmentation...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader_2d_aug:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader_2d_aug.dataset)
    train_losses.append(epoch_loss)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader_2d:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_val_loss = val_loss / len(val_loader_2d.dataset)
    val_accuracy = correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_cnn_letter_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
print("2D CNN Training with Augmentation finished.")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss for 2D CNN with Augmentation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy for 2D CNN with Augmentation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
model.eval()
test_loss = 0.0
correct = 0
total = 0
predicted_labels_2d = []
true_labels_test_2d = []
with torch.no_grad():
    for inputs, labels in test_loader_2d:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels_2d.extend(predicted.cpu().numpy())
        true_labels_test_2d.extend(labels.cpu().numpy())
final_test_loss_2d = test_loss / len(test_loader_2d.dataset)
test_accuracy_2d = correct / total
print(f'2D CNN with Augmentation Test Loss: {final_test_loss_2d:.4f}, Test Accuracy: {test_accuracy_2d:.4f}')
predicted_letters_2d = label_encoder.inverse_transform(predicted_labels_2d)
true_letters_test_2d = label_encoder.inverse_transform(true_labels_test_2d)
print("\n2D CNN with Augmentation Classification Report on Test Set:")
print(classification_report(true_letters_test_2d, predicted_letters_2d))
cm_2d = confusion_matrix(true_letters_test_2d, predicted_letters_2d, labels=label_encoder.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_2d, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('2D CNN with Augmentation Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig('cnn2d_aug_confusion_matrix.pdf', dpi=300, format='PDF')
plt.show()
model.eval()
unk_predictions_2d = []
unk_filenames_2d = []
unk_data_filenames = unk_data_split['filename'].tolist()
with torch.no_grad():
    for i in range(len(unk_loader_2d.dataset)):
        inputs = unk_loader_2d.dataset[i].unsqueeze(0).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        unk_predictions_2d.extend(predicted.cpu().numpy())
predicted_unk_letters_2d = label_encoder.inverse_transform(unk_predictions_2d)
unk_predictions_df_2d = pd.DataFrame({
    'filename': unk_data_filenames,
    'predicted_letter_2d_aug': predicted_unk_letters_2d
})
print("\n2D CNN with Augmentation Predictions for 'Other' images:")
print(unk_predictions_df_2d.head())
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
y_true_letters = label_encoder.inverse_transform(true_labels_test_2d)
y_pred_letters = label_encoder.inverse_transform(predicted_labels_2d)
labels_order = sorted(list(set(y_true_letters)))
cm = confusion_matrix(y_true_letters, y_pred_letters, labels=labels_order)
plt.figure(figsize=(14, 12))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.tight_layout()
plt.savefig('confusion_matrix.pdf', format='pdf', dpi=300)
from sklearn.metrics import classification_report
print("\nClassification Report on Test Dataset:")
print(classification_report(y_true_letters, y_pred_letters, labels=labels_order, zero_division=0))
from sklearn.metrics import classification_report
report = classification_report(y_true_letters, y_pred_letters, output_dict=True, zero_division=0)
latex_table = "\\begin{tabular}{|l|c|c|c|c|}\n"
latex_table += "\\hline\n"
latex_table += "Class & Precision & Recall & F1-Score & Support \\\\\n"
latex_table += "\\hline\n"
for label, metrics in report.items():
    if isinstance(metrics, dict):
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = metrics['support']
        latex_table += f"{label} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {support} \\\\\n"
latex_table += "\\hline\n"
macro_avg = report['macro avg']
macro_precision = macro_avg['precision']
macro_recall = macro_avg['recall']
macro_f1 = macro_avg['f1-score']
macro_support = macro_avg['support']
latex_table += f"Macro Avg & {macro_precision:.3f} & {macro_recall:.3f} & {macro_f1:.3f} & {macro_support} \\\\\n"
latex_table += "\\hline\n"
weighted_avg = report['weighted avg']
weighted_precision = weighted_avg['precision']
weighted_recall = weighted_avg['recall']
weighted_f1 = weighted_avg['f1-score']
weighted_support = weighted_avg['support']
latex_table += f"Weighted Avg & {weighted_precision:.3f} & {weighted_recall:.3f} & {weighted_f1:.3f} & {weighted_support} \\\\\n"
latex_table += "\\hline\n"
accuracy = report['accuracy']
latex_table += f"Accuracy & \\multicolumn{{3}}{{|c|}}{{{accuracy:.3f}}} & {weighted_support} \\\\\n"
latex_table += "\\hline\n"
latex_table += "\\end{tabular}"
print("LaTeX table for classification report:")
latex_table
model = CNN2D(num_classes=len(label_encoder.classes_), image_size=(64, 64))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch
                labels.extend(targets.numpy())
            else:
                images = batch
                targets = None
            images = images.to(device)
            emb = model.get_embeddings(images)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    if labels:
        labels = np.array(labels)
        return embeddings, labels
    else:
        return embeddings
train_embeddings, train_labels = extract_embeddings(model, train_loader_2d_aug, device)
test_embeddings, test_labels = extract_embeddings(model, test_loader_2d, device)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=11, metric='cosine', weights='distance')
knn.fit(train_embeddings, train_labels)
y_pred = knn.predict(test_embeddings)
print("\nClassification Report of K-NN on Test Dataset:")
print(classification_report(label_encoder.inverse_transform(test_labels), label_encoder.inverse_transform(y_pred), zero_division=0))
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
test_letters = label_encoder.inverse_transform(test_labels)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
embeddings_2d = tsne.fit_transform(test_embeddings)
plt.figure(figsize=(12, 10))
for letter in np.unique(test_letters):
    idx = test_letters == letter
    plt.scatter(
        embeddings_2d[idx, 0], embeddings_2d[idx, 1],
        label=letter, alpha=0.7, s=40
    )
plt.legend(title="Letter", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("t-SNE visualization of CNN embeddings (letters)")
plt.show()
def extract_embeddings_tta(model, dataset, device, n_aug=5):
    """
    Extract embeddings with Test-Time Augmentation (TTA).
    Args:
        model: CNN2D model
        dataset: torch Dataset (not DataLoader)
        device: torch device
        n_aug: number of augmentations per sample
    Returns:
        embeddings (N, D), labels (N,)
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            aug_embs = []
            for _ in range(n_aug):
                aug_img, _ = dataset[i]
                aug_img = aug_img.unsqueeze(0).to(device)
                emb = model.get_embeddings(aug_img)
                aug_embs.append(emb.cpu().numpy())
            avg_emb = np.mean(aug_embs, axis=0)
            embeddings.append(avg_emb)
            labels.append(label)
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels
test_dataset_aug = ImageDatasetAugmented(X_test_2d, y_test_2d, transform=data_transform)
test_embeddings_tta, test_labels_tta = extract_embeddings_tta(model, test_dataset_aug, device, n_aug=5)
train_embeddings, train_labels = extract_embeddings(model, train_loader_2d_aug, device)
knn = KNeighborsClassifier(n_neighbors=17, metric="cosine", weights='distance')
knn.fit(train_embeddings, train_labels)
y_pred_tta = knn.predict(test_embeddings_tta)
print("\nClassification Report of K-NN on Test Dataset:")
print(classification_report(label_encoder.inverse_transform(test_labels_tta), label_encoder.inverse_transform(y_pred_tta), zero_division=0))
import matplotlib.pyplot as plt
def visualize_tta_predictions(model, knn, dataset, idx, device, n_aug=5):
    """
    Show augmentations of one test image and their KNN classifications.
    Args:
        model: trained CNN
        knn: fitted KNN classifier
        dataset: Dataset with augmentation transforms
        idx: dataset index of the test image
        device: torch.device
        n_aug: number of augmentations to visualize
    """
    model.eval()
    aug_images = []
    aug_preds = []
    aug_embs = []
    with torch.no_grad():
        for _ in range(n_aug):
            img, label = dataset[idx]
            aug_images.append(img.squeeze(0).numpy())
            img = img.unsqueeze(0).to(device)
            emb = model.get_embeddings(img).cpu().numpy()
            aug_embs.append(emb)
            pred = knn.predict(emb)[0]
            aug_preds.append(pred)
    aug_embs = np.vstack(aug_embs)
    avg_emb = aug_embs.mean(axis=0, keepdims=True)
    avg_pred = knn.predict(avg_emb)[0]
    true_label = dataset[idx][1]
    true_letter = label_encoder.inverse_transform([true_label])[0]
    fig, axes = plt.subplots(1, n_aug, figsize=(2*n_aug, 3))
    for i, ax in enumerate(axes):
        ax.imshow(aug_images[i], cmap="gray")
        pred_letter = label_encoder.inverse_transform([aug_preds[i]])[0]
        ax.set_title(pred_letter, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"TTA predictions (true={true_letter}, avg_pred={label_encoder.inverse_transform([avg_pred])[0]})", fontsize=14)
    plt.show()
test_dataset_aug = ImageDatasetAugmented(X_test_2d, y_test_2d, transform=data_transform)
visualize_tta_predictions(model, knn, test_dataset_aug, idx=1, device=device, n_aug=5)
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
train_embeddings, train_labels = extract_embeddings(model, train_loader_2d_aug, device)
test_embeddings, test_labels = extract_embeddings(model, test_loader_2d, device)
num_classes = len(np.unique(train_labels))
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=20)
kmeans.fit(train_embeddings)
from scipy.stats import mode
cluster_labels = {}
for c in range(num_classes):
    indices = np.where(kmeans.labels_ == c)[0]
    if len(indices) > 0:
        majority_label = mode(train_labels[indices], keepdims=False).mode
        cluster_labels[c] = majority_label
test_clusters = kmeans.predict(test_embeddings)
y_pred_kmeans = np.array([cluster_labels[c] for c in test_clusters])
print(f"KMeans-based classifier accuracy: {accuracy_score(test_labels, y_pred_kmeans)}")
print(classification_report(test_labels, y_pred_kmeans, target_names=label_encoder.classes_, zero_division=0))
print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(test_labels, y_pred_kmeans)}")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(test_labels, y_pred_kmeans)}")
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
print(f"Homogeneity: {homogeneity_score(test_labels, y_pred_kmeans)}")
print(f"Completeness: {completeness_score(test_labels, y_pred_kmeans)}")
print(f"V-measure: {v_measure_score(test_labels, y_pred_kmeans)}")
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances
n_classes = len(np.unique(train_labels))
spectral = SpectralClustering(
    n_clusters=n_classes,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    random_state=42
)
train_clusters = spectral.fit_predict(train_embeddings)
contingency = np.zeros((n_classes, n_classes), dtype=int)
for t, c in zip(train_labels, train_clusters):
    contingency[t, c] += 1
row_ind, col_ind = linear_sum_assignment(-contingency)
cluster_to_label = {c: l for l, c in zip(row_ind, col_ind)}
train_pred_labels = np.array([cluster_to_label[c] for c in train_clusters])
print("Train classification report:")
print(classification_report(train_labels, train_pred_labels, target_names=label_encoder.classes_))
train_cluster_centroids = []
for c in range(n_classes):
    cluster_embs = train_embeddings[train_clusters == c]
    train_cluster_centroids.append(cluster_embs.mean(axis=0))
train_cluster_centroids = np.stack(train_cluster_centroids)
dists = cosine_distances(test_embeddings, train_cluster_centroids)
test_clusters = np.argmin(dists, axis=1)
test_pred_labels = np.array([cluster_to_label[c] for c in test_clusters])
print("Test classification report:")
print(classification_report(test_labels, test_pred_labels, target_names=label_encoder.classes_))
ari = adjusted_rand_score(test_labels, test_pred_labels)
nmi = normalized_mutual_info_score(test_labels, test_pred_labels)
print(f"Spectral Clustering Results:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
embeddings, labels = extract_embeddings(model, train_loader_2d_aug, device)
num_classes = len(np.unique(labels))
optimal_clusters = {}
for c in range(num_classes):
    class_indices = np.where(labels == c)[0]
    class_embeddings = embeddings[class_indices]
    if len(class_embeddings) < 5:
        optimal_clusters[c] = 1
        continue
    best_k = 1
    best_score = -1
    for k in range(2, min(10, len(class_embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(class_embeddings)
        score = silhouette_score(class_embeddings, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    optimal_clusters[c] = best_k
    print(f"Letter {label_encoder.inverse_transform([c])[0]}: optimal k = {best_k}, silhouette = {best_score:.3f}")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
embeddings, labels = extract_embeddings(model, test_loader_2d, device)
tau_idx = np.where(label_encoder.classes_ == "Tau")[0][0]
tau_indices = np.where(labels == tau_idx)[0]
tau_embeddings = embeddings[tau_indices]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
tau_clusters = kmeans.fit_predict(tau_embeddings)
def show_cluster_images(cluster_id, n=10):
    idxs = tau_indices[tau_clusters == cluster_id]
    if len(idxs) > n:
        idxs = np.random.choice(idxs, n, replace=False)
    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(idxs):
        img, _ = test_loader_2d.dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        plt.subplot(1, len(idxs), i+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Tau - Cluster {cluster_id}", fontsize=14)
    plt.show()
for c in range(3):
    show_cluster_images(c, n=10)
mss_image_folder = 'marias-inference-dataset/'
mss_image_files = [os.path.join(mss_image_folder, f) for f in os.listdir(mss_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(mss_image_files)} images in maria's folder.")
marias_image_data = []
marias_filenames = []
for img_file_path in mss_image_files:
    try:
        processed_img = preprocess_image(img_file_path)
        marias_image_data.append(processed_img)
        marias_filenames.append(os.path.basename(img_file_path))
    except Exception as e:
        print(f"Error processing image {img_file_path}: {e}")
marias_image_data = np.array(marias_image_data)
print(f"Processed {len(marias_image_data)} images from maria's cliplets.")
mss_image_folder = 'marias-inference-dataset/'
mss_image_files = [os.path.join(mss_image_folder, f) for f in os.listdir(mss_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(mss_image_files)} images in maria's folder.")
marias_image_data_2d = []
marias_filenames = []
for img_file_path in mss_image_files:
    try:
        processed_img_2d = preprocess_image_2d(img_file_path)
        marias_image_data_2d.append(processed_img_2d)
        marias_filenames.append(os.path.basename(img_file_path))
    except Exception as e:
        print(f"Error processing image {img_file_path} for 2D CNN: {e}")
marias_image_data_2d = np.array(marias_image_data_2d)
print(f"Processed {len(marias_image_data_2d)} images from maria's cliplets.")
class ImageDatasetInference(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = self.X[idx]
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        return image
marias_dataset = ImageDatasetInference(marias_image_data_2d, transform=test_transform)
marias_loader = DataLoader(marias_dataset, batch_size=batch_size)
model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
model.eval()
model.to(device)
marias_predictions = []
with torch.no_grad():
    for inputs in marias_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        marias_predictions.extend(predicted.tolist())
marias_predicted_letters = label_encoder.inverse_transform(marias_predictions)
marias_results_df = pd.DataFrame({
    'filename': marias_filenames,
    'inferred_label': marias_predicted_letters
})
print("\nInference completed. Predicted labels for Maria's images:")
print(marias_results_df)
num_marias_to_display = len(marias_results_df)
images_per_row = 5
plt.figure(figsize=(15, int(num_marias_to_display / images_per_row) * 3))
for i in range(num_marias_to_display):
    filename = marias_results_df.iloc[i]['filename']
    img_path = os.path.join(mss_image_folder, filename)
    inferred_label = marias_results_df.iloc[i]['inferred_label']
    try:
        img = mpimg.imread(img_path)
        num_rows = (num_marias_to_display + images_per_row - 1) // images_per_row
        num_cols = images_per_row
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{inferred_label}', fontsize=16)
        plt.axis('off')
    except Exception as e:
        print(f"Error displaying Maria's image {img_path}: {e}")
plt.tight_layout()
plt.savefig('marias_inferred_images.pdf', format='pdf', dpi=300)
plt.show()
tta_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def extract_embeddings_tta(model, dataset, device, n_aug=5):
    """
    Compute embeddings for each image using TTA (average over n_aug augmentations)
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img = dataset[i]
            tta_embs = []
            for _ in range(n_aug):
                augmented_img = tta_transform(Image.fromarray((img.numpy().squeeze()*255).astype(np.uint8)))
                augmented_img = augmented_img.unsqueeze(0).to(device)
                x = model.pool1(model.relu(model.conv1(augmented_img)))
                x = model.pool2(model.relu(model.conv2(x)))
                x = model.pool3(model.relu(model.conv3(x)))
                x = x.view(x.size(0), -1)
                x = model.relu(model.fc1(x))
                tta_embs.append(x.cpu().numpy())
            tta_embs = np.vstack(tta_embs)
            embeddings.append(tta_embs.mean(axis=0))
    return np.vstack(embeddings)
marias_embeddings_tta = extract_embeddings_tta(model, marias_dataset, device, n_aug=5)
train_embeddings, train_labels = extract_embeddings(model, train_loader_2d_aug, device)
train_images = X_train_2d
train_pred_labels = label_encoder.inverse_transform(train_labels)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
def fetch_similar_images_with_labels(query_embeddings, gallery_embeddings, gallery_images, gallery_labels, top_k=3):
    query_embeddings = normalize(query_embeddings)
    gallery_embeddings = normalize(gallery_embeddings)
    dists = cosine_distances(query_embeddings, gallery_embeddings)
    nearest_idxs = np.argsort(dists, axis=1)[:, :top_k]
    similar_images = []
    similar_labels = []
    for idxs in nearest_idxs:
        similar_images.append([gallery_images[i] for i in idxs])
        similar_labels.append([gallery_labels[i] for i in idxs])
    return similar_images, similar_labels
marias_similar_images, marias_similar_labels = fetch_similar_images_with_labels(
    marias_embeddings_tta,
    train_embeddings,
    train_images,
    train_pred_labels,
    top_k=3
)
import matplotlib.pyplot as plt
for i, query_img in enumerate(marias_image_data_2d):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(query_img, cmap='gray')
    axes[0].set_title(f'Maria\n{marias_predicted_letters[i]}')
    axes[0].axis('off')
    for j, (sim_img, sim_label) in enumerate(zip(marias_similar_images[i], marias_similar_labels[i])):
        axes[j+1].imshow(sim_img, cmap='gray')
        axes[j+1].set_title(f'{sim_label}', fontsize=10)
        axes[j+1].axis('off')
    plt.tight_layout()
    plt.show()
model.eval()
unk_predictions_2d = []
unk_filenames_2d = []
unk_data_filenames = unk_data_split['filename'].tolist()
with torch.no_grad():
    for i in range(len(unk_loader_2d.dataset)):
        inputs = unk_loader_2d.dataset[i].unsqueeze(0).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        unk_predictions_2d.extend(predicted.cpu().numpy())
predicted_unk_letters_2d = label_encoder.inverse_transform(unk_predictions_2d)
unk_predictions_df_2d = pd.DataFrame({
    'filename': unk_data_filenames,
    'predicted_letter_2d_aug': predicted_unk_letters_2d,
    'year': unk_data_split['year'].tolist()
})
print("\n2D CNN with Augmentation Predictions for 'Other' images:")
print(unk_predictions_df_2d.head())
plt.figure(figsize=(10, 3))
unk_predictions_df_2d['predicted_letter_2d_aug'].value_counts().plot.bar()
plt.title('Distribution of Inferred Letters on "Other" Images')
plt.xlabel('Inferred Letter')
plt.ylabel('Count')
sns.despine(left=True, bottom=True)
plt.show()
plt.figure(figsize=(4, 6))
ax = sns.boxplot(y='predicted_letter_2d_aug', x='year', data=unk_predictions_df_2d, orient='horizontal', fill=False, color='salmon')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "grid.alpha": 0.5,
    "grid.color": "grey",
    "axes.facecolor": "whitesmoke",
    "figure.facecolor": "white"
})
ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xlabel(''); plt.ylabel('')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
sns.despine()
plt.tight_layout()
plt.savefig('years_per_inferred_letter.pdf', format='pdf', dpi=300)
plt.show()
num_unk_to_display = len(unk_predictions_df_2d)
plt.figure(figsize=(12, int(num_unk_to_display / 4) * 3))
for i in range(num_unk_to_display):
    original_index = data[data['filename'] == unk_predictions_df_2d.iloc[i]['filename']].index[0]
    img_path = image_files[original_index]
    inferred_label = unk_predictions_df_2d.iloc[i]['predicted_letter_2d_aug']
    try:
        img = mpimg.imread(img_path)
        plt.subplot(int(num_unk_to_display / 4) + 1, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{inferred_label}?', fontsize=20)
        plt.axis('off')
    except Exception as e:
        print(f"Error displaying unknown image {img_path}: {e}")
plt.tight_layout()
plt.savefig('other_images.pdf', format='pdf', dpi=300)
plt.show()
train_data_with_indices = data[data['letter'] != 'Other'].copy()
train_data_with_indices['original_index'] = train_data_with_indices.index
train_X_for_split = X[train_data_with_indices['original_index'].values]
train_y_for_split_encoded = label_encoder.transform(train_data_with_indices['letter'])
X_train_split_df, X_test_split_df, y_train_split_encoded, y_test_split_encoded = train_test_split(
    train_data_with_indices,
    train_y_for_split_encoded,
    test_size=0.2,
    random_state=42,
    stratify=train_y_for_split_encoded
)
X_train_split_df, X_val_split_df, y_train_split_encoded, y_val_split_encoded = train_test_split(
    X_train_split_df,
    y_train_split_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_train_split_encoded
)
if len(y_true_letters) != len(X_test_split_df):
     print("Warning: Length of y_true_letters does not match the length of the test split DataFrame.")
else:
    test_data_with_years = X_test_split_df.copy()
    test_data_with_years['true_label'] = y_true_letters
    test_data_with_years['pred_label'] = y_pred_letters
    correct_predictions_df_with_years = test_data_with_years[test_data_with_years['true_label'] == test_data_with_years['pred_label']]
    mistaken_predictions_df_with_years = test_data_with_years[test_data_with_years['true_label'] != test_data_with_years['pred_label']]
    plt.figure(figsize=(12, 6))
    sns.histplot(data=correct_predictions_df_with_years, x='year', color='skyblue', label='Correct', kde=True, bins=30)
    sns.histplot(data=mistaken_predictions_df_with_years, x='year', color='salmon', label='Mistaken', kde=True, bins=30)
    plt.title('Distribution of Years for Correct vs. Mistaken Predictions (using Original Year Data)')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8, 6))
    comparison_data_with_years = pd.concat([
        correct_predictions_df_with_years.assign(Prediction_Type='Correct'),
        mistaken_predictions_df_with_years.assign(Prediction_Type='Mistaken')
    ])
    sns.boxplot(x='Prediction_Type', y='year', data=comparison_data_with_years)
    plt.title('Year Distribution Comparison: Correct vs. Mistaken Predictions (using Original Year Data)')
    plt.xlabel('Prediction Type')
    plt.ylabel('Year')
    sns.despine()
    plt.show()
    print("\nSummary Statistics for Years (using Original Year Data):")
    print("Correct Predictions:")
    print(correct_predictions_df_with_years['year'].describe())
    print("\nMistaken Predictions:")
    print(mistaken_predictions_df_with_years['year'].describe())
    test_data_with_years = test_data_with_years.drop(columns=['original_index'])
letter_year_comparison_data = []
for letter in labels_order:
    letter_subset = test_data_with_years[test_data_with_years['true_label'] == letter]
    correct_for_letter = letter_subset[letter_subset['true_label'] == letter_subset['pred_label']]
    mistaken_for_letter = letter_subset[letter_subset['true_label'] != letter_subset['pred_label']]
    if not correct_for_letter.empty:
        letter_year_comparison_data.extend([
            {'letter': letter, 'year': row['year'], 'prediction_type': 'Correct'}
            for index, row in correct_for_letter.iterrows()
        ])
    if not mistaken_for_letter.empty:
        letter_year_comparison_data.extend([
            {'letter': letter, 'year': row['year'], 'prediction_type': 'Mistaken'}
            for index, row in mistaken_for_letter.iterrows()
        ])
letter_year_comparison_df = pd.DataFrame(letter_year_comparison_data)
plt.figure(figsize=(15, 8))
ax = sns.boxplot(x='letter', y='year', hue='prediction_type', data=letter_year_comparison_df, palette={'Correct': 'skyblue', 'Mistaken': 'salmon'})
plt.title('Year Distribution Comparison for Correct vs. Mistaken Predictions by Letter')
plt.xlabel('Letter', fontsize=20); plt.ylabel('Year', fontsize=20); plt.xticks(rotation=90); plt.legend(title='Prediction Type')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
sns.despine()
plt.savefig('error_analysis.pdf', format='pdf', dpi=300)
plt.show()
letter_counts = letter_year_comparison_df.groupby(['letter', 'prediction_type']).size().unstack(fill_value=0)
plt.figure(figsize=(15, 8))
ax = letter_counts.plot(kind='bar', stacked=True, figsize=(15, 8), color={'Correct': 'skyblue', 'Mistaken': 'salmon'})
plt.title('Number of Correct and Mistaken Predictions per Letter')
plt.xlabel('Letter')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Prediction Type', fontsize=14)
sns.despine()
for c in ax.containers:
    labels = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center')
plt.tight_layout()
plt.show()
correct_predictions_df_with_regions = test_data_with_years[test_data_with_years['true_label'] == test_data_with_years['pred_label']]
mistaken_predictions_df_with_regions = test_data_with_years[test_data_with_years['true_label'] != test_data_with_years['pred_label']]
letter_region_comparison_data = []
for letter in labels_order:
    letter_subset = test_data_with_years[test_data_with_years['true_label'] == letter]
    correct_for_letter = letter_subset[letter_subset['true_label'] == letter_subset['pred_label']]
    mistaken_for_letter = letter_subset[letter_subset['true_label'] != letter_subset['pred_label']]
    correct_region_counts = correct_for_letter['region'].value_counts().reset_index()
    correct_region_counts.columns = ['region', 'count']
    correct_region_counts['letter'] = letter
    correct_region_counts['prediction_type'] = 'Correct'
    mistaken_region_counts = mistaken_for_letter['region'].value_counts().reset_index()
    mistaken_region_counts.columns = ['region', 'count']
    mistaken_region_counts['letter'] = letter
    mistaken_region_counts['prediction_type'] = 'Mistaken'
    letter_region_comparison_data.extend(correct_region_counts.to_dict('records'))
    letter_region_comparison_data.extend(mistaken_region_counts.to_dict('records'))
letter_region_comparison_df = pd.DataFrame(letter_region_comparison_data)
mistaken_regions_df = letter_region_comparison_df[letter_region_comparison_df['prediction_type'] == 'Mistaken']
all_regions = data['region'].unique()
pivot_data = mistaken_regions_df.pivot_table(index='region', columns='letter', values='count', fill_value=0)
pivot_data = pivot_data.reindex(all_regions, columns=labels_order, fill_value=0)
pivot_data = pivot_data.div(pivot_data.sum(axis=1), axis=0)
annot = pivot_data.applymap(lambda x: "0" if x == 0 else f"{x:.2f}")
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_data, annot=annot, fmt="", cmap='YlGnBu', linewidths=.5)
plt.title('Region Distribution in Misclassified Test Images per Letter')
plt.xlabel('Letter')
plt.ylabel('Region Frequency')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
get_ipython().run_cell_magic('capture', '', '!gdown 1ve4Lz7d0wuxhGepUHP_VbPMw7J9vdEqA && mss_cliplets.zip\n!git clone https://github.com/ipavlopoulos/greek-letter-vision.git\n')
import os
import numpy as np
mss_image_folder = 'mss_cliplets/'
mss_image_files = [os.path.join(mss_image_folder, f) for f in os.listdir(mss_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(mss_image_files)} images in the mss_cliplets folder.")
mss_image_data = []
mss_filenames = []
for img_file_path in mss_image_files:
    try:
        processed_img = preprocess_image(img_file_path)
        mss_image_data.append(processed_img)
        mss_filenames.append(os.path.basename(img_file_path))
    except Exception as e:
        print(f"Error processing image {img_file_path}: {e}")
mss_image_data = np.array(mss_image_data)
print(f"Processed {len(mss_image_data)} images from mss_cliplets.")
if mss_image_data.size > 0:
    mss_image_data_pca = pca.transform(mss_image_data)
    print(f"Transformed mss_cliplets data with PCA: {mss_image_data_pca.shape}")
    X_mss_tensor = torch.tensor(mss_image_data_pca, dtype=torch.float32)
    mss_dataset = PCADataset(X_mss_tensor)
    mss_loader = DataLoader(mss_dataset, batch_size=batch_size)
    model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
    model.eval()
    mss_predictions = []
    with torch.no_grad():
        for inputs in mss_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            mss_predictions.extend(predicted.tolist())
    mss_predicted_letters = label_encoder.inverse_transform(mss_predictions)
    mss_results_df = pd.DataFrame({
        'filename': mss_filenames,
        'inferred_label': mss_predicted_letters
    })
else:
    print("No images were successfully processed from mss_cliplets. Cannot perform PCA or inference.")
mss_results_df['letter'] = mss_results_df.filename.apply(lambda x: x.split('_')[0])
mss_results_df['ID'] = mss_results_df.filename.apply(lambda x: int(x.split('_')[1]))
mss_results_df['uid'] = mss_results_df.filename.apply(lambda x: int(x.split('.')[0].split('_')[2]))
mss_results_df.sample(5)
mss_results_df.replace({'Sigmai': 'Sigma', 'SIgma':'Sigma', 'Lmbda': 'Lambda', 'ChI':'Chi'}, inplace=True)
mss_results_df.letter.value_counts()
import pandas as pd
sheet_url = "https://docs.google.com/spreadsheets/d/17ZHNCKgEeUZ-78nCwaJo9UGtKoCWqlQZYO4uja-jxuk/edit?gid=0
csv_url = sheet_url.replace('/edit?gid=', '/export?format=csv&gid=')
try:
    df_sheet = pd.read_csv(csv_url)
    print("Successfully parsed the Google Sheet:")
except Exception as e:
    print(f"Error parsing the Google Sheet: {e}")
    df_sheet = None
df_sheet.sample()
merged_df = pd.merge(df_sheet, mss_results_df, left_on='ΙD', right_on='ID', how='inner')
merged_df['year'] = merged_df['Year post quem']
merged_df.sample()
ce_dataset = merged_df[['ID', 'uid', 'filename', 'year', 'inferred_label', 'letter']]
ce_dataset.sample()
ce_dataset.groupby('letter').year.agg(['min', 'max', 'mean']).sample(3)
labels_order = sorted(list(set(ce_dataset.letter.values)))
cm = confusion_matrix(ce_dataset.letter.values, ce_dataset.inferred_label.values, labels=labels_order)
plt.figure(figsize=(14, 12))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('True', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
plt.tight_layout()
plt.savefig('ood_confusion_matrix.pdf', format='pdf', dpi=300)
from sklearn.metrics import classification_report
print("\nClassification Report on Test Dataset:")
print(classification_report(mss_results_df.letter.values, mss_results_df.inferred_label.values, labels=labels_order, zero_division=0))
ce_mistakes = ce_dataset[ce_dataset['letter'] != ce_dataset['inferred_label']].copy()
ce_mistakes.sample()
ood_df = ce_mistakes.groupby('letter').year.apply(list)
plt.figure(figsize=(15, 8))
ax = plt.boxplot(ood_df.values, vert=True, patch_artist=True, labels=ood_df.index, widths=0.6)
for i, letter in enumerate(ood_df.index):
    num_mistakes = len(ood_df[letter])
    median_year = np.median(ood_df[letter]) if num_mistakes > 0 else 0
    y_pos = median_year+20
    x_pos = i + 1
    plt.text(x_pos, y_pos, str(num_mistakes),
             horizontalalignment='center', verticalalignment='bottom',
             fontsize=12, color='red', weight='bold')
plt.title('Boxplot of Years for Misclassified Out-of-Distribution Images per Letter (with Count of Mistakes)')
plt.xlabel('Letter', fontsize=20)
plt.ylabel('Year', fontsize=20)
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig('ood_error_years_boxplot.pdf', format='pdf', dpi=300)
plt.show()
ce_dataset.to_csv('asiminas_dataset.csv', index=False)
import os
import numpy as np
mss_image_folder = 'marias-inference-dataset/'
mss_image_files = [os.path.join(mss_image_folder, f) for f in os.listdir(mss_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(mss_image_files)} images in maria's folder.")
marias_image_data = []
marias_filenames = []
for img_file_path in mss_image_files:
    try:
        processed_img = preprocess_image(img_file_path)
        marias_image_data.append(processed_img)
        marias_filenames.append(os.path.basename(img_file_path))
    except Exception as e:
        print(f"Error processing image {img_file_path}: {e}")
marias_image_data = np.array(marias_image_data)
print(f"Processed {len(marias_image_data)} images from maria's cliplets.")
if marias_image_data.size > 0:
    marias_image_data_pca = pca.transform(marias_image_data)
    print(f"Transformed maria's cliplets with PCA: {marias_image_data_pca.shape}")
    X_mss_tensor = torch.tensor(marias_image_data_pca, dtype=torch.float32)
    marias_dataset = PCADataset(X_mss_tensor)
    marias_loader = DataLoader(marias_dataset, batch_size=batch_size)
    model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
    model.eval()
    marias_predictions = []
    with torch.no_grad():
        for inputs in marias_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            marias_predictions.extend(predicted.tolist())
    marias_predicted_letters = label_encoder.inverse_transform(marias_predictions)
    marias_results_df = pd.DataFrame({
        'filename': marias_filenames,
        'inferred_label': marias_predicted_letters
    })
else:
    print("No images were successfully processed from mss_cliplets. Cannot perform PCA or inference.")
num_marias_to_display = len(marias_results_df)
images_per_row = 5
plt.figure(figsize=(15, int(num_marias_to_display / images_per_row) * 3))
for i in range(num_marias_to_display):
    filename = marias_results_df.iloc[i]['filename']
    img_path = os.path.join(mss_image_folder, filename)
    inferred_label = marias_results_df.iloc[i]['inferred_label']
    try:
        img = mpimg.imread(img_path)
        num_rows = (num_marias_to_display + images_per_row - 1) // images_per_row
        num_cols = images_per_row
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{inferred_label}', fontsize=16)
        plt.axis('off')
    except Exception as e:
        print(f"Error displaying Maria's image {img_path}: {e}")
plt.tight_layout()
plt.savefig('marias_inferred_images.pdf', format='pdf', dpi=300)
plt.show()
merged_data = data.copy()
merge_map = {
    'Alpha': 'Alpha-Lambda-Delta',
    'Beta': 'Beta',
    'Eta': 'Eta',
    'Mu': 'Mu',
    'Nu': 'Nu',
    'Omega': 'Omega',
    'Kappa': 'Kappa',
    'Lambda': 'Alpha-Lambda-Delta',
    'Delta': 'Alpha-Lambda-Delta',
    'Epsilon': 'Epsilon-Theta-Sigma-Omicron',
    'Theta': 'Epsilon-Theta-Sigma-Omicron',
    'Sigma': 'Epsilon-Theta-Sigma-Omicron',
    'Omicron': 'Epsilon-Theta-Sigma-Omicron',
    'Gamma': 'Gamma-Iota-Rho',
    'Iota': 'Gamma-Iota-Rho',
    'Rho': 'Gamma-Iota-Rho',
    'Upsilon': 'Upsilon',
    'Psi': 'Psi-Tau',
    'Phi': 'Phi',
    'Tau': 'Psi-Tau',
    'Chi': 'Chi-Xi',
    'Xi': 'Chi-Xi'
}
merged_data['merged_letter'] = merged_data['letter'].apply(lambda x: merge_map.get(x, x))
delete_classes = ['Psi', 'Zeta']
merged_data_filtered = merged_data[~merged_data['letter'].isin(delete_classes)].copy()
data_for_reclassification = data.copy()
data_for_reclassification['merged_letter'] = data_for_reclassification['letter'].apply(lambda x: merge_map.get(x, x))
data_for_reclassification_filtered = data_for_reclassification[~data_for_reclassification['letter'].isin(delete_classes)].copy()
print(f"Original number of samples: {data.shape[0]}")
print(f"Number of samples after merging and filtering: {data_for_reclassification_filtered.shape[0]}")
print("\nDistribution of letters after merging and filtering:")
print(data_for_reclassification_filtered['merged_letter'].value_counts())
remaining_filenames = data_for_reclassification_filtered['filename'].tolist()
original_indices_remaining = [image_files.index(os.path.join(image_folder, fn)) for fn in remaining_filenames]
X_reclassified = image_data_pca[original_indices_remaining]
train_data_reclassified = data_for_reclassification_filtered[data_for_reclassification_filtered['merged_letter'] != 'Unknown'].copy()
train_X_reclassified = X_reclassified[data_for_reclassification_filtered['merged_letter'] != 'Unknown']
train_y_reclassified = train_data_reclassified['merged_letter']
unk_data_reclassified = data_for_reclassification_filtered[data_for_reclassification_filtered['merged_letter'] == 'Unknown'].copy()
unk_X_reclassified = X_reclassified[data_for_reclassification_filtered['merged_letter'] == 'Unknown']
unk_y_reclassified = unk_data_reclassified['merged_letter']
label_encoder_reclassified = LabelEncoder()
train_y_encoded_reclassified = label_encoder_reclassified.fit_transform(train_y_reclassified)
X_train_reclassified, X_val_reclassified, y_train_reclassified, y_val_reclassified = train_test_split(
    train_X_reclassified, train_y_encoded_reclassified, test_size=0.2, random_state=42, stratify=train_y_encoded_reclassified)
X_train_reclassified, X_test_reclassified, y_train_reclassified, y_test_reclassified = train_test_split(
    X_train_reclassified, y_train_reclassified, test_size=0.2, random_state=42, stratify=y_train_reclassified)
X_train_tensor_reclassified = torch.tensor(X_train_reclassified, dtype=torch.float32)
y_train_tensor_reclassified = torch.tensor(y_train_reclassified, dtype=torch.long)
X_val_tensor_reclassified = torch.tensor(X_val_reclassified, dtype=torch.float32)
y_val_tensor_reclassified = torch.tensor(y_val_reclassified, dtype=torch.long)
X_test_tensor_reclassified = torch.tensor(X_test_reclassified, dtype=torch.float32)
y_test_tensor_reclassified = torch.tensor(y_test_reclassified, dtype=torch.long)
X_unk_tensor_reclassified = torch.tensor(unk_X_reclassified, dtype=torch.float32) if len(unk_X_reclassified) > 0 else None
train_dataset_reclassified = PCADataset(X_train_tensor_reclassified, y_train_tensor_reclassified)
val_dataset_reclassified = PCADataset(X_val_tensor_reclassified, y_val_tensor_reclassified)
test_dataset_reclassified = PCADataset(X_test_tensor_reclassified, y_test_tensor_reclassified)
unk_dataset_reclassified = PCADataset(X_unk_tensor_reclassified) if X_unk_tensor_reclassified is not None else None
batch_size = 16
train_loader_reclassified = DataLoader(train_dataset_reclassified, batch_size=batch_size, shuffle=True)
val_loader_reclassified = DataLoader(val_dataset_reclassified, batch_size=batch_size)
test_loader_reclassified = DataLoader(test_dataset_reclassified, batch_size=batch_size)
unk_loader_reclassified = DataLoader(unk_dataset_reclassified, batch_size=batch_size) if unk_dataset_reclassified is not None else None
input_dimension_reclassified = X_train_tensor_reclassified.shape[1]
num_classes_reclassified = len(label_encoder_reclassified.classes_)
model_reclassified = CNN1D(input_dim=input_dimension_reclassified, num_classes=num_classes_reclassified)
criterion_reclassified = nn.CrossEntropyLoss()
optimizer_reclassified = optim.Adam(model_reclassified.parameters(), lr=0.001)
patience = 10
best_val_loss_reclassified = float('inf')
epochs_no_improve_reclassified = 0
num_epochs = 100
train_losses_reclassified = []
val_losses_reclassified = []
val_accuracies_reclassified = []
print("\nStarting training with reclassified data...")
for epoch in range(num_epochs):
    model_reclassified.train()
    running_loss_reclassified = 0.0
    for inputs, labels in train_loader_reclassified:
        optimizer_reclassified.zero_grad()
        outputs = model_reclassified(inputs)
        loss = criterion_reclassified(outputs, labels)
        loss.backward()
        optimizer_reclassified.step()
        running_loss_reclassified += loss.item() * inputs.size(0)
    epoch_train_loss_reclassified = running_loss_reclassified / len(train_loader_reclassified.dataset)
    train_losses_reclassified.append(epoch_train_loss_reclassified)
    model_reclassified.eval()
    running_val_loss_reclassified = 0.0
    correct_reclassified = 0
    total_reclassified = 0
    with torch.no_grad():
        for inputs, labels in val_loader_reclassified:
            outputs = model_reclassified(inputs)
            loss = criterion_reclassified(outputs, labels)
            running_val_loss_reclassified += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_reclassified += labels.size(0)
            correct_reclassified += (predicted == labels).sum().item()
    epoch_val_loss_reclassified = running_val_loss_reclassified / len(val_loader_reclassified.dataset)
    epoch_val_accuracy_reclassified = correct_reclassified / total_reclassified
    val_losses_reclassified.append(epoch_val_loss_reclassified)
    val_accuracies_reclassified.append(epoch_val_accuracy_reclassified)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss_reclassified:.4f}, Val Loss: {epoch_val_loss_reclassified:.4f}, Val Accuracy: {epoch_val_accuracy_reclassified:.4f}')
    if epoch_val_loss_reclassified < best_val_loss_reclassified:
        best_val_loss_reclassified = epoch_val_loss_reclassified
        epochs_no_improve_reclassified = 0
        torch.save(model_reclassified.state_dict(), 'best_cnn_letter_model_reclassified.pth')
    else:
        epochs_no_improve_reclassified += 1
        if epochs_no_improve_reclassified >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
model_reclassified.load_state_dict(torch.load('best_cnn_letter_model_reclassified.pth'))
plt.figure(figsize=(10, 5))
plt.plot(train_losses_reclassified, label='Train Loss')
plt.plot(val_losses_reclassified, label='Validation Loss')
plt.title('Training and Validation Loss (Reclassified Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies_reclassified, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy (Reclassified Data)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
model_reclassified.eval()
y_true_reclassified = []
y_pred_reclassified = []
with torch.no_grad():
    for inputs, labels in test_loader_reclassified:
        outputs = model_reclassified(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true_reclassified.extend(labels.tolist())
        y_pred_reclassified.extend(predicted.tolist())
y_true_letters_reclassified = label_encoder_reclassified.inverse_transform(y_true_reclassified)
y_pred_letters_reclassified = label_encoder_reclassified.inverse_transform(y_pred_reclassified)
labels_order_reclassified = sorted(list(set(y_true_letters_reclassified)))
cm_reclassified = confusion_matrix(y_true_letters_reclassified, y_pred_letters_reclassified, labels=labels_order_reclassified)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_reclassified, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order_reclassified, yticklabels=labels_order_reclassified)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Dataset (Reclassified Data)')
plt.show()
print("\nClassification Report on Test Dataset (Reclassified Data):")
print(classification_report(y_true_letters_reclassified, y_pred_letters_reclassified, labels=labels_order_reclassified, zero_division=0))
if unk_loader_reclassified is not None and len(unk_data_reclassified) > 0:
    print("\nInferring labels on Unknown data (Reclassified)...")
    model_reclassified.eval()
    unk_predictions_reclassified = []
    with torch.no_grad():
        for inputs in unk_loader_reclassified:
            outputs = model_reclassified(inputs)
            _, predicted = torch.max(outputs.data, 1)
            unk_predictions_reclassified.extend(predicted.tolist())
    unk_predicted_letters_reclassified = label_encoder_reclassified.inverse_transform(unk_predictions_reclassified)
    unk_data_reclassified['inferred_label_reclassified'] = unk_predicted_letters_reclassified
    print("\nUnknown Data with Inferred Labels (Reclassified):")
    print(unk_data_reclassified[['filename', 'inferred_label_reclassified', 'year']].sample(min(5, len(unk_data_reclassified))))
    print("\nDistribution of Inferred Labels on Unknown Data (Reclassified):")
    print(unk_data_reclassified['inferred_label_reclassified'].value_counts())
    plt.figure(figsize=(10, 6))
    unk_data_reclassified['inferred_label_reclassified'].value_counts().plot.bar()
    plt.title('Distribution of Inferred Labels on Unknown Data (Reclassified)')
    plt.xlabel('Inferred Merged Letter')
    plt.ylabel('Count')
    sns.despine(left=True, bottom=True)
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='inferred_label_reclassified', y='year', data=unk_data_reclassified)
    plt.title('Boxplot of Years per Inferred Merged Letter (Unknown Data, Reclassified)')
    plt.xlabel('Inferred Merged Letter')
    plt.ylabel('Year')
    plt.xticks(rotation=90)
    sns.despine()
    plt.show()
    num_unk_to_display_reclassified = len(unk_data_reclassified)
    if num_unk_to_display_reclassified > 0:
        plt.figure(figsize=(15, int(num_unk_to_display_reclassified / 5 + 1) * 3))
        for i in range(num_unk_to_display_reclassified):
            original_index = data[data['filename'] == unk_data_reclassified.iloc[i]['filename']].index[0]
            img_path = image_files[original_index]
            inferred_label_reclassified = unk_data_reclassified.iloc[i]['inferred_label_reclassified']
            try:
                img = mpimg.imread(img_path)
                plt.subplot(int(num_unk_to_display_reclassified / 5) + 1, 5, i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f'Pred: {inferred_label_reclassified}', fontsize=8)
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying unknown image {img_path}: {e}")
        plt.tight_layout()
        plt.show()
else:
    print("\nNo 'Unknown' data samples remaining after filtering.")
best_k_results = {}
unique_letters = data['letter'].unique()
for letter in unique_letters:
    print(f"\nProcessing letter: {letter}")
    letter_indices = data[data['letter'] == letter].index
    if len(letter_indices) < 2:
        print(f"  Skipping letter '{letter}' - not enough samples for clustering (need at least 2).")
        continue
    letter_image_data_pca = image_data_pca[letter_indices]
    best_silhouette = -1
    best_k_for_letter = None
    for k in range(2, 6):
        if k > len(letter_image_data_pca):
            print(f"  Skipping k={k} for letter '{letter}' - more clusters than samples.")
            continue
        print(f"  Testing k = {k} for letter '{letter}'...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(letter_image_data_pca)
        try:
            silhouette_avg = silhouette_score(letter_image_data_pca, clusters)
            print(f"    Silhouette Score for k={k}: {silhouette_avg:.4f}")
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k_for_letter = k
        except Exception as e:
            print(f"    Could not calculate silhouette score for k={k}: {e}")
            continue
    if best_k_for_letter is not None:
        best_k_results[letter] = {'best_k': best_k_for_letter, 'silhouette_score': best_silhouette}
        print(f"  Best k for letter '{letter}': {best_k_for_letter} (Silhouette: {best_silhouette:.4f})")
    else:
         print(f"  No valid k found for letter '{letter}'.")
print("\n--- Best K and Silhouette Scores per Letter ---")
for letter, result in best_k_results.items():
    print(f"Letter '{letter}': Best K = {result['best_k']}, Silhouette Score = {result['silhouette_score']:.4f}")
from scipy.linalg import orthogonal_procrustes
import cv2
def align_images_procrustes(image_path1, image_path2, size=(64, 64)):
    img1_data = preprocess_image(image_path1, size=size)
    img2_data = preprocess_image(image_path2, size=size)
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        img1_resized = cv2.resize(img1, size)
        img2_resized = cv2.resize(img2, size)
        _, thresh1 = cv2.threshold(img1_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(img2_resized, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour1 = max(contours1, key=cv2.contourArea) if contours1 else None
        contour2 = max(contours2, key=cv2.contourArea) if contours2 else None
        if contour1 is None or contour2 is None:
            print("Could not find contours in one or both images.")
            return None, None, None
        n_points = 100
        points1 = []
        points2 = []
        for i in range(n_points):
            t = i / (n_points - 1) * cv2.arcLength(contour1, True)
            pt = cv2.getAffineTransform(np.float32([[0,0],[1,0],[0,1]]), np.float32([[0,0],[1,0],[0,1]]))
            _, pt = cv2.getRotationMatrix2D((0,0),0,1)[0:2,:].dot(np.append(cv2.approxPolyDP(contour1, 1, True)[i % len(cv2.approxPolyDP(contour1, 1, True))][0], 1))
        def resample_contour(contour, num_points):
            arc_length = cv2.arcLength(contour, True)
            points = []
            for i in range(num_points):
                t = i / (num_points - 1) * arc_length
                index = int((i / (num_points - 1)) * len(contour)) % len(contour)
                points.append(contour[index][0])
            return np.array(points, dtype=np.float32)
        try:
             points1 = resample_contour(contour1, n_points)
             points2 = resample_contour(contour2, n_points)
        except Exception as e:
             print(f"Error resampling contours: {e}")
             min_len = min(len(contour1), len(contour2))
             points1 = np.array([pt[0] for pt in contour1[:min_len]], dtype=np.float64)
             points2 = np.array([pt[0] for pt in contour2[:min_len]], dtype=np.float64)
             if min_len == 0:
                 print("Contours are empty after resampling attempt.")
                 return None, None, None
        try:
            R, s = orthogonal_procrustes(points1, points2)
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            translation = centroid1 - s * centroid2.dot(R.T)
            aligned_points2 = s * points2.dot(R.T) + translation
        except ValueError as e:
             print(f"Orthogonal Procrustes failed: {e}")
             print(f"Shape of points1: {points1.shape}")
             print(f"Shape of points2: {points2.shape}")
             return None, None, None
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(points1[:, 0], points1[:, 1], label='Rho (Original)', alpha=0.6)
        plt.scatter(points2[:, 0], points2[:, 1], label='Iota (Original)', alpha=0.6)
        plt.title('Original Contour Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.subplot(1, 2, 2)
        plt.scatter(points1[:, 0], points1[:, 1], label='Rho (Target)', alpha=0.6)
        plt.scatter(aligned_points2[:, 0], aligned_points2[:, 1], label='Iota (Aligned)', alpha=0.6)
        plt.title('Aligned Contour Points (Iota to Rho)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        return R, translation, s
    except Exception as e:
        print(f"Error during image alignment: {e}")
        return None, None, None
rho_image_path = None
iota_image_path = None
rho_entries = merged_data[merged_data['letter'] == 'Rho']
iota_entries = merged_data[merged_data['letter'] == 'Iota']
if not rho_entries.empty:
    rho_filename = rho_entries.iloc[0]['filename']
    rho_image_path = os.path.join(image_folder, rho_filename)
    print(f"Found Rho image: {rho_image_path}")
else:
    print("No 'Rho' images found in the data.")
if not iota_entries.empty:
    iota_filename = iota_entries.iloc[0]['filename']
    iota_image_path = os.path.join(image_folder, iota_filename)
    print(f"Found Iota image: {iota_image_path}")
else:
     print("No 'Iota' images found in the data.")
if rho_image_path and iota_image_path:
    print("\nAttempting to align Iota to Rho using Orthogonal Procrustes...")
    rotation_matrix, translation_vector, scale_factor = align_images_procrustes(rho_image_path, iota_image_path)
    if rotation_matrix is not None:
        print("\nAlignment successful!")
        print("Rotation Matrix (R):")
        print(rotation_matrix)
        print("\nTranslation Vector (t):")
        print(translation_vector)
        print("\nScale Factor (s):")
        print(scale_factor)
    else:
        print("\nImage alignment failed.")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
class PCADataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
train_dataset = PCADataset(X_train_tensor, y_train_tensor)
val_dataset = PCADataset(X_val_tensor, y_val_tensor)
test_dataset = PCADataset(X_test_tensor, y_test_tensor)
unk_dataset = PCADataset(X_unk_tensor)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
unk_loader = DataLoader(unk_dataset, batch_size=batch_size)
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            dummy_output = self.pool2(self.relu(self.conv2(self.pool(self.relu(self.conv1(dummy_input))))))
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
input_dimension = X_train_tensor.shape[1]
num_classes = len(label_encoder.classes_)
model = CNN1D(input_dim=input_dimension, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
num_epochs = 100
train_losses = []
val_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_accuracy = correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_cnn_letter_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
model.load_state_dict(torch.load('best_cnn_letter_model.pth'))
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
