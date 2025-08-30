import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, silhouette_score,
    silhouette_samples, homogeneity_score, completeness_score, v_measure_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from scipy.linalg import orthogonal_procrustes
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2


class ImageProcessor:
    """Handles image preprocessing operations"""
    
    @staticmethod
    def preprocess_image(image_path, size=(64, 64)):
        """Preprocess image for 1D analysis"""
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        img_resized = cv2.resize(img_bin, size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized.flatten()
    
    @staticmethod
    def preprocess_image_2d(image_path, size=(64, 64)):
        """Preprocess image for 2D analysis"""
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        _, img_bin = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        img_resized = cv2.resize(img_bin, size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    
    @staticmethod
    def process_image_folder(image_files, preprocess_func):
        """Process all images in a folder"""
        image_data = []
        for img_file in image_files:
            try:
                processed_img = preprocess_func(img_file)
                image_data.append(processed_img)
            except Exception as e:
                print(f"Error processing image {img_file}: {e}")
        return np.array(image_data)


class DataManager:
    """Handles data loading and preparation"""
    
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = None
        self.data = None
        self.metadata = None
        
    def load_image_files(self):
        """Load image file paths"""
        self.image_files = [
            os.path.join(self.image_folder, f) 
            for f in os.listdir(self.image_folder) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        return self.image_files
    
    def create_dataframe(self):
        """Create dataframe from filenames"""
        filenames = os.listdir(self.image_folder)
        self.data = pd.DataFrame({'filename': filenames})
        self.data['letter'] = self.data.filename.apply(lambda x: x.split('_')[0])
        self.data['TM'] = self.data.filename.apply(lambda x: int(x.split('_')[1]))
        self.data['number'] = self.data.filename.apply(lambda x: x.split('_')[2].split('.')[0])
        return self.data
    
    def load_metadata(self, metadata_path):
        """Load metadata CSV"""
        self.metadata = pd.read_csv(metadata_path)
        return self.metadata
    
    def merge_with_metadata(self):
        """Merge data with metadata"""
        if self.metadata is not None and self.data is not None:
            self.data['year'] = self.data.TM.apply(
                lambda x: self.metadata.loc[self.metadata['TM'] == x]['Year ante quem'].values[0]
            )
            self.data['region'] = self.data.TM.apply(
                lambda x: self.metadata.loc[self.metadata['TM'] == x]['Production Nome (supposed)'].values[0]
            )
        return self.data


class PCAAnalyzer:
    """Handles PCA analysis"""
    
    def __init__(self, n_components=400):
        self.n_components = n_components
        self.pca = None
        
    def fit_transform(self, image_data):
        """Fit PCA and transform data"""
        self.pca = PCA(n_components=self.n_components)
        image_data_pca = self.pca.fit_transform(image_data)
        return image_data_pca
    
    def transform(self, image_data):
        """Transform data using fitted PCA"""
        if self.pca is None:
            raise ValueError("PCA must be fitted first")
        return self.pca.transform(image_data)
    
    def plot_explained_variance(self):
        """Plot cumulative explained variance"""
        if self.pca is None:
            raise ValueError("PCA must be fitted first")
            
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), 
                cumulative_explained_variance, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by Number of Principal Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.grid(True)
        plt.show()
        
        print(f"Cumulative explained variance with {self.pca.n_components} components: "
              f"{cumulative_explained_variance[-1]:.4f}")


class ClusterAnalyzer:
    """Handles clustering analysis"""
    
    def __init__(self):
        self.kmeans = None
        self.spectral = None
        
    def kmeans_clustering(self, data, n_clusters=24, random_state=42):
        """Perform K-means clustering"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = self.kmeans.fit_predict(data)
        return clusters
    
    def spectral_clustering(self, data, n_clusters=24, random_state=42):
        """Perform spectral clustering"""
        self.spectral = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels='discretize',
            affinity='nearest_neighbors',
            n_neighbors=15,
            random_state=random_state,
            n_init=10
        )
        clusters = self.spectral.fit_predict(data)
        return clusters
    
    def evaluate_clustering(self, ground_truth_labels, clusters, features):
        """Evaluate clustering performance"""
        nmi = normalized_mutual_info_score(ground_truth_labels, clusters)
        ari = adjusted_rand_score(ground_truth_labels, clusters)
        
        results = {
            'nmi': nmi,
            'ari': ari
        }
        
        if len(np.unique(clusters)) > 1:
            average_silhouette = silhouette_score(features, clusters)
            results['silhouette'] = average_silhouette
        
        homogeneity = homogeneity_score(ground_truth_labels, clusters)
        completeness = completeness_score(ground_truth_labels, clusters)
        v_measure = v_measure_score(ground_truth_labels, clusters)
        
        results.update({
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure
        })
        
        return results
    
    def hungarian_accuracy(self, ground_truth, predictions):
        """Compute Hungarian accuracy for clustering"""
        ground_truth = np.asarray(ground_truth)
        predictions = np.asarray(predictions)
        cm = confusion_matrix(ground_truth, predictions)
        cost_matrix = -cm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_assignment_sum = -cost_matrix[row_ind, col_ind].sum()
        total_samples = cm.sum()
        accuracy = optimal_assignment_sum / total_samples
        return accuracy
    
    def calculate_cluster_purity(self, ground_truth_labels, clusters, n_clusters):
        """Calculate purity for each cluster"""
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
                
        return cluster_purity, cluster_most_represented_letter


class Visualizer:
    """Handles visualization tasks"""
    
    @staticmethod
    def display_sample_images(image_files, num_images=5):
        """Display sample images"""
        num_images_to_display = min(num_images, len(image_files))
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
    
    @staticmethod
    def plot_cluster_distribution(clusters):
        """Plot cluster distribution"""
        pd.Series(clusters).value_counts().plot.bar(figsize=(10, 2))
        sns.despine(left=True, bottom=True)
        plt.show()
    
    @staticmethod
    def display_cluster_samples(image_files, clusters, n_clusters, max_images=10):
        """Display sample images from each cluster"""
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            num_images_in_cluster = len(cluster_indices)
            print(f"Cluster {cluster_id} ({num_images_in_cluster} images):")
            
            images_to_display = cluster_indices[:min(max_images, num_images_in_cluster)]
            if images_to_display.size > 0:
                plt.figure(figsize=(5, 10))
                for i, img_index in enumerate(images_to_display):
                    img_path = image_files[img_index]
                    img = Image.open(img_path).convert('RGB').resize((12, 12))
                    plt.subplot(1, len(images_to_display), i + 1)
                    plt.imshow(img)
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
            else:
                print("No images in this cluster to display.")
    
    @staticmethod
    def create_clustered_image_grid(image_files, clusters, n_clusters, images_per_cluster_row=5):
        """Create a grid of clustered images"""
        cluster_image_indices = [np.where(clusters == i)[0] for i in range(n_clusters)]
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
        max_actual_row_width = 0
        for cluster_id in range(n_clusters):
            images_in_row = min(len(cluster_image_indices[cluster_id]), images_per_cluster_row)
            actual_row_width = images_in_row * display_img_size[0] + (images_in_row - 1) * margin
            max_actual_row_width = max(max_actual_row_width, actual_row_width)
        
        actual_width = max_actual_row_width
        final_image_cropped = final_image.crop((0, 0, actual_width, actual_height))
        return final_image_cropped
    
    @staticmethod
    def plot_clusters_2d(image_data_pca, clusters, cluster_centers=None, image_files=None, n_clusters=None):
        """Plot 2D visualization of clusters"""
        plt.figure(figsize=(15, 12))
        scatter = plt.scatter(image_data_pca[:, 0], image_data_pca[:, 1], 
                            c=clusters, cmap='viridis', s=10, alpha=0.5)
        plt.grid(True)
        
        if cluster_centers is not None and image_files is not None and n_clusters is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            
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
                        
                        imagebox = OffsetImage(img_thumb_np, zoom=1.0)
                        ab = AnnotationBbox(imagebox, (center_x, center_y), frameon=False, pad=0.1)
                        plt.gca().add_artist(ab)
                    except Exception as e:
                        print(f"Error processing image for annotation {closest_image_path}: {e}")
        
        sns.despine(left=True, bottom=True)
        plt.savefig('image_clusters.pdf', dpi=300, format='PDF')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('True', fontsize=18)
        plt.title(title)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300)
        plt.show()
    
    @staticmethod
    def plot_purity_bars(purity_df, title="Cluster Purity", save_path=None):
        """Plot cluster purity bars"""
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Purity', y='Cluster', data=purity_df, orient='h', color='skyblue')
        plt.yticks(ticks=purity_df.index, fontsize=18, 
                  labels=[f'{c} ({l})' for c, l in zip(purity_df['Cluster'], purity_df['Letter'])])
        plt.xlabel('Purity', fontsize=14)
        plt.ylabel('Cluster (Majority Letter)', fontsize=18)
        plt.title(title)
        plt.grid(axis='x')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, format='PDF')
        plt.show()


class PCADataset(Dataset):
    """Dataset class for PCA-transformed data"""
    
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
        

class FlattenedImageDataset(Dataset):
    """Dataset wrapper that flattens images for 1D models (MLP, CNN1D)"""
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
        # Flatten to 1D vector
        image = image.view(-1)
        if self.y is not None:
            return image, self.y[idx]
        else:
            return image


class ImageDatasetAugmented(Dataset):
    """Dataset class for augmented images"""
    
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


class ImageDatasetInference(Dataset):
    """Dataset class for inference"""
    
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


class CNN1D(nn.Module):
    """1D CNN for PCA-transformed or flattened image data"""
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
        if x.dim() > 2:  # e.g. (batch, 1, 64, 64)
            x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
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

    def get_embeddings(self, x):
        """Return embeddings before final classification layer"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
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
        return x

class CNN1DDeep(nn.Module):
    """Deep 1D CNN for PCA-transformed or flattened image data"""
    
    def __init__(self, input_dim, num_classes):
        super(CNN1DDeep, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            dummy_output = self.pool3(
                self.relu(
                    self.conv3(
                        self.pool2(
                            self.relu(
                                self.conv2(
                                    self.pool1(
                                        self.relu(self.conv1(dummy_input))
                                    )
                                )
                            )
                        )
                    )
                )
            )
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # flatten if input is an image (batch, 1, 64, 64)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
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

    def get_embeddings(self, x):
        """Return embeddings before final classification layer"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x

class CNN2D(nn.Module):
    """2D CNN for image data"""
    
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


class ModelTrainer:
    """Handles model training"""
    
    def __init__(self, model, criterion, optimizer, device, patience=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(train_loader.dataset)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        return epoch_val_loss, val_accuracy
    
    def train(self, train_loader, val_loader, num_epochs=100, save_path='best_model.pth'):
        """Full training loop"""
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        print(f"Starting training on device: {self.device}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs.')
                    break
        
        self.model.load_state_dict(torch.load(save_path))
        print("Training finished.")
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class ModelEvaluator:
    """Handles model evaluation"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, test_loader, label_encoder):
        """Evaluate model on test set"""
        self.model.eval()
        predicted_labels = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        predicted_letters = label_encoder.inverse_transform(predicted_labels)
        true_letters = label_encoder.inverse_transform(true_labels)
        
        return true_letters, predicted_letters
    
    def predict(self, data_loader, label_encoder):
        """Make predictions on unlabeled data"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs in data_loader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        predicted_letters = label_encoder.inverse_transform(predictions)
        return predicted_letters
    
    def extract_embeddings(self, dataloader):
        """Extract embeddings from model"""
        self.model.eval()
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
                
                images = images.to(self.device)
                if hasattr(self.model, 'get_embeddings'):
                    emb = self.model.get_embeddings(images)
                else:
                    # For models without get_embeddings method
                    emb = self.model(images)
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        if labels:
            labels = np.array(labels)
            return embeddings, labels
        else:
            return embeddings


class SimilarityAnalyzer:
    """Handles similarity analysis and retrieval"""
    
    @staticmethod
    def fetch_similar_images_with_labels(query_embeddings, gallery_embeddings, 
                                       gallery_images, gallery_labels, top_k=3):
        """Fetch similar images based on embeddings"""
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
    
    @staticmethod
    def visualize_similar_images(query_images, query_labels, similar_images, similar_labels):
        """Visualize query images with their similar matches"""
        for i, query_img in enumerate(query_images):
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(query_img, cmap='gray')
            axes[0].set_title(f'Query\n{query_labels[i]}')
            axes[0].axis('off')
            
            for j, (sim_img, sim_label) in enumerate(zip(similar_images[i], similar_labels[i])):
                axes[j+1].imshow(sim_img, cmap='gray')
                axes[j+1].set_title(f'{sim_label}', fontsize=10)
                axes[j+1].axis('off')
            
            plt.tight_layout()
            plt.show()


class TTAAnalyzer:
    """Handles Test-Time Augmentation analysis"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def extract_embeddings_tta(self, dataset, n_aug=5):
        """Extract embeddings with Test-Time Augmentation"""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                img, label = dataset[i]
                aug_embs = []
                for _ in range(n_aug):
                    aug_img, _ = dataset[i]
                    aug_img = aug_img.unsqueeze(0).to(self.device)
                    emb = self.model.get_embeddings(aug_img)
                    aug_embs.append(emb.cpu().numpy())
                
                avg_emb = np.mean(aug_embs, axis=0)
                embeddings.append(avg_emb)
                labels.append(label)
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        return embeddings, labels
    
    def visualize_tta_predictions(self, knn, dataset, idx, n_aug=5, label_encoder=None):
        """Visualize TTA predictions for a single image"""
        self.model.eval()
        aug_images = []
        aug_preds = []
        aug_embs = []
        
        with torch.no_grad():
            for _ in range(n_aug):
                img, label = dataset[idx]
                aug_images.append(img.squeeze(0).numpy())
                img = img.unsqueeze(0).to(self.device)
                emb = self.model.get_embeddings(img).cpu().numpy()
                aug_embs.append(emb)
                pred = knn.predict(emb)[0]
                aug_preds.append(pred)
        
        aug_embs = np.vstack(aug_embs)
        avg_emb = aug_embs.mean(axis=0, keepdims=True)
        avg_pred = knn.predict(avg_emb)[0]
        
        true_label = dataset[idx][1]
        if label_encoder:
            true_letter = label_encoder.inverse_transform([true_label])[0]
            avg_pred_letter = label_encoder.inverse_transform([avg_pred])[0]
            pred_letters = label_encoder.inverse_transform(aug_preds)
        else:
            true_letter = true_label
            avg_pred_letter = avg_pred
            pred_letters = aug_preds
        
        fig, axes = plt.subplots(1, n_aug, figsize=(2*n_aug, 3))
        for i, ax in enumerate(axes):
            ax.imshow(aug_images[i], cmap="gray")
            ax.set_title(pred_letters[i], fontsize=10)
            ax.axis("off")
        
        plt.suptitle(f"TTA predictions (true={true_letter}, avg_pred={avg_pred_letter})", fontsize=14)
        plt.show()


class OptimalClusterAnalyzer:
    """Analyzes optimal number of clusters per class"""
    
    @staticmethod
    def find_optimal_clusters_per_letter(embeddings, labels, label_encoder, max_k=10):
        """Find optimal number of clusters for each letter"""
        best_k_results = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            letter = label_encoder.inverse_transform([label])[0]
            print(f"\nProcessing letter: {letter}")
            
            label_indices = np.where(labels == label)[0]
            if len(label_indices) < 2:
                print(f"  Skipping letter '{letter}' - not enough samples for clustering (need at least 2).")
                continue
            
            label_embeddings = embeddings[label_indices]
            best_silhouette = -1
            best_k_for_letter = None
            
            for k in range(2, min(max_k, len(label_embeddings))):
                print(f"  Testing k = {k} for letter '{letter}'...")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(label_embeddings)
                
                try:
                    silhouette_avg = silhouette_score(label_embeddings, clusters)
                    print(f"    Silhouette Score for k={k}: {silhouette_avg:.4f}")
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_k_for_letter = k
                except Exception as e:
                    print(f"    Could not calculate silhouette score for k={k}: {e}")
                    continue
            
            if best_k_for_letter is not None:
                best_k_results[letter] = {
                    'best_k': best_k_for_letter, 
                    'silhouette_score': best_silhouette
                }
                print(f"  Best k for letter '{letter}': {best_k_for_letter} (Silhouette: {best_silhouette:.4f})")
            else:
                print(f"  No valid k found for letter '{letter}'.")
        
        return best_k_results
    
    @staticmethod
    def visualize_letter_clusters(embeddings, labels, letter_label, label_encoder, n_clusters=3):
        """Visualize clusters within a specific letter"""
        letter_indices = np.where(labels == letter_label)[0]
        letter_embeddings = embeddings[letter_indices]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        letter_clusters = kmeans.fit_predict(letter_embeddings)
        
        return letter_indices, letter_clusters


class ProcrustesAnalyzer:
    """Handles Procrustes analysis for shape alignment"""
    
    @staticmethod
    def align_images_procrustes(image_path1, image_path2, size=(64, 64)):
        """Align two images using Orthogonal Procrustes analysis"""
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
            
            def resample_contour(contour, num_points):
                points = []
                for i in range(num_points):
                    index = int((i / (num_points - 1)) * len(contour)) % len(contour)
                    points.append(contour[index][0])
                return np.array(points, dtype=np.float32)
            
            n_points = 100
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
                return None, None, None
            
            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(points1[:, 0], points1[:, 1], label='Image 1 (Original)', alpha=0.6)
            plt.scatter(points2[:, 0], points2[:, 1], label='Image 2 (Original)', alpha=0.6)
            plt.title('Original Contour Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.gca().invert_yaxis()
            
            plt.subplot(1, 2, 2)
            plt.scatter(points1[:, 0], points1[:, 1], label='Image 1 (Target)', alpha=0.6)
            plt.scatter(aligned_points2[:, 0], aligned_points2[:, 1], label='Image 2 (Aligned)', alpha=0.6)
            plt.title('Aligned Contour Points')
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


class DataMerger:
    """Handles data merging and reclassification"""
    
    @staticmethod
    def create_merge_map():
        """Create mapping for merging similar letters"""
        return {
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
    
    @staticmethod
    def apply_merging(data, merge_map, delete_classes=['Psi', 'Zeta']):
        """Apply merging and filtering to data"""
        merged_data = data.copy()
        merged_data['merged_letter'] = merged_data['letter'].apply(
            lambda x: merge_map.get(x, x)
        )
        
        # Filter out delete classes
        merged_data_filtered = merged_data[
            ~merged_data['letter'].isin(delete_classes)
        ].copy()
        
        return merged_data_filtered


class AnalysisReporter:
    """Generates analysis reports"""
    
    @staticmethod
    def print_clustering_results(results, method_name):
        """Print clustering evaluation results"""
        print(f"\n{method_name} Results:")
        print(f"Normalized Mutual Information (NMI): {results['nmi']:.4f}")
        print(f"Adjusted Rand Index (ARI): {results['ari']:.4f}")
        if 'silhouette' in results:
            print(f"Average Silhouette Score: {results['silhouette']:.4f}")
        print(f"Homogeneity: {results['homogeneity']:.4f}")
        print(f"Completeness: {results['completeness']:.4f}")
        print(f"V-measure: {results['v_measure']:.4f}")
    
    @staticmethod
    def generate_latex_table(classification_report_dict):
        """Generate LaTeX table from classification report"""
        latex_table = "\\begin{tabular}{|l|c|c|c|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "Class & Precision & Recall & F1-Score & Support \\\\\n"
        latex_table += "\\hline\n"
        
        for label, metrics in classification_report_dict.items():
            if isinstance(metrics, dict):
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = metrics['support']
                latex_table += f"{label} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {support} \\\\\n"
        
        latex_table += "\\hline\n"
        
        # Add macro avg
        macro_avg = classification_report_dict['macro avg']
        latex_table += f"Macro Avg & {macro_avg['precision']:.3f} & {macro_avg['recall']:.3f} & {macro_avg['f1-score']:.3f} & {macro_avg['support']} \\\\\n"
        
        # Add weighted avg
        weighted_avg = classification_report_dict['weighted avg']
        latex_table += f"Weighted Avg & {weighted_avg['precision']:.3f} & {weighted_avg['recall']:.3f} & {weighted_avg['f1-score']:.3f} & {weighted_avg['support']} \\\\\n"
        
        # Add accuracy
        accuracy = classification_report_dict['accuracy']
        latex_table += f"Accuracy & \\multicolumn{{3}}{{|c|}}{{{accuracy:.3f}}} & {weighted_avg['support']} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}"
        
        return latex_table


class BaseEmbeddingModel(nn.Module):
    """Abstract base class for embedding models"""
    def __init__(self):
        super(BaseEmbeddingModel, self).__init__()

    def get_embeddings(self, x):
        """Return embeddings for input batch"""
        raise NotImplementedError("Subclasses must implement get_embeddings()")

    def get_model_name(self):
        """Return a string identifier for this model"""
        return self.__class__.__name__


class CNN2D(BaseEmbeddingModel):
    """2D CNN for image data"""
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
            dummy_output = self.pool3(self.relu(self.conv3(
                self.pool2(self.relu(self.conv2(
                    self.pool1(self.relu(self.conv1(dummy_input)))))))))
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
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x

    def get_model_name(self):
        return "CNN2D"


class ResNetEmbedder(BaseEmbeddingModel):
    """Pretrained ResNet backbone for embeddings + classification head"""
    def __init__(self, num_classes, model_name="resnet18", pretrained=True):
        super(ResNetEmbedder, self).__init__()
        self.model_name = model_name
        
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            embedding_dim = 512
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            embedding_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Modify first conv layer to accept grayscale (1 channel instead of 3)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Add new classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embeddings = self.backbone(x)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(self, x):
        return self.backbone(x)

    def get_model_name(self):
        return f"ResNet_{self.model_name}"
    

class MLPBaseline(BaseEmbeddingModel):
    """Simple MLP baseline for flattened image input"""
    def __init__(self, input_dim, num_classes):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() > 2:  # e.g. (batch, 1, 64, 64)
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def get_embeddings(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def get_model_name(self):
        return "MLPBaseline"


class ModelManager:
    """Manages multiple embedding models for comparison"""
    
    def __init__(self, num_classes, image_size=(64, 64), device='cpu'):
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = device
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        
    def add_model(self, model_key, model_type, **model_kwargs):
        """Add a model to the manager"""
        if model_type == "CNN2D":
            model = CNN2D(self.num_classes, self.image_size)
        elif model_type == "ResNet18":
            model = ResNetEmbedder(self.num_classes, "resnet18", pretrained=True)
        elif model_type == "ResNet50":
            model = ResNetEmbedder(self.num_classes, "resnet50", pretrained=True)
        elif model_type == "MLP":
            input_dim = self.image_size[0] * self.image_size[1]  # flattened grayscale
            model = MLPBaseline(input_dim, self.num_classes)
        elif model_type == "CNN1D":
            input_dim = self.image_size[0] * self.image_size[1]
            model = CNN1D(input_dim, self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.to(self.device)
        self.models[model_key] = model
        
        # Create trainer and evaluator
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.trainers[model_key] = ModelTrainer(model, criterion, optimizer, self.device)
        self.evaluators[model_key] = ModelEvaluator(model, self.device)
        
        print(f"Added {model_type} model with key '{model_key}'")
        return model
    
    def train_model(self, model_key, train_loader, val_loader, num_epochs=100):
        """Train a specific model, or load if checkpoint exists"""
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
            
        save_path = f'best_{model_key}_model.pth'
        trainer = self.trainers[model_key]

        # If checkpoint exists, load and skip training
        if os.path.exists(save_path):
            print(f"Checkpoint found for {model_key}, loading from {save_path} and skipping training.")
            self.models[model_key].load_state_dict(torch.load(save_path, map_location=self.device))
            return trainer

        print(f"\nTraining {model_key}...")
        trainer.train(train_loader, val_loader, num_epochs, save_path)
        return trainer
    
    def evaluate_model(self, model_key, test_loader, label_encoder):
        """Evaluate a specific model"""
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
            
        return self.evaluators[model_key].evaluate(test_loader, label_encoder)
    
    def extract_embeddings(self, model_key, dataloader):
        """Extract embeddings from a specific model"""
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found")
            
        return self.evaluators[model_key].extract_embeddings(dataloader)
    
    def get_model(self, model_key):
        """Get a specific model"""
        return self.models.get(model_key)
    
    def get_all_model_keys(self):
        """Get all model keys"""
        return list(self.models.keys())


class EmbeddingComparator:
    """Compares embeddings from different models"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.results = {}
        
    def compare_embeddings_knn(self, train_loader, test_loader, label_encoder, k=11):
        """Compare embeddings using KNN classification"""
        print("\n=== Comparing Embeddings with KNN ===")
        
        for model_key in self.model_manager.get_all_model_keys():
            print(f"\nExtracting embeddings from {model_key}...")
            
            # Extract embeddings
            train_embeddings, train_labels = self.model_manager.extract_embeddings(model_key, train_loader)
            test_embeddings, test_labels = self.model_manager.extract_embeddings(model_key, test_loader)
            
            # KNN classification
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
            knn.fit(train_embeddings, train_labels)
            y_pred = knn.predict(test_embeddings)
            
            # Calculate accuracy
            accuracy = accuracy_score(test_labels, y_pred)
            
            # Store results
            self.results[f"{model_key}_knn"] = {
                'accuracy': accuracy,
                'predictions': label_encoder.inverse_transform(y_pred),
                'true_labels': label_encoder.inverse_transform(test_labels),
                'embeddings': test_embeddings
            }
            
            print(f"{model_key} KNN Accuracy: {accuracy:.4f}")
            
        return self.results
    
    def compare_embeddings_clustering(self, dataloader, ground_truth_labels, n_clusters=24):
        """Compare embeddings using clustering"""
        print("\n=== Comparing Embeddings with Clustering ===")
        
        cluster_analyzer = ClusterAnalyzer()
        
        for model_key in self.model_manager.get_all_model_keys():
            print(f"\nClustering embeddings from {model_key}...")
            
            # Extract embeddings
            embeddings, _ = self.model_manager.extract_embeddings(model_key, dataloader)
            
            # K-means clustering
            clusters = cluster_analyzer.kmeans_clustering(embeddings, n_clusters=n_clusters)
            
            # Evaluate clustering
            results = cluster_analyzer.evaluate_clustering(ground_truth_labels, clusters, embeddings)
            
            # Store results
            self.results[f"{model_key}_clustering"] = results
            
            print(f"{model_key} Clustering Results:")
            print(f"  NMI: {results['nmi']:.4f}")
            print(f"  ARI: {results['ari']:.4f}")
            if 'silhouette' in results:
                print(f"  Silhouette: {results['silhouette']:.4f}")
                
        return self.results
    
    def visualize_embeddings_comparison(self, test_loader, label_encoder):
        """Create t-SNE visualizations for all models"""
        print("\n=== Creating t-SNE Visualizations ===")
        
        model_keys = self.model_manager.get_all_model_keys()
        n_models = len(model_keys)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
            
        for i, model_key in enumerate(model_keys):
            print(f"Creating t-SNE for {model_key}...")
            
            # Extract embeddings
            embeddings, labels = self.model_manager.extract_embeddings(model_key, test_loader)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot
            test_letters = label_encoder.inverse_transform(labels)
            for letter in np.unique(test_letters):
                idx = test_letters == letter
                axes[i].scatter(
                    embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=letter, alpha=0.7, s=40
                )
            
            axes[i].set_title(f't-SNE: {model_key}')
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.show()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*50)
        print("EMBEDDING COMPARISON REPORT")
        print("="*50)
        
        # KNN Results
        knn_results = {k: v for k, v in self.results.items() if 'knn' in k}
        if knn_results:
            print("\nKNN Classification Results:")
            print("-" * 30)
            for model_key, results in knn_results.items():
                model_name = model_key.replace('_knn', '')
                print(f"{model_name:15}: {results['accuracy']:.4f}")
        
        # Clustering Results
        clustering_results = {k: v for k, v in self.results.items() if 'clustering' in k}
        if clustering_results:
            print("\nClustering Results:")
            print("-" * 30)
            print(f"{'Model':<15} {'NMI':<8} {'ARI':<8} {'Silhouette':<12}")
            print("-" * 45)
            for model_key, results in clustering_results.items():
                model_name = model_key.replace('_clustering', '')
                sil_score = results.get('silhouette', 0.0)
                print(f"{model_name:<15} {results['nmi']:<8.4f} {results['ari']:<8.4f} {sil_score:<12.4f}")


