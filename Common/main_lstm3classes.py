
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import os
import time

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. This script will run on CPU, which may be slower.")
    exit()
else:
    print("CUDA is available")

# Hyperparameters
SEQUENCE_LENGTH = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
TRAIN_WINDOW_SIZE = 30000
TEST_HORIZON = 5000
STEP_SIZE = TEST_HORIZON
DROPOUT_RATE = 0.4
hidden_size = 256  # Increased from 256 to 512
BIDIRECTIONAL = True
CHECKPOINT_INTERVAL = 5

# Data Preparation
file_path = 'C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/4_0_5TP_1SL/merge'
file_name = "20190303_20240523_4TicksRevFeaturesCropped_wihSelection"
file_name ="20190303_20240523_4TicksRevFeaturesCropped_ALL_1-99"
file_namePCA = "20190303_20240523_4TicksRevFeaturesPCA6"

# Demander à l'utilisateur de choisir le fichier
#user_input = input(
 #   "Entrez 'p' pour sélectionner les données PCA ou un autre numéro pour garder les données originales : ")
user_input='rrr'

if user_input.lower() == 'p':
    file_full_path = os.path.join(file_path, f'{file_namePCA}.csv')

    # Extraire le nombre de colonnes PCA à partir du nom de fichier
    num_pca_columns = int(file_namePCA.split('PCA')[-1])
else:
    file_full_path = os.path.join(file_path, f'{file_name}.csv')

# Data Preparation
data = pd.read_csv(file_full_path, sep=';')  # Charger les données avec un séparateur point-virgule

if user_input.lower() == 'p':
    # Sélectionner les colonnes PCA spécifiées dans le nom de fichier
    pca_columns = [f'PCA_{i + 1}' for i in range(num_pca_columns)]
    X = data[pca_columns].values
else:
    X = data[[
       # 'deltaTimestamp',
        #'diffPriceClosePoc_0_0',
        'diffPriceCloseHigh_0_0',
        'diffPriceCloseLow_0_0',
        #'ratioVolCandleMeanx',
        #'ratioVolPocVolCandle',
        #  'ratioPocDeltaPocVol',
        #'ratioVolBlw',
        #'ratioVolAbv',
        #'ratioDeltaBlw',
        #'ratioDeltaAbv',
        'imbFactorAskL',
        'imbFactorBidH',
        #'bidVolumeAtBarLow',
        #'askVolumeAtBarLow',
        #'bidVolumeAtBarHigh',
        #'askVolumeAtBarHigh',
       # 'diffPocPrice_0_1',
        #'diffHighPrice_0_1',
        #'diffLowPrice_0_1',
        #'diffVolCandle_0_1',
        #'diffVolDelta_0_1',
       # 'diffPriceCloseVWAP',
        #'diffPriceCloseVWAPsd3Top',
        #'diffPriceCloseVWAPsd3Bot',
        'tradeDir'
    ]].values

y = data['tradeResult'].values

# Normalize input data
# scaler = MinMaxScaler()
scaler = StandardScaler()

if user_input.lower() != 'p':
    X = scaler.fit_transform(X)


# Split data into input sequences and labels
def create_subsequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        seq_X = X[i:i + seq_length]
        if len(seq_X) == seq_length:  # Vérifier si la longueur de la séquence est correcte
            X_seq.append(seq_X)
            y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)


X_sequences, y_sequences = create_subsequences(X, y, SEQUENCE_LENGTH)

# Convert class labels from (-1, 0, 1) to (0, 1, 2)
y_sequences = y_sequences + 1

# Verify label values after conversion
print(f"Unique labels after conversion: {np.unique(y_sequences)}")

# Afficher la répartition des classes
class_counts = np.bincount(y_sequences)
total_samples = np.sum(class_counts)
class_percentages = (class_counts / total_samples) * 100

print("Répartition des classes :")
for i, count in enumerate(class_counts):
    print(f"Classe {i} : {count} échantillons ({class_percentages[i]:.2f}%)")

# Convert to PyTorch tensors
X_sequences = torch.from_numpy(X_sequences).float()
y_sequences = torch.from_numpy(y_sequences).long()  # Use long for class labels

# Split data into train, validation, and test sets
train_val_size = int(0.9 * len(X_sequences))
X_train_val, X_test = X_sequences[:train_val_size], X_sequences[train_val_size:]
y_train_val, y_test = y_sequences[:train_val_size], y_sequences[train_val_size:]

# Convert class labels from (-1, 0, 1) to (0, 1, 2) for class weight calculation
y_train_val_converted = y_train_val.numpy()

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train_val_converted)
class_weights = 1. / class_counts

# Adjust class weights to give more weight to class 2
class_weights[2] = class_weights[2] * 4  # quadripuple the weight for class 2
class_weights[1] = class_weights[1] * 2  # quadripuple the weight for class 2

class_weights = torch.tensor(class_weights).float().to(DEVICE)

# Rearrange class weights to match the original class order (-1, 0, 1)
class_weights = torch.tensor([class_weights[2], class_weights[0], class_weights[1]]).to(DEVICE)

# Data augmentation to address class imbalance
def augment_data(X, y):
    X_aug = X.clone().detach()
    y_aug = y.clone().detach()

    for class_label in torch.unique(y):
        class_count = torch.sum(y == class_label).item()
        if class_count < BATCH_SIZE:
            indices = torch.where(y == class_label)[0]
            repeat_factor = int(BATCH_SIZE / class_count) + 1
            X_aug = torch.cat((X_aug, X[indices].repeat(repeat_factor, 1, 1)), dim=0)
            y_aug = torch.cat((y_aug, y[indices].repeat(repeat_factor)), dim=0)

    return X_aug, y_aug

# CNN-LSTM Model
# Modifications dans la définition du modèle pour augmenter les unités cachées

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, batch_first=True, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(hidden_size * (2 if bidirectional else 1), hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 128)
        self.fc2 = nn.Linear(128, output_size)  # Output size is 3 for multi-class classification
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)  # Apply Softmax activation
        return x

# Instantiate model with increased hidden units and additional LSTM layer

model = LSTMModel(X_sequences.shape[2], hidden_size, 3, bidirectional=BIDIRECTIONAL)
model.to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# TensorBoard writer
writer = SummaryWriter()

# Walk-Forward Validation
def walk_forward_validation(X_train_val, y_train_val, model, criterion, optimizer, scheduler, anchored=True):
    total_val_loss = 0.0
    total_val_accuracy = 0.0
    total_val_precision = np.zeros(3)  # Precision for each class
    total_val_recall = np.zeros(3)  # Recall for each class
    total_val_f1 = np.zeros(3)  # F1 score for each class

    start_idx = 0
    end_idx = TRAIN_WINDOW_SIZE + TEST_HORIZON
    # Calculate the total number of loops
    total_loops = (len(X_train_val) - TRAIN_WINDOW_SIZE - TEST_HORIZON) // STEP_SIZE + 1
    current_loop = 0

    while end_idx <= len(X_train_val):
        current_loop += 1
        print(f"Loop {current_loop}/{total_loops}")
        # Split data for the current window
        X_train = X_train_val[start_idx:start_idx + TRAIN_WINDOW_SIZE]
        y_train = y_train_val[start_idx:start_idx + TRAIN_WINDOW_SIZE]
        X_val = X_train_val[start_idx + TRAIN_WINDOW_SIZE:end_idx]
        y_val = y_train_val[start_idx + TRAIN_WINDOW_SIZE:end_idx]

        # Data augmentation
        X_train, y_train = augment_data(X_train, y_train)

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Training and evaluation loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()  # Enregistrer le temps de début

            train_loss = 0.0
            val_loss = 0.0

            # Training
            model.train()
            for X_batch, y_batch in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True):
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                # Assert labels are in the correct range
                assert torch.all(y_batch >= 0) and torch.all(y_batch < 3), f"Invalid label values in y_batch: {y_batch}"

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for X_val, y_val in DataLoader(val_dataset, batch_size=BATCH_SIZE):
                    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)

                    # Assert labels are in the correct range
                    assert torch.all(y_val >= 0) and torch.all(y_val < 3), f"Invalid label values in y_val: {y_val}"

                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)

                    # Calculate predictions and labels for metrics
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y_val.cpu().numpy())

            # Logging and early stopping
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            end_time = time.time()  # Enregistrer le temps de fin
            elapsed_time = (end_time - start_time) * 1000  # Calculer le temps écoulé en millisecondes

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed_time:.2f}ms')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), f'best_model_{start_idx}.pth')  # Save the best model
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f'Early stopping after {epoch + 1} epochs.')
                    break

            scheduler.step(val_loss)
            print(f'Learning rate: {scheduler.optimizer.param_groups[0]["lr"]}')
            # Checkpoint the model every CHECKPOINT_INTERVAL epochs
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                torch.save(model.state_dict(), f'checkpoint_model_{start_idx}_epoch_{epoch + 1}.pth')

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)  # Precision for each class
        val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)  # Recall for each class
        val_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)  # F1 score for each class

        # Check for absent predictions and log warnings
        for i, label in enumerate([0, 1, 2]):  # Adjusted label values after conversion
            if val_precision[i] == 0:
                print(f"Warning: No predicted samples for class {label} in validation set during loop {current_loop}/{total_loops}")

        # Update total validation metrics
        total_val_loss += val_loss
        total_val_accuracy += val_accuracy
        total_val_precision += val_precision
        total_val_recall += val_recall
        total_val_f1 += val_f1

        if not anchored:
            # Move the window forward for non-anchored walk-forward validation
            start_idx += STEP_SIZE
        end_idx += STEP_SIZE

    # Calculate average validation metrics
    num_windows = (end_idx - TRAIN_WINDOW_SIZE - TEST_HORIZON) // STEP_SIZE + 1
    avg_val_loss = total_val_loss / num_windows
    avg_val_accuracy = total_val_accuracy / num_windows
    avg_val_precision = total_val_precision / num_windows
    avg_val_recall = total_val_recall / num_windows
    avg_val_f1 = total_val_f1 / num_windows

    return avg_val_loss, avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, start_idx


# Demander à l'utilisateur de choisir entre la validation walk-forward ancrée et non ancrée
validation_type = input("Entrez 'a' pour la validation walk-forward ancrée ou une autre touche pour la validation non ancrée : ")
anchored = validation_type.lower() == 'a'

# Perform walk-forward validation
avg_val_loss, avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, start_idx = walk_forward_validation(
    X_train_val, y_train_val, model, criterion, optimizer, scheduler, anchored=anchored
)

print(f'Average Validation Loss: {avg_val_loss:.4f}')
print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')
print(f'Average Validation Precision (0, 1, 2): {avg_val_precision[0]:.4f}, {avg_val_precision[1]:.4f}, {avg_val_precision[2]:.4f}')
print(f'Average Validation Recall (0, 1, 2): {avg_val_recall[0]::.4f}, {avg_val_recall[1]:.4f}, {avg_val_recall[2]:.4f}')
print(f'Average Validation F1 Score (0, 1, 2): {avg_val_f1[0]:.4f}, {avg_val_f1[1]:.4f}, {avg_val_f1[2]:.4f}')

writer.close()

# Final evaluation on the test set
model.load_state_dict(torch.load(f'best_model_{start_idx - STEP_SIZE}.pth'))
model.eval()

test_preds = []
test_labels = []
with torch.no_grad():
    for X_test_batch, y_test_batch in DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE):
        X_test_batch, y_test_batch = X_test_batch.to(DEVICE), y_test_batch.to(DEVICE)

        # Assert labels are in the correct range
        assert torch.all(y_test_batch >= 0) and torch.all(y_test_batch < 3), f"Invalid label values in y_test_batch: {y_test_batch}"

        outputs = model(X_test_batch)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y_test_batch.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds, average=None, zero_division=0)  # Precision for each class
test_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)  # Recall for each class
test_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)  # F1 score for each class

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision (0, 1, 2): {test_precision[0]:.4f}, {test_precision[1]:.4f}, {test_precision[2]:.4f}')
print(f'Test Recall (0, 1, 2): {test_recall[0]:.4f}, {test_recall[1]:.4f}, {test_recall[2]:.4f}')
# Créer un dictionnaire associant les classes aux scores F1
f1_scores = dict(zip([0, 1, 2], test_f1))

# Afficher les scores F1 pour chaque classe
print(f"Test F1 Score: 0: {f1_scores[0]:.4f}, 1: {f1_scores[1]:.4f}, 2: {f1_scores[2]:.4f}")