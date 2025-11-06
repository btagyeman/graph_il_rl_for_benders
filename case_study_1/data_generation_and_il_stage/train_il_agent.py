import numpy as np
import pickle
import numpy as np
import pickle
import os
import tensorflow as tf 
from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tqdm import tqdm 
import matplotlib.pyplot as plt


# Load original graph dataset
with open("./datasets/graph_gbd_dataset_3k_improved.pkl", "rb") as f_3k:
    graph_dataset_3k = pickle.load(f_3k)

graph_dataset = graph_dataset_3k


# Standardize the node features
all_x = np.concatenate([g.x for g in graph_dataset], axis=0)
x_mean = all_x.mean(axis=0)
x_std = all_x.std(axis=0) + 1e-8

for g in graph_dataset:
    g.x = (g.x - x_mean) / x_std


# Standardize the edge features
all_e = np.concatenate([g.e for g in graph_dataset], axis=0)
e_mean = all_e.mean(axis=0)
e_std = all_e.std(axis=0) + 1e-8

for g in graph_dataset:
    g.e = (g.e - e_mean) / e_std


class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list
    

class IL_AGENT(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.conv3 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.out_y1 = Dense(1)
        self.out_y2 = Dense(1)
        self.out_y3 = Dense(1)
        self.out_y4 = Dense(1)
        self.out_y5 = Dense(1)

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dense2(x)
        y1 = self.out_y1(x)
        y2 = self.out_y2(x)
        y3 = self.out_y3(x)
        y4 = self.out_y4(x)
        y5 = self.out_y5(x)
        return tf.concat([y1, y2, y3, y4, y5], axis=-1)
    
os.makedirs("saved_models", exist_ok=True)

# Save the stats for the training of the rl agent
np.save("saved_models/x_mean.npy", x_mean)
np.save("saved_models/x_std.npy", x_std)
np.save("saved_models/e_mean.npy", e_mean)
np.save("saved_models/e_std.npy", e_std)

# Train/test split
training_data = graph_dataset[:int(0.95 * len(graph_dataset))]
test_data = graph_dataset[int(0.95 * len(graph_dataset)):]  

train_dataset = GraphListDataset(training_data)
test_dataset = GraphListDataset(test_data)

batch_size = 128
n_epochs = 100 
loader = DisjointLoader(train_dataset, batch_size=batch_size, epochs=n_epochs, shuffle=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Initialize model
il_agent_a = IL_AGENT()

# Training loop
avg_train_losses = []
epoch_loss = 0
batches_in_epoch = 0
current_epoch = 1

for batch in tqdm(loader, total=loader.steps_per_epoch * n_epochs, desc="Training"):
    inputs, y_true = batch
    with tf.GradientTape() as tape:
        preds = il_agent_a(inputs, training=True)
        loss_y1 = loss_fn(y_true[:, 0], preds[:, 0])
        loss_y2 = loss_fn(y_true[:, 1], preds[:, 1])
        loss_y3 = loss_fn(y_true[:, 2], preds[:, 2])
        loss_y4 = loss_fn(y_true[:, 3], preds[:, 3])
        loss_y5 = loss_fn(y_true[:, 4], preds[:, 4])
        loss = (loss_y1 + loss_y2 + loss_y3 + loss_y4 + loss_y5) / 5
    grads = tape.gradient(loss, il_agent_a.trainable_variables)
    optimizer.apply_gradients(zip(grads, il_agent_a.trainable_variables))

    epoch_loss += loss.numpy()
    batches_in_epoch += 1

    if batches_in_epoch == loader.steps_per_epoch:
        avg_loss = epoch_loss / batches_in_epoch
        avg_train_losses.append(avg_loss)
        print(f"Epoch {current_epoch} - Avg Train Loss: {avg_loss:.4f}")
        current_epoch += 1
        epoch_loss = 0
        batches_in_epoch = 0

# Plotting loss curve
plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, marker='o', linestyle='-', color='#1f77b4', label='Train Loss')
plt.title("Training Loss Curve", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=13, fontweight='bold')
plt.ylabel("Average Loss", fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Evaluation on the test set
test_loader = DisjointLoader(GraphListDataset(test_data), batch_size=1, epochs=1, shuffle=True)
@tf.function(experimental_relax_shapes=True)
def predict(inputs):
    return il_agent_a(inputs, training=False)

@tf.function
def compute_errors(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae

y_preds = []
y_trues = []
def evaluate_il_agent(loader):
    total_mae = 0.0
    n_batches = 0
    pbar = tqdm(loader, total=loader.steps_per_epoch, desc="Evaluating")
    for batch in pbar:
        inputs, y_true = batch
        logits = predict(inputs)
        y_pred = tf.sigmoid(logits)  # Apply sigmoid to get probabilities
        y_preds.append(y_pred.numpy().squeeze())
        y_trues.append(y_true.squeeze())
        mae = compute_errors(y_true, y_pred)
        total_mae += mae
        n_batches += 1
    avg_mae = total_mae.numpy() / n_batches
    print(f"Test MAE: {avg_mae:.6f}")
    return 

evaluate_il_agent(test_loader)

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)
THRESH_HIGH = 0.95
THRESH_LOW = 0.05

plt.figure(figsize=(8, 4))
plt.hist(y_preds.flatten(), bins=30, color='skyblue', edgecolor='k')
plt.axvline(THRESH_LOW, color='red', linestyle='--', label='Low threshold')
plt.axvline(THRESH_HIGH, color='green', linestyle='--', label='High threshold')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

confident_mask = (y_preds >= THRESH_HIGH) | (y_preds <= THRESH_LOW)
binary_preds = np.where(y_preds >= THRESH_HIGH, 1, np.where(y_preds <= THRESH_LOW, 0, -1))
correct = (binary_preds == y_trues)
correct_confident = correct[confident_mask]
accuracy = correct_confident.sum() / confident_mask.sum()
confident_errors = ((binary_preds != y_trues) & confident_mask).sum()
uncertain_mask = (binary_preds == -1)
uncertainty_rate = uncertain_mask.sum() / np.prod(y_trues.shape)

n_vars = y_trues.shape[1]
print("\nPer-variable confident accuracy:")
for i in range(n_vars):
    mask_i = confident_mask[:, i]
    correct_i = (binary_preds[:, i] == y_trues[:, i])[mask_i]
    acc_i = correct_i.mean() if mask_i.sum() > 0 else np.nan
    print(f"  y_{i+1}: {acc_i:.4f} accuracy on {mask_i.sum()} confident predictions")

print("\n=== Confidence-Based Evaluation Summary ===")
print(f"Thresholded Accuracy (confident preds only): {accuracy:.4f}")
print(f"Confidence coverage: {confident_mask.sum()} / {np.prod(y_trues.shape)} ({(confident_mask.sum() / np.prod(y_trues.shape)) * 100:.2f}%)")
print(f"Confident misclassifications: {confident_errors}")
print(f"Uncertain predictions: {uncertain_mask.sum()} ({uncertainty_rate * 100:.2f}%)")

correct_confident_count = correct_confident.sum()
confident_error_count = int((binary_preds != y_trues)[confident_mask].sum())
uncertain_count = (binary_preds == -1).sum()
labels = ['Correct (Confident)', 'Incorrect (Confident)', 'Uncertain']
sizes = [correct_confident_count, confident_error_count, uncertain_count]
colors = ['#4F81BD', '#6BAED6', '#A6CEE3']

plt.figure(figsize=(14, 6))
bar_height = 0.4
bars = plt.barh(labels, sizes, color=colors, height=bar_height)
plt.xlabel("Number of Predictions", fontsize=13, fontweight='bold')
plt.title("Variable-wise Prediction Outcomes", fontsize=14, fontweight='bold')
for bar, size in zip(bars, sizes):
    plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
             f"{size}", va='center', fontsize=12, fontweight='bold', color='black')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xticks(fontsize=11)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

#Save the trained model
# il_agent_a.save("saved_models/il_agent")