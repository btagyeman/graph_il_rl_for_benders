import numpy as np
import pickle
import tensorflow as tf 
from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os

os.makedirs("results/saved_agents", exist_ok=True)


def elementwise_scale_node_feature(graph_dataset, scale_range=(-1, 1), tol=1e-20):

    # Collect all node feature values
    all_vals = np.concatenate([g.x.flatten() for g in graph_dataset], axis=0)

    # Identify non-binary values
    is_non_binary = ~(
        np.isclose(all_vals, 0.0, atol=tol) | np.isclose(all_vals, 1.0, atol=tol)
    )

    # Extract only non-binary values for computing min and max
    non_binary_vals = all_vals[is_non_binary]
    if len(non_binary_vals) == 0:
        print("All values are binary (0 or 1). No scaling applied.")
        return graph_dataset, None, None

    x_mean = non_binary_vals.mean()
    x_std = non_binary_vals.std() + 1e-8  # avoid division by zero

    # Apply per-element transformation
    for g in graph_dataset:
        x = g.x.flatten()
        x_scaled = []
        for val in x:
            if np.isclose(val, 0.0, atol=tol) or np.isclose(val, 1.0, atol=tol):
                x_scaled.append(val)
            else:
                scaled = (val - x_mean) / x_std  # Standardize to mean=0, std=1
                x_scaled.append(scaled)
        g.x = np.array(x_scaled, dtype=np.float32).reshape(-1, 1)
    return graph_dataset, x_mean, x_std

    




class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list


class IL_AGENT(Model):
    def __init__(self, num_outputs=14):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.conv3 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_heads = [Dense(1, activation='sigmoid') for _ in range(num_outputs)]

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dense2(x)  
        outputs = [head(x) for head in self.output_heads]  
        return tf.concat(outputs, axis=-1)  



il_agent_a = IL_AGENT()

# Load graph dataset
with open("./gbd_datasets/graph_gbd_dataset_scheduler_2009.pkl", "rb") as f_2009:
    graph_dataset_2009 = pickle.load(f_2009)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2010.pkl", "rb") as f_2010:
    graph_dataset_2010 = pickle.load(f_2010)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2011.pkl", "rb") as f_2011:
    graph_dataset_2011 = pickle.load(f_2011)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2012.pkl", "rb") as f_2012:
    graph_dataset_2012 = pickle.load(f_2012)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2013.pkl", "rb") as f_2013:
    graph_dataset_2013 = pickle.load(f_2013)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2014.pkl", "rb") as f_2014:
    graph_dataset_2014 = pickle.load(f_2014)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2015.pkl", "rb") as f_2015:
    graph_dataset_2015 = pickle.load(f_2015)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2016.pkl", "rb") as f_2016:
    graph_dataset_2016 = pickle.load(f_2016)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2017.pkl", "rb") as f_2017:
    graph_dataset_2017 = pickle.load(f_2017)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2018.pkl", "rb") as f_2018:
    graph_dataset_2018 = pickle.load(f_2018)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2019.pkl", "rb") as f_2019:
    graph_dataset_2019 = pickle.load(f_2019)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2020.pkl", "rb") as f_2020:
    graph_dataset_2020 = pickle.load(f_2020)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2021.pkl", "rb") as f_2021:
    graph_dataset_2021 = pickle.load(f_2021)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2022.pkl", "rb") as f_2022:
    graph_dataset_2022 = pickle.load(f_2022)

with open("./gbd_datasets/graph_gbd_dataset_scheduler_2023.pkl", "rb") as f_2023:
    graph_dataset_2023 = pickle.load(f_2023)




# combine the datasets
graph_dataset = (graph_dataset_2009 + graph_dataset_2010 + graph_dataset_2011 +
    graph_dataset_2012 + graph_dataset_2013 + graph_dataset_2014 +
    graph_dataset_2015 + graph_dataset_2016 + graph_dataset_2017 +
    graph_dataset_2018 + graph_dataset_2019 + graph_dataset_2020 + 
    graph_dataset_2021 + graph_dataset_2022 + graph_dataset_2023)

graph_dataset, x_mean, x_std = elementwise_scale_node_feature(graph_dataset)


# === Edge feature normalization ===
# Stack all edge features to compute mean and std
all_e = np.concatenate([g.e for g in graph_dataset ], axis=0)
e_mean = all_e.mean(axis=0)
e_std = all_e.std(axis=0) + 1e-8  # avoid division by zero


# Normalize edge features
for g in graph_dataset:
    if g.e is not None:
        # g.e = ((g.e - e_min) / (e_max - e_min + 1e-8)) * (e_norm_max - e_norm_min) + e_norm_min
        g.e = (g.e - e_mean) / e_std
        pass
    pass
np.random.seed(42)  # For reproducibility
np.random.shuffle(graph_dataset)  # Shuffle the dataset randomly

# Step 1: Split into train/test BEFORE fitting scaler
training_data = graph_dataset[:int(0.85*len(graph_dataset))]  # 85% for training
test_data = graph_dataset[int(0.85*len(graph_dataset)):]      # 15% for testing


# Now you can proceed to create datasets as before
train_dataset = GraphListDataset(training_data)
test_dataset = GraphListDataset(test_data)




#TODO : FIND A BETTER WAY TO SELECT THE HYPERPARAMETERS
batch_size = 128
n_epochs = 100
loader = DisjointLoader(train_dataset, batch_size=batch_size, epochs=n_epochs, shuffle=True)
# val_loader = DisjointLoader(val_dataset, batch_size=batch_size,epochs = 1, shuffle=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

avg_train_losses =  []
avg_val_losses = []
val_accuracies =  []
epoch_loss = 0
batches_in_epoch = 0
current_epoch = 1
for batch in tqdm(loader, total=loader.steps_per_epoch*n_epochs, desc="Training"):
    inputs, y_true = batch
    with tf.GradientTape() as tape:
        preds = il_agent_a(inputs, training=True)
        # loss = loss_fn(y_true, preds)
        loss = 0 
        for i in range(14):
            loss += loss_fn(y_true[:, i], preds[:, i])
        loss /= 14
    grads = tape.gradient(loss, il_agent_a.trainable_variables)
    optimizer.apply_gradients(zip(grads, il_agent_a.trainable_variables))
    
    epoch_loss += loss.numpy()
    batches_in_epoch += 1

    if  batches_in_epoch == loader.steps_per_epoch:
        avg_loss = epoch_loss / batches_in_epoch
        avg_train_losses.append(avg_loss)
        print(f"Epoch {current_epoch} - Avg Train Loss: {avg_loss:.4f}")
        current_epoch += 1
        epoch_loss = 0
        batches_in_epoch = 0

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

#Evaluation on the test set
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
        y_pred = predict(inputs)
        y_preds.append(y_pred.numpy().squeeze())
        y_trues.append(y_true.squeeze())
        mae = compute_errors(y_true, y_pred)
        total_mae += mae
        n_batches += 1
    avg_mae = total_mae.numpy() / n_batches
    print(f"Test MAE: {avg_mae:.6f}")
    return 
# Run evaluation
evaluate_il_agent(test_loader)




y_preds = np.array(y_preds)
y_trues = np.array(y_trues)
THRESH_HIGH = 0.80
THRESH_LOW = 0.20

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


#Variable-wise comparison
# --- Create confidence mask ---
confident_mask = (y_preds >= THRESH_HIGH) | (y_preds <= THRESH_LOW)
binary_preds = np.where(y_preds>=THRESH_HIGH,1,np.where(y_preds<=THRESH_LOW, 0, -1))  # -1 = uncertain
correct = (binary_preds == y_trues)
correct_confident = correct[confident_mask]
accuracy = correct_confident.sum() / confident_mask.sum()

# --- Confident misclassifications ---
confident_errors = ((binary_preds != y_trues) & confident_mask).sum()

# --- Uncertainty rate ---
uncertain_mask = (binary_preds == -1)
uncertainty_rate = uncertain_mask.sum() / np.prod(y_trues.shape)

# --- Per-variable confidence accuracy ---
n_vars = y_trues.shape[1]
accs = []
print("\nPer-variable confident accuracy:")
for i in range(n_vars):
    mask_i = confident_mask[:, i]
    correct_i = (binary_preds[:, i] == y_trues[:, i])[mask_i]
    acc_i = correct_i.mean() if mask_i.sum() > 0 else np.nan
    accs.append(acc_i)
    print(f"  y_{i+1}: {acc_i:.4f} accuracy on {mask_i.sum()} confident predictions")

# --- Summary ---
print("\n=== Confidence-Based Evaluation Summary ===")
print(f"Thresholded Accuracy (confident preds only): {accuracy:.4f}")
print(f"Confidence coverage: {confident_mask.sum()} / {np.prod(y_trues.shape)} "
      f"({(confident_mask.sum() / np.prod(y_trues.shape)) * 100:.2f}%)")
print(f"Confident misclassifications: {confident_errors}")
print(f"Uncertain predictions: {uncertain_mask.sum()} "
      f"({uncertainty_rate * 100:.2f}%)")


correct_confident_count = correct_confident.sum()
confident_error_count = int((binary_preds != y_trues)[confident_mask].sum())
uncertain_count = (binary_preds == -1).sum()
labels = ['Correct (Confident)', 'Incorrect (Confident)', 'Uncertain']
sizes = [correct_confident_count, confident_error_count, uncertain_count]
colors = ['#4F81BD', '#6BAED6', '#A6CEE3']

plt.figure(figsize=(14, 6))
bar_height = 0.4  # Default is ~0.8; reduce for slimmer bars
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



# Save the trained model
il_agent_a.save("./results/saved_agents/il_agent")
# Save the scaler parameters
np.save("./results/x_mean.npy", x_mean)
np.save("./results/x_std.npy", x_std)
np.save("./results/e_mean.npy", e_mean)
np.save("./results/e_std.npy", e_std)