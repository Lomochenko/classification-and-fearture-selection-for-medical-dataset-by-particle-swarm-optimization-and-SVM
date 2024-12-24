from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to preprocess and split data
def preprocess_data(X, y):
    # Handle missing values by replacing them with the mean of the column
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)

# PSO implementation
def pso_run(X_train, X_test, y_train, y_test, n_features, n_particles=20, n_iterations=50):
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient

    # Bounds for parameters C and gamma (log scale)
    c_min, c_max = -5, 5
    gamma_min, gamma_max = -5, 5

    # Initialize particle positions and velocities
    positions = np.random.rand(n_particles, n_features + 2)
    positions[:, :n_features] = (positions[:, :n_features] > 0.5).astype(float)  # Binary for feature selection
    velocities = np.random.uniform(-1, 1, (n_particles, n_features + 2))

    # Initialize personal and global bests
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.full(n_particles, np.inf)
    global_best_position = None
    global_best_score = np.inf

    # Objective function
    def evaluate_particle(position):
        feature_mask = position[:n_features] > 0.5
        if not np.any(feature_mask):
            return 1e6
        C = 10 ** position[n_features]
        gamma = 10 ** position[n_features + 1]
        X_train_selected = X_train[:, feature_mask]
        X_test_selected = X_test[:, feature_mask]
        svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
        svm.fit(X_train_selected, y_train)
        y_pred = svm.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        n_selected_features = np.sum(feature_mask)
        cost = (100 - accuracy * 100) + (n_selected_features / n_features) * 10
        return cost

    # PSO main loop
    for _ in range(n_iterations):
        for i in range(n_particles):
            score = evaluate_particle(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i]
        for i in range(n_particles):
            r1 = np.random.rand(n_features + 2)
            r2 = np.random.rand(n_features + 2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )
            positions[i] += velocities[i]
            positions[i, n_features] = np.clip(positions[i, n_features], c_min, c_max)
            positions[i, n_features + 1] = np.clip(positions[i, n_features + 1], gamma_min, gamma_max)
            positions[i, :n_features] = (positions[i, :n_features] > 0.5).astype(float)

    best_feature_mask = global_best_position[:n_features] > 0.5
    best_C = 10 ** global_best_position[n_features]
    best_gamma = 10 ** global_best_position[n_features + 1]
    X_train_selected = X_train[:, best_feature_mask]
    X_test_selected = X_test[:, best_feature_mask]
    final_svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf', random_state=42)
    final_svm.fit(X_train_selected, y_train)
    final_accuracy = accuracy_score(y_test, final_svm.predict(X_test_selected))
    n_selected_features = np.sum(best_feature_mask)
    return global_best_score, final_accuracy, n_selected_features, np.where(best_feature_mask)[0]

# Datasets configuration
datasets = {
    "Diabetes": {
        "fetch_function": lambda: fetch_openml(name="diabetes", version=1, as_frame=True),
        "target_processing": lambda y: (y > np.mean(y)).astype(int),  # Binary classification
    },
    "Hepatitis": {
        "fetch_function": lambda: fetch_openml(name="hepatitis", version=1, as_frame=True),
        "target_processing": lambda y: (y == '2').astype(int),  # Binary classification
    },
    "Liver Disorder": {
        "fetch_function": lambda: fetch_openml(name="liver-disorders", version=1, as_frame=True),
        "target_processing": lambda y: (y > np.mean(y)).astype(int),  # Binary classification
    },
}

# Run PSO for each dataset
results = {}
for dataset_name, config in datasets.items():
    data = config["fetch_function"]()
    X = data.data.values
    y = config["target_processing"](data.target.values)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    n_features = X_train.shape[1]
    results[dataset_name] = pso_run(X_train, X_test, y_train, y_test, n_features)

# Print results
for dataset_name, result in results.items():
    print(f"{dataset_name} Results:")
    print(f"Best Cost: {result[0]:.2f}")
    print(f"Final Accuracy: {result[1] * 100:.2f}%")
    print(f"Number of Selected Features: {result[2]}")
    print(f"Selected Features Indices: {result[3]}")
    print("-" * 30)
