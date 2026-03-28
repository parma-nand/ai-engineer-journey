import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Step 1: Load & Prepare Data ───────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df  = pd.read_csv(url)
df['Age']      = df['Age'].fillna(df['Age'].mean())
df['Fare']     = df['Fare'].fillna(df['Fare'].mean())
df['Sex']      = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = df[['Age', 'Sex', 'Pclass', 'Fare',
        'Embarked', 'SibSp', 'Parch']].values
y = df['Survived'].values
scaler = StandardScaler()
X      = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"TensorFlow version: {tf.__version__}")

# ── Step 2: Sequential API ────────────────
print("\n=== Approach 1: Sequential API ===")
model_seq = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu',
                          input_shape=(7,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile — set loss + optimizer + metrics
model_seq.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)

# Print model summary
model_seq.summary()
# Train — no manual loop needed!
print("\nTraining Sequential Model...")
history_seq = model_seq.fit(
    X_train, y_train,
    epochs          = 10,
    batch_size      = 32,
    validation_split= 0.2,   # auto validation split!
    verbose         = 1
)

# Evaluate
test_loss, test_acc = model_seq.evaluate(
    X_test, y_test, verbose=0)
print(f"\nSequential Model Test Accuracy: {test_acc*100:.1f}%")
# ── Step 3: Functional API ────────────────
print("\n=== Approach 2: Functional API ===")
# More flexible — can have multiple inputs/outputs
inputs  = tf.keras.Input(shape=(7,))
x       = tf.keras.layers.Dense(32, activation='relu')(inputs)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dropout(0.3)(x)
x       = tf.keras.layers.Dense(16, activation='relu')(x)
x       = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model_func = tf.keras.Model(inputs=inputs,
                             outputs=outputs)
model_func.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)
print("\nTraining Functional Model...")
history_func = model_func.fit(
    X_train, y_train,
    epochs           = 10,
    batch_size       = 32,
    validation_split = 0.2,
    verbose          = 0   # silent training
)
_, func_acc = model_func.evaluate(X_test, y_test,
                                   verbose=0)
print(f"Functional Model Test Accuracy: {func_acc*100:.1f}%")

# ── Step 4: Callbacks ─────────────────────
print("\n=== Approach 3: With Callbacks ===")
# Callbacks — automatic actions during training
callbacks = [
    # Stop early if no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor  = 'val_loss',
        patience = 10,          # wait 10 epochs
        restore_best_weights = True
    ),
    # Save best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        filepath     = 'best_model.keras',
        monitor      = 'val_accuracy',
        save_best_only = True
    ),
    # Reduce lr when stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,        # halve learning rate
        patience = 5
    )
]
model_cb = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu',
                          input_shape=(7,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])
model_cb.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)
print("Training with Callbacks...")
history_cb = model_cb.fit(
    X_train, y_train,
    epochs           = 100,   # high epochs
    batch_size       = 32,
    validation_split = 0.2,
    callbacks        = callbacks,
    verbose          = 0
)
actual_epochs = len(history_cb.history['loss'])
print(f"Early stopping kicked in at epoch: {actual_epochs}")
_, cb_acc = model_cb.evaluate(X_test, y_test, verbose=0)
print(f"Callbacks Model Test Accuracy: {cb_acc*100:.1f}%")
# ── Step 5: PyTorch vs TF Comparison ──────
print("\n=== PyTorch vs TensorFlow Syntax ===")
print("""
      PYTORCH:                    TENSORFLOW/KERAS:
─────────────────────────────────────────────
nn.Linear(7, 32)         →  Dense(32)
nn.ReLU()                →  activation='relu'
nn.Dropout(0.3)          →  Dropout(0.3)
nn.BatchNorm1d(32)       →  BatchNormalization()
nn.BCELoss()             →  'binary_crossentropy'
optim.Adam(lr=0.001)     →  Adam(lr=0.001)
model.fit() manual loop  →  model.fit() automatic!
model.eval()             →  model.predict()
""")
# ── Step 6: Visualize Training ────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Accuracy comparison
axes[0].plot(history_seq.history['accuracy'],
             label='Sequential Train', color='blue')
axes[0].plot(history_seq.history['val_accuracy'],
             label='Sequential Val',   color='blue',
             linestyle='--')
axes[0].plot(history_func.history['accuracy'],
             label='Functional Train', color='green')
axes[0].plot(history_func.history['val_accuracy'],
             label='Functional Val',   color='green',
             linestyle='--')
axes[0].set_title("Training vs Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
# Loss comparison
axes[1].plot(history_seq.history['loss'],
             label='Sequential Train', color='blue')
axes[1].plot(history_seq.history['val_loss'],
             label='Sequential Val',   color='blue',
             linestyle='--')
axes[1].plot(history_cb.history['loss'],
             label='Callbacks Train',  color='red')
axes[1].plot(history_cb.history['val_loss'],
             label='Callbacks Val',    color='red',
             linestyle='--')
axes[1].set_title("Training vs Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
plt.suptitle("TensorFlow/Keras Training Results",
             fontsize=13, fontweight='bold')
plt.tight_layout()
# plt.savefig("tensorflow_results.png")
plt.show()

# ── Step 7: Save and Load Model ───────────
print("\n=== Save and Load Model ===")
# Save model
model_seq.save('titanic_model.keras')
print("Model saved!")
# Load model
loaded_model = tf.keras.models.load_model(
    'titanic_model.keras')
print("Model loaded!")
# Predict with loaded model
y_pred_prob = loaded_model.predict(X_test, verbose=0)
y_pred      = (y_pred_prob >= 0.5).astype(int).flatten()
loaded_acc  = accuracy_score(y_test, y_pred)
print(f"Loaded Model Accuracy: {loaded_acc*100:.1f}%")
# ── Step 8: Final Comparison ──────────────
print("\n=== Final Summary ===")
print(f"Sequential API  : {test_acc*100:.1f}%")
print(f"Functional API  : {func_acc*100:.1f}%")
print(f"With Callbacks  : {cb_acc*100:.1f}%")
print(f"\nEarly stopping stopped at epoch: {actual_epochs}")
print("\nKey TensorFlow advantages:")
print("✅ No manual training loop")
print("✅ Built-in validation split")
print("✅ Callbacks for automation")
print("✅ Easy model save/load")
print("✅ model.summary() for architecture")





