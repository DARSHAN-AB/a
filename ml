import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Types & Missing Values ---")
print(df.info())
print("\nMissing values:", df.isnull().sum())
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))
print("\n--- Class Distribution ---")
print(df['species'].value_counts())

df.hist(figsize=(8,6))
plt.suptitle("Histograms of Iris Features", fontsize=14)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df, orient="h")
plt.title("Boxplots of Iris Features")
plt.show()

sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Pairwise Scatter Plots of Iris Features", y=1.02)
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df,s=70)
plt.title("Sepal Length vs Sepal Width")
plt.show()


==================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url ="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

X = df[['rm']] # Feature must be 2D
y = df['medv'] # Target

model = LinearRegression()
model.fit(X, y)
print("\nIntercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

y_pred = model.predict(X)

print("\nMean Squared Error:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))

# Regression Line Plot
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Median home value (MEDV)")
plt.title("Simple Linear Regression: RM vs MEDV")
plt.legend()
plt.show()
# Residuals Plot
residuals = y - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Predicted MEDV")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.show()

========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, classification_report
# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
stratify=y)
# Create Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print results
print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

=======================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 1️⃣Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2] # Take only the first 2 features for 2D visualization
y = iris.target
# 2️⃣Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42, stratify=y
)
# 3️⃣Standardize the features for better k-NN performance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Function to plot decision boundaries for different k values
def plot_decision_boundaries(k_values):
    h = 0.02 # step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['red', 'green', 'blue']
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    plt.figure(figsize=(15, 4))
    for i, k in enumerate(k_values):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.subplot(1, len(k_values), i + 1)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,edgecolor='k', s=40, cmap=ListedColormap(cmap_bold))
        plt.title(f"k = {k}")
plt.suptitle("k-NN Decision Boundaries for Different k Values")
plt.show()
# 4️⃣Experiment with different k values and visualize
k_values = [1, 5, 15]
plot_decision_boundaries(k_values)
# 5️⃣Evaluate accuracy for different k
print("Accuracy on Test Set:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k = {k}: {acc:.2f}")

=================================================================================================
