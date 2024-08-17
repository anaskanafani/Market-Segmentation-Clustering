# Market Segmentation Project 📊👥

## 1. About ✍️
This project focuses on developing a clustering model to segment customers into distinct, meaningful groups. The segmentation is based on customer behavior, preferences, and demographics, using unsupervised learning techniques. The model provides insights that allow businesses to target specific customer segments more effectively. A user-friendly Streamlit interface is also integrated, enabling real-time visualization and interaction with the model.

## 2. Impact 🌍
The model's ability to identify unique customer segments allows businesses to tailor their marketing strategies, optimize customer engagement, and increase customer satisfaction. The segmentation results provide actionable insights that lead to more personalized customer experiences and improved resource allocation.

## 3. Methodology 🔬

### Data Preprocessing:
- **Missing Data Handling**: Missing values in the `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` columns were filled using the mean and median values.
- **Feature Scaling**: Data was standardized using `StandardScaler` to ensure that all features contribute equally to the clustering process.

### Clustering:
- **PCA**: Dimensionality reduction was performed using Principal Component Analysis (PCA) to reduce the dataset to two principal components.
- **KMeans Clustering**: The KMeans algorithm was applied to segment the data into 5 clusters.

### Model Evaluation:
- **Elbow Method**: Used to determine the optimal number of clusters for the KMeans algorithm.
- **Classification Report**: The model's performance was evaluated using metrics such as precision, recall, and F1-score.

```python
# Preprocessing data
scalar = StandardScaler()
scaled_df = scalar.fit_transform(df)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
```

## 4. Findings 🔍

- **Cluster Profiles**: Each cluster represents a distinct group of customers with unique spending patterns, payment behaviors, and preferences.
- **Cluster 1**: High balance with frequent purchases.
- **Cluster 2**: Moderate balance with a focus on installment purchases.
- **Cluster 3**: Low balance with infrequent purchases but high credit usage.
- **Cluster 4**: High credit usage with a focus on cash advances.
- **Cluster 5**: Consistent purchases with moderate credit usage.


## 5. Visualizations with Observations 📊🔍

### Elbow Method for Optimal k 📈

![output1](https://github.com/user-attachments/assets/a338da1f-d8bb-4d38-abf9-0583bdbfea8c)

Observations:
- The elbow curve shows a sharp decrease in inertia as the number of clusters increases from 1 to 4.
- The "elbow" of the curve appears around 4-5 clusters, suggesting this might be the optimal number.
- After 5 clusters, the decrease in inertia becomes more gradual, indicating diminishing returns.
- Based on this plot, choosing 5 clusters for the KMeans algorithm seems to be a good balance.

### KMeans Clustering 🎨

![output](https://github.com/user-attachments/assets/910101e4-21a1-4fa8-be9a-7e6aa8665f11)

Observations:
- The plot shows clear separation between different customer segments, indicating successful identification of distinct groups.
- Cluster 0 (red) appears to be the largest and most spread out, suggesting a diverse group of customers with varying behaviors.
- Cluster 1 (blue) is more compact and concentrated, possibly representing a more homogeneous group of customers.
- Clusters 2 (green) and 3 (yellow) overlap slightly, indicating some similarities between these segments.
- Cluster 4 (Black) is the smallest and most distinct, potentially representing a niche customer segment with unique behaviors.

### Feature Distributions 📉

![output3](https://github.com/user-attachments/assets/d20d15a1-5186-4f6d-98f7-6934f7df40cb)

Observations:
- Most features exhibit right-skewed distributions, indicating that a majority of customers have lower values for these attributes.
- 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENTS_FREQUENCY' show peaks at both ends of the scale, indicating distinct shopping behaviors.
- 'CASH_ADVANCE_FREQUENCY' is heavily skewed towards zero, implying that most customers rarely use cash advances.

### Feature Histograms 📊

![output4](https://github.com/user-attachments/assets/6a61728c-9556-4f29-8e84-317f0ec672be)

Observations:
- 'PURCHASES' and 'ONEOFF_PURCHASES' show similar distributions, with a long tail of high-value purchasers.
- 'INSTALLMENTS_PURCHASES' has a more uniform distribution, suggesting consistent use of installment plans.
- 'CASH_ADVANCE' and 'CASH_ADVANCE_TRX' both show extremely right-skewed distributions, confirming infrequent use.

### Cluster Distribution 📊

![output5](https://github.com/user-attachments/assets/73f064cb-82d0-4082-9546-bb248008e8ae)

Observations:
- Cluster 4 is the largest group, comprising about 40% of the total customers.
- Cluster 3 is the second largest, representing approximately 25% of the customers.
- Clusters 1 and 2 are medium-sized groups, each containing about 15-20% of the customers.
- Cluster 0 is the smallest group, making up less than 10% of the total customer base.
- The uneven distribution suggests that customer behaviors and characteristics are not uniformly spread, with certain patterns being more common than others.

### Confusion Matrix 🧩

![output7](https://github.com/user-attachments/assets/718543da-cd02-493c-ac44-d9f573180d7f)

Observations:
- The diagonal elements show high values, indicating good classification performance for most clusters.
- Cluster 4 has the highest correct classifications (1171), suggesting it's the most distinct and easily identifiable cluster.
- There's some misclassification between clusters 0 and 3, and between 1 and 3, indicating potential similarities between these groups.
- Clusters 1 and 2 show relatively low misclassification with other clusters, suggesting they have distinct characteristics.


## 6. Accuracy Results 📈

The classification model achieved an accuracy of **97.02%** on the test data, with a mean squared error of **0.126**.

```python
print(f'Accuracy: {round(accuracy * 100, 2)}%')
print("Mean Squared Error:", mse)
```

## 7. Technologies Used 🛠️

- **Python**: Primary programming language.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Matplotlib, Seaborn**: Data visualization.
- **scikit-learn**: Machine learning algorithms and model evaluation.
- **Streamlit**: Interactive user interface.
- **XGBoost**: Gradient boosting classifier.

## 8. Code Snippets 💻

### Data Loading and Preprocessing:

```python
df = pd.read_csv('../data/Customer_Data.csv')
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())
```

### Clustering:

```python
kmeans_model = KMeans(5)
kmeans_model.fit_predict(scaled_df)
pca_df_kmeans = pd.concat([pca_df, pd.DataFrame({'CLUSTER': kmeans_model.labels_})], axis=1)
```

### Model Evaluation:

```python
accuracy = best_pipeline.score(X_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}%')
```

## 9. Challenges and How They Were Overcome 🚧

- **High Dimensionality**: The dataset had many features, making clustering challenging. PCA was used to reduce dimensionality and improve model performance.
- **Missing Data**: Missing values in key columns were handled using median and mean imputation to avoid biases.

## 10. Conclusion 🏁

This project successfully segmented customers into meaningful clusters, providing valuable insights for personalized marketing. The model's integration with Streamlit enhances its usability, making it accessible to non-technical users.

## 11. How to Run on Your Machine 🖥️

### Prerequisites:
- Python 3.x
- Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Model:

```bash
streamlit run app.py
```

## 12. Deployment and Previews 🌐

### Streamlit Interface
- **Deployed Link:** [Market Segmentation Streamlit Website](https://market-segmentation-clustering.streamlit.app/)

The Streamlit app allows users to upload datasets, perform clustering, and visualize the results interactively. This deployment makes the model accessible to a wider audience and enables real-time experimentation with different clustering scenarios.
