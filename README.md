# GROUP 32 - FOUNDATION OF DATA MINING PROJECT

# Description
This project contain the implementation of clustering algorithms & classification experiments, which is prerequisite project for "Foundation of Data Mining" module.

# Submitted By:
* Muhammad Shahzaib - 5019437
* Aman Khan - 5015811

## You can find the project on BTU gitlab

* [dm_project_group_32](https://git.informatik.tu-cottbus.de/shahzmuh/dm_project_group_32)

## Task 1 - Implementation of Clustering Algorithms

The task 1 is related to the implementation of two clustering algorithms
* Kmeans
* DBSCAN

## Where you can find the files?

There is a folder named "clustering". In that folder you can find these files:
* utils.py 
  * This file contain all the utility functions needed for algorithms
* k_mean_algo.py 
  * As by name, this file contain the kmeans algorithm implementation
* dbscan_algo.py 
  * As by name, this file contain the dbscan algorithm implementation
* main 
  * This is the starting point of our program

## Input & Output

There are two folders for our images: 
* input_images
  * This folder contain two types of images:
    * x_kmeans.png 
    * x_dbscan.jpg
    * where x=1,2,3
  * We are using different images for dbscan as it is not optimal for images that contain large pixels (data points)
* output_images
  * In this folder we are storing our segmented images

## Project setup for clustering algos
You need two libraries numpy and pillow. These both are included in requirements.txt file. First make virtual environment and then run the command bellow:

```
pip install requirements.txt
```

## How to test

To test this task. First go to the "clustering" folder.


```
cd clustering
```
Run this command for kmeans algorithm:
```
python .\main.py --input input_images\1_kmeans.png --output output_images\result.png --algorithm kmeans --k 4 --distance euclidean --verbose
```
Run this command for dbscan algorithm:
```
python .\main.py --input input_images/1_dbscan.jpg --output output_images/result.jpg --algorithm dbscan --eps 10 --min-samples 80 --verbose
```
Explanation:
* Run this command for complete details:
  * ``python main.py --help``
* --input
  * Image path for input
* --output
  * Image path for segmented image
* --algorithm
  * kmeans or dbscan
* --k
  * Number of clusters
* --distance
  * Distance function which you want to use
    * euclidean (default)
    * manhattan
    * maximum
* --eps
  * Value for epsilon (Minimum distance which needed)
* --min-samples (Minimum number of points that needed that make point a core point)
* --verbose
  * For extra details


***

# Task 2 - Classification experiments

This notebook implements a custom **k-Nearest-Neighbor (KNN)** classifier from scratch and benchmarks its performance (accuracy and runtime) against the standard Scikit-Learn implementations of **KNN** and **Decision Trees**. The testing is performed on two classic UCI datasets: **Iris** and **Wine**.

### 1. Where to find the notebook ?
You can find the notebook inside the KNN folder with the title KNN.ipynb.

## 2. Prerequisites
The notebook requires a Python environment with the following libraries. The first cell in the notebook will automatically install these for you via `pip`:
* `ucimlrepo` (To fetch the Iris and Wine datasets)
* `numpy`
* `matplotlib` & `seaborn` (For visualizations)
* `scikit-learn`

## 3. How to Run the Notebook
1.  **Open the Notebook:** Load the `.ipynb` file in Google Colab, Jupyter Notebook, or VS Code.
2.  **Run All Cells:** You can run all cells in sequence (typically `Ctrl + F9` in Colab).
3.  **Step-by-Step Execution:**
    * **Cell 1:** Installs dependencies.
    * **Cell 2:** Imports necessary libraries.
    * **Cell 3:** Defines the custom `KNNClassifier` class.
    * **Cell 4 & 5:** Fetches data from the UCI repository and executes the comparison logic.
    * **Cell 6:** Displays the raw performance metrics (Accuracy, Train/Pred time).
    * **Cell 7-9:** Generates the **Scientific Report** and **Confusion Matrices**.

## 4. Key Components to Inspect

### The Custom KNN Algorithm
The class `KNNClassifier` (Cell 3) uses **Euclidean Distance** to identify neighbors. It is a "Lazy Learner," meaning the `fit` method merely stores data ($O(1)$ complexity), while the `predict` method performs the heavy distance computation ($O(N)$ complexity).

### The Benchmarking Logic
In **Cell 5**, the function `run_comparison` performs the following:
* Splits data into **70% Training** and **30% Test** sets.
* Uses a `random_state=42` to ensure the professor sees the exact same results as the student.
* Sets $k=5$ for both the custom and scikit-learn KNN models.

### Visual Results
* **Performance Table:** Located under the "Presentation of Results" section. Note that the **Custom KNN** will have a significantly lower training time but a higher prediction time than the Decision Tree.
* **Confusion Matrices:** A 2x3 grid of heatmaps will be generated. Look for the **Wine** dataset results; they highlight why the Decision Tree (which is scale-invariant) often outperforms KNN (which is scale-sensitive).

## 5. Interpreting the Results
When grading or testing, the professor should observe:
1.  **Mathematical Correctness:** The **Custom KNN** and **Sklearn KNN** should yield nearly identical accuracy scores, verifying the student's implementation.
2.  **Runtime Trade-offs:** The report correctly identifies that the Decision Tree is the fastest at making predictions, while the custom KNN is the slowest at prediction due to its brute-force calculation.
3.  **Data Sensitivity:** The **Wine** dataset results illustrate the need for feature scaling in distance-based models—a key discussion point in the included scientific report.

























