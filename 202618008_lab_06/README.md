## Lab Assignment 6:Feature Extraction and Machine Learning with Image and Text Data
- **Student Name:** Nandini Sanjaybhai Pipaliya
- **Student ID:** 202618008
- **Dataset1:** [Asphalt Crack Dataset - 400 Images (Mendeley Data)](https://data.mendeley.com/datasets/xnzhj3x8v4/1)
- **Dataset1:** [Email Spam Classification Dataset - 5,172 Emails (Kaggle)](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

| Part | Task | Data |
|---|---|---|
| A | Classify asphalt images as **cracked** or **non-cracked** using hand-crafted intensity and edge features | 400 images |
| B | Classify emails as **spam** or **non-spam** from word counts | 5,172 emails, 3,000 word columns |
| C | Speed up and improve the Part B model with a better representation | same as Part B |

## Project structure
 
```
.
├── 202618008_lab_06.ipynb        # notebook with all three parts
├── README.md
├── emails.csv                    # input for Parts B and C
├── image_data/                   # input for Part A
│   ├── Cracks/                   # label 1
│   └── NonCracks/                # label 0
├── asphalt_features.csv          # generated in A.2 (intensity features)
└── asphalt_edge_features.csv     # generated in A.3 (edge features)
```

**A.1 Loading and preprocessing.** Images are read with OpenCV, resized to 128×128, and converted to grayscale. The result is 400 colour images of shape `(128, 128, 3)` and 400 grayscale images of shape `(128, 128)`.
 
**A.2 Intensity features (NumPy).** For each grayscale image, 12 features are computed: mean brightness, contrast (standard deviation), variance, dark-pixel ratio (pixels below 50), bright-pixel ratio (pixels above 200), min, max, median, 25th and 75th percentiles, intensity range, and interquartile range.
 
**A.3 Edge features (Canny).** Canny edge detection (thresholds 50 and 150) gives four features: edge count, edge density, edge percentage, and mean intensity of the edge pixels.
 
**A.4 Classification.** The 16 features were combined into one table (400 rows, balanced at 200 per class) and split 80/20 with stratification (320 train, 80 test). Four classifiers were trained. Features are standardised for Logistic Regression, KNN and SVM.
 
| Model | Accuracy | Precision | Recall | F1 | Train (s) | Predict (ms/image) |
|---|---|---|---|---|---|---|
| KNN (k=5) | 0.950 | 0.929 | 0.975 | 0.951 | 0.008 | 5.96 |
| Logistic Regression | 0.950 | 0.950 | 0.950 | 0.950 | 0.066 | 0.03 |
| SVM (RBF) | 0.938 | 0.907 | 0.975 | 0.940 | 0.011 | 0.04 |
| Random Forest (100 trees) | 0.900 | 0.900 | 0.900 | 0.900 | 0.348 | 1.31 |
 
**Takeaways**
- Logistic Regression gives the best balance of accuracy and speed.
- KNN and SVM have the highest recall (they catch the most cracks) but more false positives.
- KNN is the slowest at prediction time because it compares each image with all training samples.
- Random Forest scored lowest and had the longest training time.
- With only 80 test images, differences of a few percent correspond to one or two images.


## Part B: Text vectorization and spam classification
 
**Data.** `emails.csv` has 5,172 emails: 3,672 non-spam (71%) and 1,500 spam (29%). Each row holds the count of each of 3,000 words and the label in the `Prediction` column (1 = spam).
 
**Preprocessing.**
- The file already contains word counts, so no raw text is available and `CountVectorizer` are not used. The counts are used directly as the feature matrix.
- 541 exact duplicate emails were removed (4,631 remain), so that the same email cannot appear in both train and test.
- Stratified 80/20 split (3,704 train, 927 test).

**Models on raw counts.**
 
| Model | Accuracy | Precision | Recall | F1 | Train (s) |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.952 | 0.895 | 0.959 | 0.926 | 0.37 |
| Logistic Regression | 0.974 | 0.956 | 0.962 | 0.959 | 14.35 |
 
Logistic Regression is more accurate (13 false positives and 11 false negatives on the test set) but slow, because the matrix is dense and the counts are unscaled.
 
**Most informative words** (from the logistic regression coefficients). Toward spam: `men`, `pt`, `http`, `remove`. Toward non-spam: `enron`, `love`, `attached`, `deal`.
 
## Part C: Improving the representation
 
To fix the 14-second training time, the Part B model is rebuilt as a scikit-learn `Pipeline`:
 
1. **Sparse conversion** (`csr_matrix`): about 94% of the cells are zero, so only non-zero values are stored.
2. **`log1p` transform**: compresses large counts so a word repeated many times does not dominate.
3. **`MaxAbsScaler`**: scales each word to the range [0, 1] without breaking sparsity.
4. **Logistic Regression** (`C=1.0`).
| Setup | Accuracy | Spam F1 | False pos. / neg. | Train (s) |
|---|---|---|---|---|
| Original (raw counts) | 0.974 | 0.959 | 13 / 11 | 14.35 |
| Improved (sparse + log1p + MaxAbs) | 0.978 | 0.966 | 10 / 10 | 0.23 |
 
Training is about 60 times faster, and the total number of test errors fell from 24 to 20. Prediction time is similar in both cases (tens of milliseconds). The accuracy gain is small and comes from a single split, so it should be treated as "similar or slightly better", not as a proven improvement.
 
 
