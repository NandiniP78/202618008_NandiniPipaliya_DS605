## Lab Assignment - 7 Image Representation, PCA, Eigenfaces and Face Recognition
- **Student Name:** Nandini Sanjaybhai Pipaliya
- **Student ID:** 202618008
- **Dataset:** [AT&T Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)


## 1. Project Overview

This project explores image representation, Principal Component Analysis (PCA), eigenfaces, image compression, and face recognition using the AT&T Database of Faces (formerly the ORL Face Database).

The main objective is to reduce the dimensionality of grayscale face images while preserving useful facial information and evaluating how well different recognition methods identify subjects.

## 2. Dataset

- **Dataset:** AT&T Database of Faces
- **Number of subjects:** 40
- **Images per subject:** 10
- **Total images:** 400
- **Image representation:** Grayscale
- **Image dimensions:** 112 × 92 pixels
- **Original features per image:** 10,304 pixel values

Each image belongs to one of 40 subjects. The images are converted into one-dimensional feature vectors before applying PCA.

The ORL face database
---------------------
The data directory contains a set of faces taken between April 1992 and
April 1994 at the Olivetti Research Laboratory in Cambridge, UK.

There are 10 different images of 40 distinct subjects. For some of the
subjects, the images were taken at different times, varying lighting
slightly, facial expressions (open/closed eyes, smiling/non-smiling)
and facial details (glasses/no-glasses).  All the images are taken
against a dark homogeneous background and the subjects are in
up-right, frontal position (with tolerance for some side movement).

The files are in PGM format and can be conveniently viewed using the 'xv'
program. The size of each image is 92x112, 8-bit grey levels. The images
are organised in 40 directories (one for each subject) named as:

		sX

where X indicates the subject number (between 1 and 40). In each directory
there are 10 different images of the selected subject named as:

		Y.pgm

where Y indicates which image for the specific subject (between 1 and 10).

When using these images, please give credit to Olivetti Research Laboratory.
A convenient reference is the face recognition work which uses some of
these images:

 F. Samaria and A. Harter 
  "Parameterisation of a stochastic model for human face identification"
  2nd IEEE Workshop on Applications of Computer Vision
  December 1994, Sarasota (Florida).

The paper is available via anonymous ftp from quince.cam-orl.co.uk and is
stored in pub/users/fs/IEEE_workshop.ps.Z

If you have any question, please email Ferdinando Samaria: fs@cam-orl.co.uk
---------------------

## 3. Methodology

### Part A: Image Representation

1. Load the face images and convert them to grayscale.
2. Inspect image dimensions, pixel types, and intensity values.
3. Flatten each image into a one-dimensional vector.
4. Create the feature matrix and corresponding subject labels.
5. Split the dataset into training and testing sets using an 80:20 stratified split with `random_state=42`.

The resulting dataset contains:

- Training images: 320
- Testing images: 80
- Subjects represented in both sets: 40

### Part B: PCA, Eigenfaces, and Compression

PCA is used to reduce the number of features while retaining important patterns of variation across the training faces.

The main steps are:

1. Calculate the mean face using the training images.
2. Center the training data by subtracting the mean face.
3. Compute principal directions using NumPy linear algebra.
4. Compare eigendecomposition and Singular Value Decomposition (SVD).
5. Calculate explained variance and cumulative explained variance.
6. Identify the number of components required to preserve at least 90% of the training variance.
7. Visualize eigenfaces and reconstruct images using selected components.

The experiment found that **98 principal components** were required to retain at least 90% of the training variance.

#### Reconstruction Results

| Metric | Result |
|---|---:|
| Components retained | 98 |
| Original features per image | 10,304 |
| Compressed features per image | 98 |
| Training reconstruction MSE | 154.68 |
| Testing reconstruction MSE | 334.99 |
| Reduction in per-image feature values | 99.05% |

MSE (Mean Squared Error) measures the average squared difference between original and reconstructed pixel values. Lower MSE indicates more accurate reconstruction.

The test reconstruction error was higher than the training reconstruction error, indicating that the retained PCA directions reconstructed the training images more accurately than the unseen test images.

### Part C: Face Recognition

A nearest-neighbor recognition method was implemented using Euclidean distance in PCA feature space.

For each test image:

1. Project the image into PCA space.
2. Calculate its distance from every projected training image.
3. Find the nearest training image.
4. Predict the subject label of that nearest image.
5. Compare the predicted label with the actual subject label.

#### Recognition Results

| Method | Accuracy | Correct Predictions | Incorrect Predictions |
|---|---:|---:|---:|
| PCA Nearest Neighbor (98 components) | 96.25% | 77/80 | 3 |
| Raw-Pixel Nearest Neighbor | 2.50% | 2/80 | 78 |

The raw-pixel nearest-neighbor method performed poorly compared with PCA-based recognition. Direct pixel distances are sensitive to image alignment, lighting, and facial expression.

### Part D: Traditional Machine-Learning Classifiers

KNN and SVM classifiers were trained using PCA features.

The following baseline settings were evaluated:

- **KNN:** `n_neighbors=1`
- **SVM:** RBF kernel with default regularization parameter `C=1`

#### Machine-Learning Comparison

| Method | Feature Representation | Settings | Test Accuracy | Correct Predictions |
|---|---|---|---:|---:|
| PCA Nearest Neighbor | 98 PCA components | Euclidean distance | 96.25% | 77/80 |
| KNN | 98 PCA components | `k=1` | 96.25% | 77/80 |
| SVM | 98 PCA components | RBF, `C=1` | 98.75% | 79/80 |

KNN with one neighbor uses the same nearest-neighbor decision rule in PCA space, so matching accuracy is expected.

SVM achieved the highest baseline accuracy by learning decision boundaries between the subject classes.

## 4. Hyperparameter and Component Experiments

Experiments were conducted with different numbers of PCA components and classifier settings.

The tested component counts were:

`10, 25, 50, 75, 98, 150, 200`

KNN was evaluated with:

`n_neighbors = 1, 3, 5`

SVM was evaluated with:

`C = 0.1, 1, 10, 100`

The SVM used an RBF kernel in these experiments.

#### Best Observed Configurations

| PCA Components | Classifier | Settings | Test Accuracy |
|---:|---|---|---:|
| 10 | SVM | RBF, `C=10` | 98.75% |
| 10 | SVM | RBF, `C=100` | 98.75% |
| 25 | SVM | RBF, `C=10` | 98.75% |
| 50 | KNN | `k=1` | 98.75% |
| 50 | SVM | RBF, `C=10` | 98.75% |
| 75 | SVM | RBF, `C=1` | 98.75% |
| 98 | SVM | RBF, `C=1` | 98.75% |

### Selected Configuration

The selected configuration from the reported experiments is:

- **Classifier:** SVM
- **Kernel:** RBF
- **Regularization parameter:** `C=10`
- **PCA components:** 10
- **Observed test accuracy:** 98.75%
- **Correct predictions:** 79 out of 80

This configuration achieved the same observed accuracy as the 98-component SVM baseline while using substantially fewer features.

Reducing the representation from 10,304 pixel values to 10 PCA features corresponds to a **99.90% reduction in per-image feature values**.

This is a feature-count reduction, not a complete measurement of storage requirements, since the PCA basis, mean face, and classifier model must also be stored.

**Important evaluation note:** The best configuration was selected after comparing multiple settings on the same test set. Its 98.75% score should therefore be treated as the best observed test result, not an unbiased final performance estimate. A validation split or cross-validation should be used for tuning before evaluating the selected configuration on a held-out test set.

## 5. Key Observations

1. **PCA reduces dimensionality:** Face images can be represented using far fewer values than the original pixel vectors.
2. **Eigenfaces capture major variation:** Principal directions represent patterns of variation across the training faces.
3. **Compression involves a trade-off:** Retaining more components generally improves reconstruction quality but increases representation size.
4. **PCA-based recognition performed well:** The nearest-neighbor method achieved 96.25% accuracy with 98 components.
5. **SVM performed best among the baseline models:** It achieved 98.75% accuracy using the same 98-component representation.
6. **Fewer components may be sufficient:** The tuning experiment reported 98.75% accuracy with only 10 components.
7. **Raw-pixel results require verification:** The raw-pixel nearest-neighbor accuracy was unusually low at 2.50%, so the image-label pairing and implementation should be checked before drawing strong conclusions.
8. **Test-set tuning can be optimistic:** The final selected configuration should be validated independently.

## 6. Conclusion

This project demonstrates how PCA can reduce the dimensionality of face images and support efficient face recognition. The initial PCA nearest-neighbor method achieved 96.25% accuracy, while the baseline RBF-SVM achieved 98.75%.

Experiments with the number of retained components and classifier settings showed that an SVM using 10 PCA components also achieved 98.75% observed test accuracy. This suggests that a compact representation can preserve sufficient information for accurate recognition on this dataset.

Overall, PCA provides a useful representation for face recognition, and SVM offers an effective classifier for the resulting features. Further validation is needed to establish whether the selected compact configuration generalizes reliably to unseen images.

## 7. Tools and Libraries

- Python
- NumPy
- Matplotlib
- Pillow (PIL)
- scikit-learn
- Jupyter Notebook / VS Code

