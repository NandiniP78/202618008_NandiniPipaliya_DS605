## Lab Assignment 2: Vectorized Programming with NumPy and Data Wrangling with Pandas 

- **Student Name:** Nandini Sanjaybhai Pipaliya
- **Student ID:** 202618008


- # Titanic Survival Analysis

## Project Overview

Exploratory Data Analysis (EDA) of the **Titanic train dataset** to identify factors influencing passenger survival using Python, Pandas, Matplotlib, and Seaborn.

## Dataset

* **891 passengers**
* **12 features**
* Target variable: `Survived`

  * `0` = Did not survive
  * `1` = Survived

### Key Features

`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, `Embarked`

## Key Observations

* **38.4%** of passengers survived, while **61.6%** did not.
* **74.2% of females survived** compared with only **18.9% of males**.
* Survival rate by class:

  * **1st Class:** 63.0%
  * **2nd Class:** 47.3%
  * **3rd Class:** 24.2%
* **1st-class females had the highest survival rate (~96.8%)**, while **3rd-class males had the lowest (~13.5%)**.
* Survivors paid a higher average fare (**~£48.40**) than non-survivors (**~£22.12**).
* Passengers aged **15 or below had ~59% survival**, higher than the overall survival rate.
* `Cabin` has **77.1% missing values**, while `Age` has **19.9% missing values**.

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

## Conclusion

**Gender and passenger class were the strongest factors associated with survival**, with females and higher-class passengers having significantly better survival rates.



