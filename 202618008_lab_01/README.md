
- Title : Data Scraping and Preprocessing using Python and Scrapy
- Name : Nandini Pipaliya
- ID : 202618008

## Overview

This project demonstrates a complete data analysis workflow using **Scrapy**, **Pandas**, and **Matplotlib**. The data is scraped from the practice website **Books to Scrape** and then cleaned, preprocessed, analyzed, and visualized in a Jupyter Notebook.

The project is divided into three main tasks:

* **Task 1:** Data Scraping
* **Task 2:** Data Preprocessing
* **Task 3:** Visualization and Analysis

---


## Dataset

**Source:** Books to Scrape

Website: https://books.toscrape.com/

The spider scrapes at least **100 books** from the first five catalog pages.

### Extracted Fields

* Title
* Category
* Price
* Rating
* Availability
* Product Description
* UPC
* Number of Reviews
* Product URL

---

## Task 1 – Data Scraping

A Scrapy spider was developed to:

* Crawl multiple catalog pages
* Visit each individual book page
* Extract the required book information
* Export the scraped data to CSV format

---

## Task 2 – Data Preprocessing

The following preprocessing steps were performed:

* Removed leading and trailing whitespace
* Standardized inconsistent text formatting
* Removed duplicate books using UPC
* Handled missing product descriptions
* Converted price from text to numeric values
* Converted ratings from text (One–Five) to integers (1–5)
* Extracted available stock count from the availability field

### Feature Engineering

Additional features created include:

* **Affordability Score**
* **Value Score**
* **Recommended** (based on rating and price)

---

## Task 3 – Visualization and Analysis

The notebook includes:

* Price Distribution
* Rating Distribution
* Average Price by Category
* Price vs Rating Scatter Plot
* Category vs Stock Analysis
* Word Cloud generated from book descriptions

Additional analyses include:

* Summary statistics
* Category-wise analysis
* Highly rated books
* Stock patterns
* Missing value analysis
* Identification of unusual values

---

## Technologies Used

* Python
* Scrapy
* Pandas
* NumPy
* Matplotlib
* WordCloud

---




