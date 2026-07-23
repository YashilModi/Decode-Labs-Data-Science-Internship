# DecodeLabs Internship - Task 1 and Task 2 Video Script

Target length: 10 minutes. This version presents the slide deck and cuts to the code and outputs along the way.

How to read this script:
- `[SLIDE n]` means advance the PowerPoint to that slide.
- `[CODE: ...]` means switch to VS Code and show that part of the code.
- `[OUTPUT: ...]` means show the terminal, a saved chart, or a file.
- Anything in square brackets is a screen direction, not something you say.
- `<< >>` is a number to read live off your own screen.

---

## INTRO - 0:00 to 0:50

`[SLIDE 1: From Raw Data to Fraud Detection]`

Hi, I'm Yashil. This is a walkthrough of the first two tasks of my DecodeLabs Data Science internship.

`[SLIDE 2: Two Tasks, One Data Science Workflow]`

There are two tasks here. Task 1 takes a messy retail orders export and cleans it up, then builds new features from it. Task 2 trains and compares two machine learning models to catch credit card fraud.

I'll present each task from these slides, and cut over to the code and the outputs at the key moments. Let's start with Task 1.

---

## TASK 1 - 0:50 to 4:30

`[SLIDE 3: Task 1 - Data Cleaning & Feature Engineering]`

Task 1 is one Python script that runs top to bottom. The only library it needs is pandas.

`[SLIDE 4: Dataset for Data Analytics.xlsx]`

Here's the raw data. Twelve hundred orders, fourteen columns, covering the start of 2023 to the middle of 2025. You can see the fourteen raw columns listed here, things like OrderID, Date, Product, Quantity and UnitPrice.

### Cleaning the missing values

`[SLIDE 5: Missing Values & Outlier Detection]`

First job is cleaning. Two things to check: missing values, and outliers.

`[CODE: Task 1.py - the fillna line in section 2]`

For the missing values, I count what's blank in each column first. Only one column has anything missing: CouponCode. A blank coupon code doesn't mean the data is broken, it just means the customer didn't use a coupon. So instead of deleting those rows, I fill the blanks with the text "No Coupon". That keeps all twelve hundred orders.

`[OUTPUT: Task 1/run_log.txt - the missing values block]`

You can see it here. 309 blanks in CouponCode, and after the fill, zero missing values in the whole dataset.

### Checking for outliers, two ways

`[CODE: Task 1.py - the outlier loop in section 3]`

For outliers I loop over the three numeric columns. For each one I use the IQR rule, which flags anything that sits far outside the middle of the data. And then, as a second opinion, I also check the Z-score, which flags anything more than three standard deviations from the mean.

`[OUTPUT: Task 1/run_log.txt - the three column blocks]`

Both methods agree. Zero outliers in Quantity, zero in UnitPrice, zero in ItemsInCart. So the data was already clean, and I've now confirmed that two different ways.

### Building new features

`[SLIDE 6: New Features & Final Output]`

Now the feature engineering. This is where I add six brand new columns.

`[CODE: Task 1.py - section 4]`

From the order date I pull out the month, the weekday name, and a weekend flag. HasCoupon marks whether a coupon was used. AvgItemValue is the order total divided by the number of items. And for high value orders, I take the 75th percentile of the total price as the cutoff and flag anything above it.

`[OUTPUT: Task 1/run_log.txt - cutoff, high value count, coupon count]`

The cutoff comes out at 1578.48. That flags 300 orders as high value, which is exactly the top quarter of twelve hundred. And 891 orders used a coupon. Add that to the 309 blanks we filled earlier and it comes right back to twelve hundred.

`[SLIDE 7: The Six Engineered Features]`

Here's the summary of all six features on one slide: what each one is built from, and what it tells us. So the dataset goes from fourteen columns in, to twenty columns out, cleaned and ready for a model.

---

## TASK 2 - 4:30 to 9:10

`[SLIDE 8: Task 2 - Credit Card Fraud Detection]`

Now Task 2, fraud detection.

`[SLIDE 9: creditcard.csv - A Highly Imbalanced Dataset]`

This dataset is much bigger. 284,807 transactions and 31 columns. But here's the catch. Only 492 of those transactions are fraud. That's 0.173 percent.

Look at this picture: one gold dot in a sea of grey. On average, just one fraud in every 579 transactions.

This is the whole challenge of the task. If I built a model that simply said "not fraud" every single time, it would be right 99.8 percent of the time and catch zero fraud. So accuracy is useless here, and I'll come back to which numbers I use instead.

### The scoring helper

`[CODE: Task 2.py - the evaluate_model function]`

Before the main code there's one small helper, evaluate_model. It takes a trained model, predicts on the test set, and works out precision, recall, F1, ROC-AUC, and average precision, which is the one that matters most on imbalanced data. I wrote it as a function because I run the exact same checks on both of my models.

### The leakage-safe pipeline

`[SLIDE 10: A Leakage-Safe Modelling Pipeline]`

This slide is the heart of Task 2. Five steps: split the data, apply SMOTE, fit a model, tune it, then evaluate on data it has never seen.

`[CODE: Task 2.py - the two pipelines in section 4]`

The important idea is that SMOTE lives inside the pipeline. SMOTE creates synthetic fraud examples so the model trains on a balanced picture. Because it sits inside the pipeline, it only ever runs on the training part of each fold, and never touches the test data. If I had balanced the whole dataset up front, information would leak from the test set and my scores would look better than they really are.

I tune both models with grid search and five-fold cross validation. For logistic regression it settles on a C of 0.01, and for the random forest, a max depth of 10.

### Results

`[SLIDE 11: Logistic Regression vs. Random Forest]`

Here are the results on the held-out test set.

`[OUTPUT: Task 2/run_log.txt - the metrics block for both models]`

Reading them live: logistic regression gets precision `<<read>>`, recall `<<read>>`, and PR-AUC `<<read>>`. The random forest gets precision `<<read>>`, recall `<<read>>`, and PR-AUC `<<read>>`.

Quick reminder of what these mean. Recall is the share of real frauds we actually caught. Precision is, out of everything we flagged, how much really was fraud. On the chart you can see the random forest wins clearly on precision, F1 and PR-AUC, while logistic regression catches a touch more fraud on recall.

### Why PR-AUC, not accuracy or ROC-AUC

`[SLIDE 12: Why We Judge on PR-AUC, Not Accuracy]`

This is the number that decides it. Accuracy is a trap, as we said, 99.8 percent and zero fraud caught. ROC-AUC looks strong for both models, around 0.97 and 0.98, but it flatters them because the huge pile of legitimate transactions dominates it.

PR-AUC only looks at the fraud class, and that's where the real gap shows: 0.71 for logistic regression versus 0.82 for the random forest. So I tune on one metric, but I judge the models on PR-AUC and the confusion matrix, because those reflect the real cost of each mistake.

`[OUTPUT: Task 2/outputs/precision_recall_curves.png]`

Here's the precision-recall curve that sits behind that number. The higher curve is the random forest.

### What the model learned

`[SLIDE 13: What the Random Forest Learned]`

I also pulled out which features the random forest leaned on most. The top signals are V14, V10 and V4. These are anonymised PCA features, so we can't name the real-world behaviour behind them, but the useful thing is that the model relies on a small handful of them, not on the time or the amount of the transaction.

### The two mistakes

`[SLIDE 14: Reading the Confusion Matrices]`

Finally, the confusion matrices, which show the two kinds of mistake side by side. There are 98 frauds in the test set.

Logistic regression catches 90 of them, but it raises 1,448 false alarms. The random forest catches 86, four fewer, but with only 69 false alarms. So the random forest misses a little more fraud, and in return it creates roughly twenty times fewer false alarms. That trade is why I'd pick it.

---

## CLOSE - 9:10 to 10:00

`[SLIDE 15: What This Project Demonstrates]`

So that's the two tasks.

Task 1 was about cleaning deliberately, keeping every row instead of dropping data, and confirming the outlier check two different ways. Task 2 was about a very imbalanced problem: using a pipeline and SMOTE carefully so nothing leaks, and then judging the models on PR-AUC and the confusion matrix instead of accuracy.

Thanks for watching.
