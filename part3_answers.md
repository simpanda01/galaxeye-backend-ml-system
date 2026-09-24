# Part 3 — Problem Solving

## 1. The classifier is wrong about 30% of the time. What do you do?

First, I would not decide whether the model is useful from the 70% overall accuracy alone.

I would evaluate it on a representative holdout set and look at:

- Overall accuracy
- Per-class precision and recall
- Confusion matrix
- Confidence distribution
- Performance on different image conditions

For this take-home, the model achieved 80% accuracy on the 210-image evaluation set, but the class-level results were uneven. Highway and Industrial were 96.67%, while River was 46.67% and SeaLake was 60%.

That tells me the errors are concentrated in particular classes rather than being uniformly distributed.

I would investigate those errors first. For example, I would inspect River/SeaLake confusion, check whether the training data is balanced and representative, and review incorrectly classified images.

Whether 70% accuracy is "good enough" depends on the use case. If the predictions are only used to assist an analyst, 70% may still be useful. If an incorrect classification can directly trigger an important operational decision, the required performance would be much higher.

I would therefore define the acceptable error rate based on the downstream consequence of an incorrect prediction.

I would also avoid silently trusting low-confidence predictions. In the current implementation, predictions below 0.60 confidence are stored as `uncertain` for review.

---

## 2. The service runs offline with no one watching it. A month after deployment, how would you know it is still working?

I would build observability into the system rather than depending on someone manually checking it.

I would track:

- Number of tiles processed
- Number of successful predictions
- Number of failed predictions
- Processing latency
- Confidence distribution
- Percentage of uncertain predictions
- Prediction distribution across classes
- Model version
- Application errors

I would also periodically run a small known test set through the deployed model.

The expected predictions for this test set would be stored as a deployment smoke test. If the results change unexpectedly, that could indicate a model, dependency or hardware problem.

For offline hardware, logs should be persisted locally and rotated so that the disk does not fill up.

If the environment allows scheduled local jobs, I would run periodic health checks and generate a local diagnostic report for the next operator who accesses the machine.

I would also monitor changes in the input data distribution. A system can technically continue running while the incoming imagery changes enough that model performance degrades.

---

## 3. Tiles are coming in fine, but stored results look wrong. How do you find the cause?

I would debug the pipeline from the input towards storage.

### Step 1 — Verify the input

Take one problematic tile and confirm:

- The file is valid.
- The correct image is being uploaded.
- The image dimensions and channels are correct.
- The filename matches the expected tile.

### Step 2 — Test the model independently

Run the same image directly through the classifier without the API or database.

I would compare:

- Prediction
- Confidence
- Model version

If the standalone prediction is already wrong, the problem is likely in the model or preprocessing.

### Step 3 — Verify preprocessing

Check that inference preprocessing matches training preprocessing:

- RGB conversion
- Resize to 224x224
- Tensor conversion
- Normalization

A mismatch here could produce systematically incorrect predictions.

### Step 4 — Verify the API response

Check that the prediction returned by the model is the same value returned by `/predict`.

This separates model problems from API/data-handling problems.

### Step 5 — Inspect the database

Query the SQLite row directly and compare it with the API response.

I would verify:

- filename
- prediction
- confidence
- status
- model version
- timestamp

### Step 6 — Check the model version

Make sure the model loaded by the service is the expected model version.

This is one reason I store `model_version` with every prediction.

### Step 7 — Check recent changes

Review recent changes to:

- model file
- preprocessing
- API code
- database schema
- dependencies

This gives me a controlled path from input → inference → API → database instead of changing multiple components at once.

---

## 4. What is the weakest part of your design, and what would break it first?

The weakest part is the simplicity of the current inference and storage architecture.

The current system is intentionally designed as a small offline thin slice using a local ResNet18 model and SQLite.

SQLite would become a limitation if the system needed high write concurrency, multiple independent services or very large-scale querying.

The model is also the biggest functional risk. The evaluation results show that performance is not uniform across classes, particularly for River and SeaLake. A model can therefore be technically healthy while still producing results that are not useful for a particular class.

The current confidence threshold is another simplification. A raw softmax confidence value should not automatically be treated as a calibrated probability. In a production system I would evaluate calibration and choose thresholds based on validation data and the cost of errors.

