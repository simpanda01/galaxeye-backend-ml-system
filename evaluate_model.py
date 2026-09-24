import os
import pandas as pd
from PIL import Image
from collections import defaultdict

from app.model import SatelliteClassifier


EVAL_DIR = "data/eval_set"
LABELS_FILE = "data/eval_labels.csv"


classifier = SatelliteClassifier()

labels = pd.read_csv(LABELS_FILE)

correct = 0
total = 0

class_correct = defaultdict(int)
class_total = defaultdict(int)


for _, row in labels.iterrows():

    filename = row["filename"]
    true_label = row["true_label"]

    image_path = os.path.join(EVAL_DIR, filename)

    if not os.path.exists(image_path):
        print(f"Missing image: {filename}")
        continue

    image = Image.open(image_path)

    result = classifier.predict(image)

    prediction = result["prediction"]

    total += 1
    class_total[true_label] += 1

    if prediction == true_label:
        correct += 1
        class_correct[true_label] += 1

    print(
        f"{filename}: "
        f"actual={true_label}, "
        f"predicted={prediction}, "
        f"confidence={result['confidence']}"
    )


overall_accuracy = correct / total if total else 0


print()
print("=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)

print(f"Correct: {correct}/{total}")
print(f"Overall Accuracy: {overall_accuracy:.2%}")

print()
print("Per-Class Accuracy:")
print("-" * 50)

for class_name in sorted(class_total.keys()):

    class_accuracy = (
        class_correct[class_name] / class_total[class_name]
        if class_total[class_name]
        else 0
    )

    print(
        f"{class_name:<15} "
        f"{class_correct[class_name]}/{class_total[class_name]} "
        f"({class_accuracy:.2%})"
    )

print("=" * 50)