from ai_engine.classifier import get_classification_result


results = [
    type("Result", (), {
        "label": "dog",
        "score": 0.90
    })(),

    type("Result", (), {
        "label": "cat",
        "score": 0.07
    })(),

    type("Result", (), {
        "label": "fox",
        "score": 0.03
    })()
]


result = get_classification_result(results)


print("\n===== PROCESSED RESULT =====\n")

print("Prediction:", result["label"])
print("Confidence:", result["confidence"], "%")

print("\nOther Predictions:")

for item in result["all_results"]:

    print(
        f"{item['label']} → "
        f"{item['confidence']}%"
    )