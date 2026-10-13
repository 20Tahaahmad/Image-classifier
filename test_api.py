from huggingface.api_client import classify_image


image_path = "test_image.jpg"


results = classify_image(image_path)


print("\n===== IMAGE CLASSIFICATION RESULT =====\n")


for result in results:

    print(
        f"Label: {result.label} | "
        f"Confidence: {result.score * 100:.2f}%"
    )