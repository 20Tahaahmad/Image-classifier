def get_classification_result(results):

    if not results:
        return {
            "label": "Unknown",
            "confidence": 0,
            "all_results": []
        }

    # Find the prediction with the highest confidence
    best_result = max(
        results,
        key=lambda x: x.score
    )

    label = best_result.label
    confidence = round(best_result.score * 100, 2)

    # Prepare all predictions
    all_results = []

    for result in results:

        all_results.append(
            {
                "label": result.label,
                "confidence": round(result.score * 100, 2)
            }
        )

    return {
        "label": label,
        "confidence": confidence,
        "all_results": all_results
    }