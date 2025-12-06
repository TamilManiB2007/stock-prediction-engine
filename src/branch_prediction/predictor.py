def compare_predictors():
    """Compare different branch prediction algorithms"""
    predictors = {
        "Static Always Taken": {"accuracy": 50, "type": "Static"},
        "1-bit Dynamic": {"accuracy": 78, "type": "Dynamic"},
        "2-bit Saturating": {"accuracy": 85, "type": "Dynamic"}
    }
    
    results = {}
    for name, data in predictors.items():
        results[name] = {
            "accuracy": data["accuracy"],
            "type": data["type"],
            "description": f"{name} predictor with {data['accuracy']}% accuracy"
        }
    
    return results
