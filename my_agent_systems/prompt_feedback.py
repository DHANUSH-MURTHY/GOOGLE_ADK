def analyze_failures(results):
    weak_cases = []

    for r in results:
        if not r["passed"]:
            weak_cases.append({
                "input": r["input"],
                "output": r["output"],
                "issue": "Missing key intent or keywords"
            })

    return weak_cases
