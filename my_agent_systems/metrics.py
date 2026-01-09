def keyword_match_score(output: str, expected_keywords: list[str]) -> float:
    output = output.lower()
    hits = sum(1 for k in expected_keywords if k.lower() in output)
    return hits / len(expected_keywords)


def pass_fail(score: float, threshold: float = 0.4) -> bool:
    return score >= threshold
