from evaluation.metrics import keyword_match_score, pass_fail


def evaluate_agent(agent, test_cases, threshold=0.4):
    results = []
    passed = 0

    for test in test_cases:
        response = agent.run(test["input"])
        output = response.output_text

        score = keyword_match_score(output, test["expected_keywords"])
        is_pass = pass_fail(score, threshold)

        if is_pass:
            passed += 1

        results.append({
            "input": test["input"],
            "output": output,
            "score": round(score, 2),
            "passed": is_pass
        })

    accuracy = passed / len(test_cases)
    return accuracy, results

def evaluate_routing(root_agent, routing_tests):
    correct = 0
    results = []

    for test in routing_tests:
        response = root_agent.run(test["input"])

        chosen_agent = response.metadata.get("agent_name")

        is_correct = chosen_agent == test["expected_agent"]

        if is_correct:
            correct += 1

        results.append({
            "input": test["input"],
            "expected_agent": test["expected_agent"],
            "chosen_agent": chosen_agent,
            "passed": is_correct
        })

    accuracy = correct / len(routing_tests)
    return accuracy, results
