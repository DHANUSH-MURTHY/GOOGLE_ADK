import sys

TOTAL_SCORE = (
    acc_tb +
    acc_ap +
    acc_root
) / 3

print(f"\n🏁 FINAL SYSTEM SCORE: {TOTAL_SCORE * 100:.2f}%")

if TOTAL_SCORE < 0.6:
    print("❌ Evaluation failed")
    sys.exit(1)   # CI fails
else:
    print("✅ Evaluation passed")
    sys.exit(0)   # CI passes

from agent import (
    travel_brainstormer,
    attractions_planner,
    root_agent
)

from evaluation.test_cases import (
    travel_brainstormer_tests,
    attractions_planner_tests,
    root_agent_tests
)

from evaluation.evaluator import evaluate_agent, evaluate_routing


print("\n🔹 Evaluating Travel Brainstormer")
acc_tb, res_tb = evaluate_agent(travel_brainstormer, travel_brainstormer_tests)
print(f"Accuracy: {acc_tb * 100:.2f}%")

print("\n🔹 Evaluating Attractions Planner")
acc_ap, res_ap = evaluate_agent(attractions_planner, attractions_planner_tests)
print(f"Accuracy: {acc_ap * 100:.2f}%")

print("\n🔹 Evaluating Root Agent Routing")
acc_root, res_root = evaluate_routing(root_agent, root_agent_tests)
print(f"Routing Accuracy: {acc_root * 100:.2f}%")
