from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import difflib
import re
from dotenv import load_dotenv
import google.generativeai as genai
from llm_judge_config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    EVALUATION_PROMPT_TEMPLATE,
    DEFAULT_AGENT_PURPOSE,
    DEFAULT_AGENT_CAPABILITIES
)
from json_parser import (
    parse_agent_output,
    parse_golden_dataset,
    match_records,
    detect_format
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global storage for uploaded files (in production, use proper file storage)
uploaded_files = {}

# Configure Vertex AI if enabled
if os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true':
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    if project_id:
        genai.configure(
            project=project_id,
            location=location
        )

class EvaluationScorecard:
    def __init__(self, expected, actual_obj, adaptive=False):
        self.expected = expected
        self.actual_obj = actual_obj if isinstance(actual_obj, dict) else {"output": str(actual_obj)}
        self.actual_text = self.actual_obj.get("output", "")
        self.adaptive = adaptive
        self.scores = {}
        self.reasons = {}

    def evaluate_all(self):
        self._eval_goal_success()
        self._eval_trajectory()
        self._eval_tool_usage()
        self._eval_robustness()
        self._eval_efficiency()
        return self.scores, self.reasons

    def _eval_goal_success(self):
        # 1. SYNTACTIC MATCH (Fuzzy String)
        def clean(s):
            return re.sub(r'[^\w\s]', '', str(s).lower())

        expected_text = clean(self.expected)
        actual_text = clean(self.actual_text)

        matcher = difflib.SequenceMatcher(None, expected_text, actual_text)
        syntactic_score = matcher.ratio()

        # 2. SEMANTIC MATCH (Keyword Overlap)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were", "be", "has", "have", "had", "it", "this", "that", "of", "with", "by", "as", "from"}

        def get_tokens(text):
            words = text.split()
            return {w for w in words if w not in stopwords}

        exp_tokens = get_tokens(expected_text)
        act_tokens = get_tokens(actual_text)

        semantic_score = 0.0
        if exp_tokens:
            intersection = exp_tokens.intersection(act_tokens)
            union = exp_tokens.union(act_tokens)

            # Jaccard Similarity
            jaccard = len(intersection) / len(union) if union else 0.0

            # Recall (How much of Expected is in Actual)
            recall = len(intersection) / len(exp_tokens) if exp_tokens else 0.0

            # Semantic score leans heavily on Recall
            semantic_score = (recall * 0.7) + (jaccard * 0.3)

            # Boost: If we have high recall of distinct content words
            if recall > 0.60:
                semantic_score = max(semantic_score, 0.85)

        # 3. FINAL SCORE
        if self.adaptive:
            final_score = max(syntactic_score, semantic_score)
        else:
            final_score = 1.0 if expected_text == actual_text else 0.0

        self.scores["Goal Success"] = round(final_score, 2)

        if final_score > 0.7:
             self.reasons["Goal Success"] = f"Match (Syntactic: {syntactic_score:.2f}, Semantic: {semantic_score:.2f})"
        else:
             self.reasons["Goal Success"] = f"Deviation (Syntactic: {syntactic_score:.2f}, Semantic: {semantic_score:.2f})"

    def _eval_trajectory(self):
        """
        Evaluate the agent's execution trajectory.

        A good trajectory means the agent followed a logical path to reach the answer.
        Evidence of trajectory includes:
        - Explicit steps/reasoning
        - Tool usage (shows the agent took action)
        - Multiple steps in reasoning
        """
        steps = self.actual_obj.get("steps", [])
        tools = self.actual_obj.get("tools", [])

        # If we have explicit steps, score based on that
        if steps and len(steps) > 1:
            self.scores["Trajectory"] = 1.0
            self.reasons["Trajectory"] = f"{len(steps)} logic steps"
        elif steps:
            self.scores["Trajectory"] = 0.8
            self.reasons["Trajectory"] = "Single step"
        # If tools were used, that's evidence of a good trajectory
        elif tools and len(tools) > 0:
            self.scores["Trajectory"] = 1.0
            self.reasons["Trajectory"] = f"Used {len(tools)} tool(s) to answer"
        # No steps and no tools - implicit/direct answer
        else:
            self.scores["Trajectory"] = 0.8
            self.reasons["Trajectory"] = "Direct answer (no tools needed)"

    def _eval_tool_usage(self):
        tools = self.actual_obj.get("tools", [])
        if tools:
            self.scores["Tool Usage"] = 1.0
            self.reasons["Tool Usage"] = f"{len(tools)} tools used"
        else:
            self.scores["Tool Usage"] = 1.0
            self.reasons["Tool Usage"] = "No tools needed"

    def _eval_robustness(self):
        if "error" in self.actual_text.lower():
            self.scores["Robustness"] = 0.0
            self.reasons["Robustness"] = "Error in output"
        else:
            self.scores["Robustness"] = 1.0
            self.reasons["Robustness"] = "Stable execution"

    def _eval_efficiency(self):
        latency = self.actual_obj.get("latency", 0)
        if latency < 2.0:
            self.scores["Efficiency"] = 1.0
            self.reasons["Efficiency"] = f"Low latency ({latency}s)"
        else:
            self.scores["Efficiency"] = 0.7
            self.reasons["Efficiency"] = f"High latency ({latency}s)"


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get default configuration from test_config.json"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'adk_eval_starter', 'customer_service_agent', 'test_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and store evaluation files"""
    try:
        file_type = request.form.get('type')  # 'agent' or 'golden'
        file = request.files.get('file')

        if not file or not file_type:
            return jsonify({"error": "Missing file or type"}), 400

        # Read and parse JSON
        content = file.read().decode('utf-8')
        data = json.loads(content)

        # Store in memory
        uploaded_files[file_type] = data

        return jsonify({
            "success": True,
            "type": file_type,
            "message": f"File uploaded successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Run evaluation with provided configuration - supports any JSON format"""
    try:
        config = request.json

        # Get uploaded files
        agent_data = uploaded_files.get('agent')
        golden_data = uploaded_files.get('golden')

        if not agent_data or not golden_data:
            return jsonify({"error": "Missing uploaded files"}), 400

        # Get configuration
        adaptive_mode = config.get('adaptive', True)
        traj_thresh = config.get('trajectory_threshold', 0.8)
        match_thresh = config.get('match_threshold', 0.7)

        # Use smart parsers to handle any format
        agent_records = parse_agent_output(agent_data)
        test_cases = parse_golden_dataset(golden_data)

        if not agent_records:
            return jsonify({"error": "No valid agent records found in uploaded file"}), 400
        if not test_cases:
            return jsonify({"error": "No valid test cases found in golden dataset"}), 400

        # Match records by ID or position
        matched_pairs = match_records(agent_records, test_cases)

        if not matched_pairs:
            return jsonify({"error": "Could not match agent records with test cases"}), 400

        results = []
        pass_count = 0

        for agent_rec, test_case in matched_pairs:
            # Use normalized data
            test_id = test_case["id"]
            prompt = test_case["prompt"]
            expected = test_case["expected"]
            expected_tools = test_case["expected_tools"]

            actual_output = agent_rec["output"]
            actual_tools = agent_rec["tools"]
            actual_steps = agent_rec["steps"]

            # Evaluate using the scorecard
            evaluator = EvaluationScorecard(expected, agent_rec, adaptive=adaptive_mode)
            scores, reasons = evaluator.evaluate_all()

            # Dynamic threshold logic
            traj_pass = scores["Trajectory"] >= traj_thresh
            match_pass = scores["Goal Success"] >= match_thresh
            is_pass = traj_pass and match_pass

            if is_pass:
                pass_count += 1

            # Update reasons based on thresholds
            if not traj_pass:
                reasons["Trajectory"] = f"Score {scores['Trajectory']} below threshold {traj_thresh}"
            if not match_pass:
                reasons["Goal Success"] = f"Score {scores['Goal Success']} below threshold {match_thresh}"

            results.append({
                "id": test_id,
                "input": prompt,
                "expected": expected,
                "actual": actual_output,
                "scores": scores,
                "reasons": reasons,
                "status": "PASSED" if is_pass else "FAILED",
                "tools": actual_tools,
                "steps": actual_steps,
                "expected_tools": expected_tools
            })

        length = len(results)
        fail_count = length - pass_count
        overall_status = "PASSED" if fail_count == 0 else "FAILED"

        # Calculate averages
        avg_traj = sum(r['scores']['Trajectory'] for r in results) / length if length > 0 else 0
        avg_match = sum(r['scores']['Goal Success'] for r in results) / length if length > 0 else 0

        # Detect formats for debugging
        agent_format = detect_format(agent_data)
        golden_format = detect_format(golden_data)

        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total": length,
                "passed": pass_count,
                "failed": fail_count,
                "overall_status": overall_status,
                "avg_trajectory": round(avg_traj, 2),
                "avg_match": round(avg_match, 2),
                "agent_format": agent_format,
                "golden_format": golden_format
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def evaluate_with_llm(agent_output, model_name, config, expected_response="Not specified", expected_tools="Not specified"):
    """
    Evaluate agent output using LLM as Judge

    Args:
        agent_output: Dict containing agent response data
        model_name: Gemini model to use for evaluation
        config: Configuration dict with thresholds

    Returns:
        Dict with evaluation results matching ADK format
    """
    try:
        # Extract data from agent output
        user_query = agent_output.get('input', agent_output.get('query', 'Unknown query'))
        agent_response = agent_output.get('output', '')
        tool_calls = agent_output.get('tools', [])

        # Format tool calls for prompt
        tool_calls_str = json.dumps(tool_calls, indent=2) if tool_calls else "No tools used"

        # Build evaluation prompt
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            agent_purpose=DEFAULT_AGENT_PURPOSE,
            agent_capabilities=DEFAULT_AGENT_CAPABILITIES,
            user_query=user_query,
            agent_response=agent_response,
            expected_response=expected_response,
            tool_calls=tool_calls_str,
            expected_tools=expected_tools,
            trajectory_threshold=config.get('trajectory_threshold', 0.8),
            match_threshold=config.get('match_threshold', 0.5)
        )

        # Call Gemini model
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent evaluation
                top_p=0.95,
                max_output_tokens=1024,
            )
        )

        # Extract and parse JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()

        # Parse JSON
        eval_result = json.loads(response_text)

        # Validate and normalize scores
        response_match_score = float(eval_result.get('response_match_score', 0.0))
        tool_trajectory_score = float(eval_result.get('tool_trajectory_score', 1.0))
        overall_score = float(eval_result.get('overall_score', 0.0))

        # Ensure scores are between 0 and 1
        response_match_score = max(0.0, min(1.0, response_match_score))
        tool_trajectory_score = max(0.0, min(1.0, tool_trajectory_score))
        overall_score = max(0.0, min(1.0, overall_score))

        # Determine pass/fail based on thresholds
        traj_thresh = config.get('trajectory_threshold', 0.8)
        match_thresh = config.get('match_threshold', 0.5)

        traj_pass = tool_trajectory_score >= traj_thresh
        match_pass = response_match_score >= match_thresh
        is_pass = traj_pass and match_pass

        # Build result in ADK format
        return {
            'response_match_score': round(response_match_score, 2),
            'tool_trajectory_score': round(tool_trajectory_score, 2),
            'overall_score': round(overall_score, 2),
            'status': 'PASSED' if is_pass else 'FAILED',
            'response_justification': eval_result.get('response_justification', ''),
            'trajectory_justification': eval_result.get('trajectory_justification', ''),
            'overall_justification': eval_result.get('overall_justification', ''),
            'traj_pass': traj_pass,
            'match_pass': match_pass
        }

    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {str(e)}. Response: {response_text[:200]}")
    except Exception as e:
        raise ValueError(f"LLM evaluation failed: {str(e)}")


@app.route('/api/evaluate-llm', methods=['POST'])
def evaluate_llm():
    """
    Run LLM-based evaluation - supports any JSON format and optional golden dataset
    """
    try:
        data = request.json
        config = data.get('config', {})
        model_name = data.get('model', DEFAULT_MODEL)

        # Validate model
        if model_name not in AVAILABLE_MODELS:
            return jsonify({
                "error": f"Invalid model. Available models: {', '.join(AVAILABLE_MODELS)}"
            }), 400

        # Get uploaded files
        agent_data = uploaded_files.get('agent')
        golden_data = uploaded_files.get('golden')

        if not agent_data:
            return jsonify({"error": "Missing agent output file"}), 400

        # Use smart parser to handle any format
        agent_records = parse_agent_output(agent_data)

        if not agent_records:
            return jsonify({"error": "No valid agent records found in uploaded file"}), 400

        # Match records if golden data is available
        if golden_data:
            test_cases = parse_golden_dataset(golden_data)
            matched_pairs = match_records(agent_records, test_cases)
        else:
            # Fallback: Treat each agent record as its own test case
            matched_pairs = [(rec, None) for rec in agent_records]

        results = []
        pass_count = 0

        for agent_rec, test_case in matched_pairs:
            test_id = test_case["id"] if test_case else agent_rec["id"]

            # Use info from test case if available, else from agent record
            prompt = test_case["prompt"] if test_case else agent_rec.get("input", "N/A")
            expected = test_case["expected"] if test_case else agent_rec.get("expected", "LLM-evaluated (no golden dataset)")
            expected_tools = test_case["expected_tools"] if test_case else "Not specified"

            # Evaluate with LLM using matched data
            eval_result = evaluate_with_llm(
                agent_rec,
                model_name,
                config,
                expected_response=expected,
                expected_tools=expected_tools
            )

            if eval_result['status'] == 'PASSED':
                pass_count += 1

            # Build scores dict matching golden dataset format
            scores = {
                "Goal Success": eval_result['response_match_score'],
                "Trajectory": eval_result['tool_trajectory_score'],
                "Tool Usage": 1.0,
                "Robustness": 1.0,
                "Efficiency": 1.0
            }

            reasons = {
                "Goal Success": eval_result['response_justification'],
                "Trajectory": eval_result['trajectory_justification'],
                "Tool Usage": "Evaluated by LLM",
                "Robustness": "Evaluated by LLM",
                "Efficiency": "Evaluated by LLM"
            }

            results.append({
                "id": test_id,
                "input": prompt,
                "expected": expected,
                "actual": agent_rec["output"],
                "scores": scores,
                "reasons": reasons,
                "status": eval_result['status'],
                "tools": agent_rec["tools"],
                "steps": agent_rec["steps"],
                "expected_tools": expected_tools,
                "llm_justification": eval_result['overall_justification']
            })

        length = len(results)
        fail_count = length - pass_count
        overall_status = "PASSED" if fail_count == 0 else "FAILED"

        # Calculate averages
        avg_traj = sum(r['scores']['Trajectory'] for r in results) / length if length > 0 else 0
        avg_match = sum(r['scores']['Goal Success'] for r in results) / length if length > 0 else 0

        # Detect format for debugging
        agent_format = detect_format(agent_data)

        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total": length,
                "passed": pass_count,
                "failed": fail_count,
                "overall_status": overall_status,
                "avg_trajectory": round(avg_traj, 2),
                "avg_match": round(avg_match, 2),
                "evaluation_method": "LLM as Judge",
                "model_used": model_name,
                "agent_format": agent_format
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500




@app.route('/api/inspect', methods=['POST'])
def inspect_file():
    """Debug endpoint to show how a file will be parsed"""
    try:
        file_type = request.json.get('type')  # 'agent' or 'golden'

        # Get uploaded file
        data = uploaded_files.get(file_type)

        if not data:
            return jsonify({"error": f"No {file_type} file uploaded"}), 400

        # Parse based on type
        if file_type == 'agent':
            parsed = parse_agent_output(data)
        else:
            parsed = parse_golden_dataset(data)

        detected_format = detect_format(data)

        return jsonify({
            "detected_format": detected_format,
            "parsed_count": len(parsed),
            "sample": parsed[0] if parsed else None,
            "all_ids": [p["id"] for p in parsed]
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available Gemini models for LLM evaluation"""
    return jsonify({
        "models": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
