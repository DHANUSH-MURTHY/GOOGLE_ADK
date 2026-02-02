import json
import os
import unittest
from unittest.mock import MagicMock, patch
from app import app, uploaded_files

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        uploaded_files.clear()

    @patch('google.generativeai.GenerativeModel')
    def test_evaluate_llm_with_golden(self, mock_model_class):
        # Mock Gemini response
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "response_match_score": 0.9,
            "tool_trajectory_score": 1.0,
            "overall_score": 0.95,
            "status": "PASSED",
            "response_justification": "Good match",
            "trajectory_justification": "Perfect",
            "overall_justification": "Excellent"
        })
        mock_model.generate_content.return_value = mock_response

        # Mock uploaded files
        agent_output = [
            {
                "id": "1",
                "output": "The capital of France is Paris.",
                "tools": [],
                "steps": []
            }
        ]
        golden_dataset = [
            {
                "id": "1",
                "prompt": "What is the capital of France?",
                "expected": "Paris is the capital of France.",
                "expected_tools": []
            }
        ]

        uploaded_files['agent'] = agent_output
        uploaded_files['golden'] = golden_dataset

        response = self.client.post('/api/evaluate-llm', json={
            "model": "gemini-1.5-flash",
            "config": {"trajectory_threshold": 0.8, "match_threshold": 0.7}
        })

        data = response.get_json()
        self.assertTrue(data['success'])
        result = data['results'][0]
        self.assertEqual(result['input'], "What is the capital of France?")
        self.assertEqual(result['expected'], "Paris is the capital of France.")
        self.assertEqual(result['actual'], "The capital of France is Paris.")
        self.assertEqual(result['status'], "PASSED")

        # Verify prompt contained the expected response
        args, kwargs = mock_model.generate_content.call_args
        prompt = args[0]
        self.assertIn("Expected Response (if known): Paris is the capital of France.", prompt)

    @patch('google.generativeai.GenerativeModel')
    def test_evaluate_llm_agnostic_single_file(self, mock_model_class):
        # Test with a single file containing everything (agnostic format)
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "response_match_score": 0.8,
            "tool_trajectory_score": 1.0,
            "overall_score": 0.8,
            "status": "PASSED",
            "response_justification": "Ok",
            "trajectory_justification": "Ok",
            "overall_justification": "Ok"
        })
        mock_model.generate_content.return_value = mock_response

        # Uploaded as 'agent' but contains 'prompt' and 'expected'
        agent_output = [
            {
                "id": "case_101",
                "prompt": "Tell me a joke",
                "output": "Why did the chicken cross the road?",
                "expected": "To get to the other side.",
                "tools": []
            }
        ]
        uploaded_files['agent'] = agent_output

        response = self.client.post('/api/evaluate-llm', json={
            "model": "gemini-1.5-flash",
            "config": {}
        })

        data = response.get_json()
        self.assertTrue(data['success'])
        result = data['results'][0]
        self.assertEqual(result['id'], "case_101")
        self.assertEqual(result['input'], "Tell me a joke")
        self.assertEqual(result['expected'], "To get to the other side.")

        # Verify prompt contained the expected response from the agent file
        args, kwargs = mock_model.generate_content.call_args
        prompt = args[0]
        self.assertIn("Expected Response (if known): To get to the other side.", prompt)

    def test_standard_evaluate_agnostic(self):
        # Test standard evaluation with agnostic format
        agent_output = [
            {
                "id": "case_202",
                "prompt": "Hello",
                "output": "Hi there!",
                "expected": "Hi there!",
                "tools": [],
                "steps": []
            }
        ]
        uploaded_files['agent'] = agent_output
        uploaded_files['golden'] = agent_output # Use same file as golden

        response = self.client.post('/api/evaluate', json={
            "adaptive": True
        })

        data = response.get_json()
        self.assertTrue(data['success'])
        result = data['results'][0]
        self.assertEqual(result['id'], "case_202")
        self.assertEqual(result['status'], "PASSED")
        self.assertEqual(result['input'], "Hello")
        self.assertEqual(result['expected'], "Hi there!")

if __name__ == "__main__":
    unittest.main()
