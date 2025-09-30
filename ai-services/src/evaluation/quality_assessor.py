
import json
from typing import Any, Dict

from ..models.llm_client import LLMClient
from ..utils.prompts import get_quality_assessment_prompt

class QualityAssessor:
    """
    Generates a quality assessment for a given question based on source context.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the QualityAssessor.

        Args:
            llm_client: An instance of LLMClient to interact with the language model.
        """
        self.llm_client = llm_client

    def assess_question(self, question_data: Dict[str, Any], source_context: str) -> Dict[str, Any]:
        """
        Assesses the quality of a single question.

        Args:
            question_data: A dictionary containing the generated question details.
            source_context: The source text context used to generate the question.

        Returns:
            A dictionary containing the assessment results.
        """
        prompt = get_quality_assessment_prompt()
        
        # We need to serialize the question data to a string to include it in the prompt.
        question_json_str = json.dumps(question_data, ensure_ascii=False, indent=2)
        
        formatted_prompt = prompt.format(
            source_context=source_context,
            question_json=question_json_str
        )
        
        assessment_result = self.llm_client.generate_structured_response(
            prompt=formatted_prompt,
            response_format="json"
        )
            
        return assessment_result

