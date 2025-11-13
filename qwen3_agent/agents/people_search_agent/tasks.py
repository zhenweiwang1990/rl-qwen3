"""People search agent task definitions."""

from typing import Any, List
from qwen3_agent.core.framework.task import BaseTask
from pydantic import Field


class PeopleSearchTask(BaseTask):
    """People search task.
    
    Represents a query about finding people that the agent needs to answer by
    searching and reading profiles.
    
    Attributes:
        id: Unique identifier (inherited)
        query: User's search query for finding people
        expected_profiles: List of linkedin_handles that should be in the answer
        batch: Batch identifier for the task
    """
    
    query: str = Field(description="User's search query for finding people")
    expected_profiles: List[str] = Field(description="List of linkedin_handles that should match the query")
    batch: str = Field(default="", description="Batch identifier")
    
    def get_query(self) -> str:
        """Get the user's search query.
        
        Returns:
            The query string
        """
        return self.query
    
    def get_ground_truth(self) -> Any:
        """Get ground truth answer.
        
        Returns:
            Dictionary with 'expected_profiles'
        """
        return {
            "expected_profiles": self.expected_profiles,
        }
    
    def get_context(self) -> dict[str, Any]:
        """Get additional context for the task.
        
        Returns:
            Dictionary with batch and other metadata
        """
        return {
            **self.metadata,
            "batch": self.batch,
        }

