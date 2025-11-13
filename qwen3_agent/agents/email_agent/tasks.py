"""Email agent task definitions."""

from typing import Any, List
from qwen3_agent.core.framework.task import BaseTask
from pydantic import Field


class EmailTask(BaseTask):
    """Email search task.
    
    Represents a query about emails that the agent needs to answer by
    searching and reading emails from a specific inbox.
    
    Attributes:
        id: Unique identifier (inherited)
        question: User's question about their emails
        answer: Ground truth answer
        message_ids: List of message IDs that contain the answer
        inbox_address: Email address of the inbox to search
        query_date: Date when the query is being asked (for context)
        how_realistic: Score indicating how realistic this query is (0-1)
    """
    
    question: str = Field(description="User's question about their emails")
    answer: str = Field(description="Ground truth answer to the question")
    message_ids: List[str] = Field(description="Message IDs of emails containing the answer")
    inbox_address: str = Field(description="Email address of the inbox to search")
    query_date: str = Field(description="Date when query is asked (YYYY-MM-DD format)")
    how_realistic: float = Field(default=1.0, description="Realism score (0-1)")
    
    def get_query(self) -> str:
        """Get the user's question.
        
        Returns:
            The question string
        """
        return self.question
    
    def get_ground_truth(self) -> Any:
        """Get ground truth answer and sources.
        
        Returns:
            Dictionary with 'answer' and 'message_ids'
        """
        return {
            "answer": self.answer,
            "message_ids": self.message_ids,
        }
    
    def get_context(self) -> dict[str, Any]:
        """Get additional context for the task.
        
        Returns:
            Dictionary with inbox_address, query_date, and other metadata
        """
        return {
            **self.metadata,
            "inbox_address": self.inbox_address,
            "query_date": self.query_date,
            "how_realistic": self.how_realistic,
        }
    
    @classmethod
    def from_synthetic_query(cls, query: Any) -> "EmailTask":
        """Create EmailTask from SyntheticQuery.
        
        This is a convenience method for backward compatibility with
        the existing data loading code.
        
        Args:
            query: SyntheticQuery object from data loader
            
        Returns:
            EmailTask instance
        """
        return cls(
            id=str(query.id),
            question=query.question,
            answer=query.answer,
            message_ids=query.message_ids,
            inbox_address=query.inbox_address,
            query_date=query.query_date,
            how_realistic=query.how_realistic,
        )

