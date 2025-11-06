"""Model and TrainableModel classes."""

from pydantic import BaseModel
from typing import Iterable


class Model(BaseModel):
    """A model configuration for inference and logging."""
    
    name: str
    project: str
    config: BaseModel | None = None
    
    # Inference connection information
    inference_api_key: str | None = None
    inference_base_url: str | None = None
    inference_model_name: str | None = None
    
    def get_inference_name(self) -> str:
        """Return the name that should be sent to the inference endpoint."""
        return self.inference_model_name or self.name


class TrainableModel(Model):
    """A trainable model that can be fine-tuned."""
    
    base_model: str
    trainable: bool = True

