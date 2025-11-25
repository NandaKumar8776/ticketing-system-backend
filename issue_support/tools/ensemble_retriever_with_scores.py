"""
Custom Ensemble Retriever that returns both documents and their fused scores.
"""

from typing import List, Tuple, Any, Optional
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import CallbackManager
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.json import dumpd
from langchain_core.documents import Document


class EnsembleRetrieverWithScores(EnsembleRetriever):
    """
    Extended EnsembleRetriever that returns documents with their fused scores.
    
    Returns: List[Tuple[Document, float]]
        Each tuple contains (document, fused_score)
    """

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Invoke the retriever and return documents with their fused scores.
        
        Args:
            input: Search query string
            config: Optional configuration
            **kwargs: Additional keyword arguments
            
        Returns:
            List of tuples containing (Document, fused_score)
        """
        from langchain_core.callbacks import CallbackManager
        from langchain_core.utils.function_calling import ensure_config

        config = ensure_config(config)

        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )

        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            **kwargs,
        )

        try:
            # Get results with scores using rank_fusion
            result_with_scores = self.rank_fusion(input, run_manager=run_manager, config=config)

            # Extract only documents for callback
            result = [doc for doc, score in result_with_scores]

        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,  # Pass only the documents, not the scores
                **kwargs,
            )

            return result_with_scores  # Return both documents and scores
