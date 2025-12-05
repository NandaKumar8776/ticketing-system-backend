"""
Custom Ensemble Retriever that returns both documents and their fused scores.
"""

from typing import List, Tuple, Any, Optional, Dict
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumps
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

        # normalize config to a dict-like object
        if config is None:
            config = {}
        elif not hasattr(config, "get"):
            try:
                config = dict(config)
            except Exception:
                config = {}

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
            dumps(self),
            input,
            name=config.get("run_name"),
            **kwargs,
        )

        try:
            # Manually compute fused scores by calling each retriever
            # and combining results with weighted scores
            doc_score_map: Dict[str, Tuple[Document, float]] = {}
            
            # Call each retriever and collect results
            for retriever_idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
                print(f"[ensemble] Calling retriever {retriever_idx} with weight {weight}")
                # Get documents from this retriever
                try:
                    docs = retriever.invoke(input)
                    print(f"[ensemble] Retriever {retriever_idx} returned {len(docs)} documents")
                except Exception as e:
                    print(f"[ensemble] ERROR calling retriever {retriever_idx}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
                
                # Assign reciprocal rank scores (1/rank for top result, 1/(rank+1) for second, etc.)
                for rank, doc in enumerate(docs, start=1):
                    try:
                        # Use a combination of metadata and content to create a unique key
                        # Prefer metadata file_path and page, fallback to content slice
                        metadata = getattr(doc, 'metadata', {}) or {}
                        source = metadata.get('source', 'unknown')
                        page = metadata.get('page', 'unknown')
                        content = doc.page_content if doc.page_content else 'empty'
                        print(f"[ensemble] Processing doc rank {rank}: source={source}, page={page}, content_len={len(content)}")
                        content_snippet = content[:30] if len(content) >= 30 else content
                        doc_key = f"{source}|page_{page}|{content_snippet}"
                        
                        # Reciprocal rank (1/rank) normalized to 0-1 range
                        reciprocal_rank_score = 1.0 / rank
                        weighted_score = weight * reciprocal_rank_score
                        
                        if doc_key not in doc_score_map:
                            doc_score_map[doc_key] = (doc, weighted_score)
                        else:
                            # Accumulate scores if document appears in multiple retrievers
                            existing_doc, existing_score = doc_score_map[doc_key]
                            doc_score_map[doc_key] = (existing_doc, existing_score + weighted_score)
                    except Exception as e:
                        print(f"[ensemble] ERROR processing document at rank {rank}: {e}")
                        import traceback
                        print(traceback.format_exc())
                        continue
            
            print(f"[ensemble] Total unique documents: {len(doc_score_map)}")
            
            # Sort by fused score (descending) and return list of (doc, score) tuples
            result_with_scores = sorted(
                doc_score_map.values(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"[ensemble] Sorted {len(result_with_scores)} results with scores")
            
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
