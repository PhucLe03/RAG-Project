from typing import Any, Callable
from langchain_core.documents import Document
import re

class RAGDocument(Document):
    def __init__(self, page_content: str, **kwargs: Any) -> None:
        super().__init__(page_content=page_content, **kwargs)
        
    def mark_as_text(self):
        """
        Mark this document as a text document.

        This method sets the 'type' key in the document's metadata to 'text'.

        Returns:
            None
        """
        self.metadata["type"] = "text"
        
    def mark_as_code(self):
        """
        Mark this document as a code document.

        This method sets the 'type' key in the document's metadata to 'code'.
        """
        self.metadata["type"] = "code"
        
    def __str__(self) -> str:
        return super().__str__()

    
class TextAndCodeLabeler:
    # _block_pattern = r"``(?:[\s\S]*?)"
    _code_pattern = r"```(?:[\s\S]*?)```"
    def __init__(self):
        # self._chunks: list[RAGDocument] = []
        pass
        
    def label_docs(self, documents: list[Document]) -> list[RAGDocument]:
        chunks_list = []
        for doc in documents:
            metadata = doc.metadata
            content = doc.page_content
            if re.match(self._code_pattern, content):
                metadata["type"] = "code"
            else:
                metadata["type"] = "text"
            chunks_list.append(RAGDocument(content, metadata=metadata))
        return chunks_list
    

class FixedSizeTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        delete_whitespace: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.delete_whitespace = delete_whitespace
        
    def _split_text(self, text: str, metadata: dict) -> list[RAGDocument]:
        splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return [RAGDocument(s, metadata=metadata) for s in splits]
        
    def split_docs(self, docs: list[RAGDocument]) -> list[RAGDocument]:
        docs_to_return: list[RAGDocument] = []
        # return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for doc in docs:
            if doc.metadata["type"] == "text":
                docs_to_return.extend(self._split_text(doc.page_content, doc.metadata))
            else:
                docs_to_return.append(doc)
                
        return docs_to_return
    
class SlidingWindowTextSplitter:
    def __init__(
        self,
        chunk_window_size: int = 4000,
        stride: int = 200,
        delete_whitespace: bool = True,
    ) -> None:
        self.chunk_window_size = chunk_window_size
        self.stride = stride
        self.delete_whitespace = delete_whitespace
        
    def _split_text(self, text: str, metadata: dict) -> list[RAGDocument]:
        splits = [text[i : i + self.chunk_window_size] for i in range(0, len(text) - self.chunk_window_size + 1, self.stride)]
        return [RAGDocument(s, metadata=metadata) for s in splits]
        
    def split_docs(self, docs: list[RAGDocument]) -> list[RAGDocument]:
        docs_to_return: list[RAGDocument] = []
        for doc in docs:
            if doc.metadata["type"] == "text":
                docs_to_return.extend(self._split_text(doc.page_content, doc.metadata))
            else:
                docs_to_return.append(doc)
                
        return docs_to_return

class SentenceTextSplitter:
    def __init__(self):
        pass
    
    def _split_text(self, text: str, metadata: dict) -> list[RAGDocument]:
        splits = [sentence.strip() for sentence in text.replace("\n", " ").split(". ") if sentence]
        return [RAGDocument(s, metadata=metadata) for s in splits]
    
    def split_docs(self, docs: list[RAGDocument]) -> list[RAGDocument]:
        docs_to_return: list[RAGDocument] = []
        for doc in docs:
            if doc.metadata["type"] == "text":
                docs_to_return.extend(self._split_text(doc.page_content, doc.metadata))
            else:
                docs_to_return.append(doc)
                
        return docs_to_return
    
class SematicTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        delete_whitespace: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.delete_whitespace = delete_whitespace
        
    def _split_text(self, text: str, metadata: dict) -> list[RAGDocument]:
        splits = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return [RAGDocument(s, metadata=metadata) for s in splits]