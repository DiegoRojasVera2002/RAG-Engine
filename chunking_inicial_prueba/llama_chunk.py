from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

def chunk_text(text: str):
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(
        [Document(text=text)]
    )
    return [n.text for n in nodes]
