"""Simple Example."""

from typing import TypedDict

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

from portia.config import Config, LogLevel
from portia.execution_context import ExecutionContext
from portia.llm_wrapper import LLMWrapper
from portia.runner import Runner
from portia.tool import Tool
from portia.tool_registry import InMemoryToolRegistry


class RetrievalAugmentQueryToolSchema(BaseModel):
    """Input for RetrievalAugmentQueryTool."""

    question: str = Field(
        ...,
        description=("The question to search for in the given doc source."),
    )


class RetrievalAugmentQueryTool(Tool[str]):
    """Uses RAG to answer questions."""

    id: str = "rag_tool"
    name: str = "Rag Tool"
    args_schema: type[BaseModel] = RetrievalAugmentQueryToolSchema
    output_schema: tuple[str, str] = ("str", "str: output of the search results")
    domain: str

    def run(self, _: ExecutionContext, question: str) -> str:
        """Run the Rag Tool."""
        loader = RecursiveUrlLoader(
            url=self.domain,
        )

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)

        # Define prompt for question-answering
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question},
            Context: {context} ,
            Answer:""",
                ),
            ],
        )

        # Define state for application
        class State(TypedDict):
            question: str
            context: list[Document]
            answer: str

        llm = LLMWrapper(Config.from_default()).to_langchain()

        # Define application steps
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        output = graph.invoke({"question": question})
        return output["answer"]


runner = Runner(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tool_registry=InMemoryToolRegistry.from_local_tools(
        [
            RetrievalAugmentQueryTool(
                domain="https://docs.portialabs.ai",
                description="Used to retrieve information from the Portia SDK docs.",
            ),
        ],
    ),
)


workflow = runner.execute_query(
    "How do I identify which user is running a workflow in the portia sdk?",
)
