from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool as LC_Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from retriever import load_vector_store
from utils.cache import check_cache, update_cache, embed_query, cosine_similarity
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import LLM
from langchain.llms import OpenAI  # fallback if needed
from langchain.chat_models import ChatOpenAI

# === Local LLM via LM Studio Server ===
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="mistralai/Mistral-7B-Instruct-v0.3"
)

# === Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load Vector Stores ===
small_vectorstore = load_vector_store("small_vector_db")
small_retriever = small_vectorstore.as_retriever(search_kwargs={"k": 3})

# === QA Chain with Custom Prompt (no roles) ===
def get_qa_chain(retriever):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question.

Context: {context}
Question: {question}

Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )

# === Internal Policy Tool ===
def dynamic_internal_policy_tool(query: str) -> str:
    query_vector = embed_query(query)
    small_docs = small_vectorstore.similarity_search_by_vector(query_vector.tolist(), k=3)

    best_score = -1
    best_doc = None
    for doc in small_docs:
        doc_vector = embed_query(doc.page_content)
        score = cosine_similarity(query_vector, doc_vector)
        if score > best_score:
            best_score = score
            best_doc = doc

    print(f"🔍 Small DB Cosine Score: {best_score:.3f}")

    if best_score >= 0.5:
        print("✅ Answering using small vector DB.")
        return get_qa_chain(small_retriever).invoke({"query": query})["result"]

    return "Could not confidently retrieve an answer from the policy documents."

# === Tool ===
def internal_policy_tool_wrapper(input: str) -> str:
    return dynamic_internal_policy_tool(input)

access_policy_tool = LC_Tool(
    func=internal_policy_tool_wrapper,
    name="internal_policy_assistant",
    description="Use this tool to answer technical or policy-related questions using internal documents."
)


# === Tools ===
tools = [access_policy_tool]

# === Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# === Prompt Template for Agent ===
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "chat_history"],
 template="""
You are an intelligent AI assistant having a conversation with a human.
Use the following conversation history and the current question to provide a relevant and contextual response.

If needed, use the internal_policy_assistant tool to get precise answers from internal documents.

Conversation history:
{chat_history}

Current question:
{input}

{agent_scratchpad}
"""

)

# === Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_PROMPT},
    handle_parsing_errors=True
)

# === CLI Interface ===
if __name__ == "__main__":
    print("🤖 LangChain Agent with PDF Knowledge + Memory")
    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break
            response = check_cache(query)
            if response:
                print("Bot (cached):", response)
            else:
                response = agent.invoke({"input": query})
                update_cache(query, response["output"])
                print("Bot:", response["output"])
        except KeyboardInterrupt:
            print("Goodbye!")
            break
