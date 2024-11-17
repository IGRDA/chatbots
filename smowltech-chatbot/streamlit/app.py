import streamlit as st
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="SmilyOwly Bot",
    page_icon=":robot_face:",
)

# Custom CSS for styling
st.markdown("""
    <style>
    a {
        color: black !important; /* Change hyperlink color to black */
    }
    a:hover {
        color: gray !important; /* Optional: Change color on hover */
    }
    </style>
    """, unsafe_allow_html=True)


class SentenceTransformerEmbeddings(Embeddings):
    """Embedding model using Sentence Transformers."""

    def __init__(self, model_name: str = ''):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()


def write_chat_messages(chat_container, messages):
    """Render chat messages in the Streamlit chat container."""
    if not isinstance(messages, list):
        messages = [messages]

    with chat_container:
        for message in messages:
            if message["author"] == "user":
                with st.chat_message(name="user"):
                    st.write(message["message"])
            elif message["author"] == "assistant":
                with st.chat_message(name="assistant"):
                    st.write(message["message"])


def prepare_chat_history(chat_history, max_mem=2, save_assistant=True):
    """Prepare conversation history for the chatbot."""
    history = ""

    for message in chat_history[-max_mem:]:
        if message["author"] == "user":
            history += f"\nUser:\n{message['message']}\n"
        if save_assistant and message["author"] == "assistant":
            history += f"\nAssistant:\n{message['message']}\n"

    return history


# Constants
FUNCTION_NAME = "customer-service-chatbot"
FIRST_MESSAGE = {
    "spanish": "¡Hola! 😊 Bienvenido al soporte al cliente de Smowltech. ¿Hay algo en lo que pueda ayudarte con nuestra tecnología o servicios de supervisión remota? Estoy aquí para garantizar que tus exámenes y evaluaciones en remoto se realicen de manera fluida y segura.",
}
YOUR_MESSAGE = {"spanish": "Tu mensaje:"}
PROCESSING_MESSAGE = {"spanish": "Procesando tu petición..."}

# Model and vectorstore setup
EMBEDDINGS_MODEL_ID = "intfloat/multilingual-e5-large"
LLM = OllamaLLM(model="llama3.1:8b", mirostat_tau=0)

VECTORSTORE_FAISS = FAISS.load_local(
    folder_path="../faiss_index",
    embeddings=SentenceTransformerEmbeddings(EMBEDDINGS_MODEL_ID),
    allow_dangerous_deserialization=True,
)

PROMPT_TEMPLATE = """You will be acting as a customer support chatbot named Smowlbot, created by the company Smowltech, a leader in online proctoring solutions.  
Your goal is to provide support to customers regarding Smowltech’s products and services, ensuring secure and reliable environments for remote exams and assessments.  
You are part of Smowltech's customer support team. Always stay in character.

Here are some important rules for the interaction:  
- Use the chat history and the following pieces of context to answer the question.  
- Only use relevant parts of the context.  
- If the context is meaningful to answer the query, ensure your response aligns with the context and refrain from fabricating answers.  
- Do not include web, email, or other links unless they are in the context.  
- Provide assistance, recommendations, or advice only on matters related to Smowltech's proctoring technology, services, and processes.  
- Always maintain a friendly and professional tone in your responses.  
- Use three sentences maximum and keep the answer clear, concise, and easy to follow.  
- If the question is unclear or you need additional crucial user information to provide an accurate answer, feel free to ask the user for more details.
- Don't say directly that the information was found in the context.

Here is the conversation history (between the user and you) prior to the question. There are just the last messages:
<history>
{chat_history}
</history>

Here is the context:
<context>
{{context}}
</context>
Here is the user’s question:
<question>
{{question}}
</question>

How do you respond to the user’s question?  
Answer in the language of the question.  
Always focus on responding to the user's messages and using only the context that directly relates to their queries.  
Think about your answer first before you respond.  
"""


def main():
    """Main function for the Streamlit app."""
    st.image("images/blue-owl.png", width=75)
    st.title("SmilyOwly Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"author": "assistant", "message": FIRST_MESSAGE["spanish"]}
        ]

    st.divider()
    chat_container = st.container()
    user_input = st.chat_input(YOUR_MESSAGE["spanish"])
    status_container = st.container()

    write_chat_messages(chat_container, st.session_state["chat_history"])

    if user_input:
        user_message = {"author": "user", "message": user_input}
        st.session_state["chat_history"].append(user_message)
        write_chat_messages(chat_container, user_message)

        with status_container:
            chat_history = prepare_chat_history(st.session_state["chat_history"])
            prompt = PROMPT_TEMPLATE.format(chat_history=chat_history)

            prompt_instance = PromptTemplate(
                template=prompt, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=LLM,
                chain_type="stuff",
                retriever=VECTORSTORE_FAISS.as_retriever(
                    search_type="similarity", search_kwargs={"k": 7}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_instance},
            )

            result = qa_chain({"query": user_input})

        assistant_output = result["result"]
        source_limit = 3
        sources = [
            f"- {source}\n"
            for source in list(set(doc.metadata["url"] for doc in result["source_documents"]))[:source_limit]
        ]

        assistant_message = {"author": "assistant", "message": assistant_output}
        if sources:
            assistant_message["message"] += "\n" + "".join(sources)

        st.session_state["chat_history"].append(assistant_message)
        write_chat_messages(chat_container, assistant_message)


if __name__ == "__main__":
    main()