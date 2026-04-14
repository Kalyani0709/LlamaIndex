import streamlit as st
from rag_pipeline import ask

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Mopar Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("🤖 Mopar Chatbot")
st.write("Ask questions based on Mopar website data")

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------
if prompt := st.chat_input("Ask something..."):

    # 🔹 Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 🔹 Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 🔥 CALL RAG PIPELINE
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, results = ask(prompt)

            # 🔹 Display answer
            st.markdown(answer)

            # 🔥 OPTIONAL: show sources (very useful for debugging)
            with st.expander("🔍 Sources"):
                for r in results:
                    source = r.payload.get("metadata", {}).get("source", "")
                    text_preview = r.payload.get("content", "")

                    st.markdown(f"**Source:** {source}")
                    st.write(text_preview)
                    st.markdown("---")

    # 🔹 Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })