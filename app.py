import streamlit as st
from rag_pipeline import ask

st.set_page_config(page_title="Mopar Chatbot", layout="wide")

st.markdown("""
<h2 style='text-align:center;'>🚗 Mopar Assistant</h2>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, results = ask(prompt)

            st.markdown(answer)

            with st.expander("🔍 Sources"):
                for r in results:
                    st.markdown(f"**Source:** {r.payload['metadata'].get('source')}")
                    st.write(r.payload.get("content")[:300])
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})