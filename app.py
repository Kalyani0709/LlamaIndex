import streamlit as st
from rag_pipeline import ask

st.set_page_config(page_title="Mopar Assistant", layout="centered")

# 🔥 Custom styling
st.markdown("""
<style>
.main {
    display: flex;
    justify-content: center;
}
.block-container {
    max-width: 800px;
    padding-top: 2rem;
}
.chat-title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #D32F2F;
}
.chat-sub {
    text-align: center;
    color: grey;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 🔴 Mopar Header
st.markdown("""
<div style="text-align: center; margin-bottom: 10px; margin-top: 10px">
    <img src="https://www.mopar.com/content/dam/mopar/images/header/logos/mopar_logo.svg" width="140">
</div>

<div class="chat-title">Mopar Assistant</div>
<div class="chat-sub">Ask anything about your vehicle, warranty, or services</div>
""", unsafe_allow_html=True)


# 🔹 Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# 🔹 Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 🔹 Input
if prompt := st.chat_input("Ask about Mopar services, warranty, maintenance..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔧 Thinking..."):

            answer, results = ask(prompt)

            st.markdown(answer)

            # 🔍 Sources
            with st.expander("🔍 Sources"):
                for r in results[:5]:
                    meta = r.payload.get("metadata", {})
                    source = meta.get("source", "Unknown")
                    heading = meta.get("heading", "")

                    st.markdown(f"**📄 {source}** — *{heading}*")
                    st.write(r.payload.get("content", "")[:300])
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})