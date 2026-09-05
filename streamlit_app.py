import streamlit as st
import logging
from app.pipeline.orchestrator import run_pipeline
from app.schemas import QueryRequest, InputType
from app.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Schema Rag", page_icon="🏛️", layout="centered")

st.title("🏛️ Schema Rag")
st.markdown("Ask about Indian government welfare scheme eligibility in plain language.")

with st.expander("📖 **How to use this app**", expanded=True):
    st.markdown("""
    Welcome! This AI assistant helps you find and verify your eligibility for thousands of government welfare schemes.
    
    **Tips for best results:**
    - 🧑‍🌾 **Be specific about who you are:** Include details like your occupation, gender, category, and state (e.g., *"I am a female farmer from Maharashtra"*).
    - 💰 **Include your income:** Many schemes are income-dependent (e.g., *"My annual family income is ₹1.5 Lakhs"*).
    - 📝 **Ask for specific help:** (e.g., *"What schemes can help me start a small business?"*).
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., I am a farmer, I have 2 acres of land, what help can I get?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing schemes..."):
            request = QueryRequest(input_type=InputType.TEXT, text=prompt)
            try:
                response = run_pipeline(request)
                
                if response.blocked:
                    st.error(f"**Request Blocked**: {response.block_reason}")
                    st.write(response.answer)
                else:
                    st.markdown(response.answer)
                    
                    if response.verdicts:
                        st.subheader("Eligibility Verdicts")
                        for verdict in response.verdicts:
                            emoji = "✅" if verdict.eligible == "eligible" else "❌" if verdict.eligible == "not_eligible" else "⚠️"
                            with st.expander(f"{emoji} {verdict.scheme_name}"):
                                st.write(f"**Status**: {verdict.eligible.replace('_', ' ').title()}")
                                if verdict.conditions:
                                    st.write("**Conditions Check:**")
                                    for cond in verdict.conditions:
                                        cond_emoji = "✅" if cond.status == "met" else "❌" if cond.status == "not_met" else "⚠️"
                                        st.write(f"- {cond_emoji} **{cond.condition}**: {cond.explanation}")

                    if response.citations:
                        st.subheader("Sources")
                        for citation in response.citations:
                            st.caption(f"- **{citation.scheme_name}**: {citation.text[:150]}...")
                            if citation.source_url:
                                st.caption(f"  [Link]({citation.source_url})")

                st.session_state.messages.append({"role": "assistant", "content": response.answer})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception("Pipeline failed")
