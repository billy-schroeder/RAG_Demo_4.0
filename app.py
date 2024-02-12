import streamlit as st
import openai
from components.sidebar import sidebar
from components.utils import (
    initialize_state,
    load_qa_chain
)

def main():
    # Set up the Streamlit page configuration and title
    st.set_page_config(page_title="RAG Demo")
    st.title("RAG Demo")
    # Invoke the sidebar for user inputs like file uploads and OpenAI keys
    saved_files_info, openai_keys = sidebar()
    st.markdown("***")
    st.subheader('Interact with Documents')

    # Initialize the session state variables
    initialize_state()

    # Add a flag in the session state for API key validation
    if "is_api_key_valid" not in st.session_state:
        st.session_state.is_api_key_valid = None

    # Load the QA chain if documents and OpenAI keys are provided, and handle OpenAI AuthenticationError
    if openai_keys and not st.session_state.qa_chain:
        try:
            st.session_state.qa_chain = load_qa_chain(saved_files_info, openai_keys)
            st.session_state.is_api_key_valid = True  # Valid API key
        except openai.AuthenticationError as e:
            st.error('Please provide a valid API key. Update the API key in the sidebar and click "Complete Configuration" to proceed.', icon="🚨")
            st.session_state.is_api_key_valid = False  # Invalid API key

    # Enable the chat section if the QA chain is loaded and API key is valid
    if st.session_state.qa_chain and st.session_state.is_api_key_valid:
        st.success("Configuration complete")
        prompt = st.chat_input('Ask questions about the uploaded documents', key="chat_input")

        # Process user prompts and generate responses
        if prompt and (st.session_state.messages[-1]["content"] != prompt or st.session_state.messages[-1]["role"] != "user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Retrieving relevant information and generating output..."):
                response = st.session_state.qa_chain.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the conversation messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    else:
        st.info("Please complete the configuration in the sidebar to proceed.")
        # Disable the chat input if the API key is invalid
        no_prompt = st.chat_input('Ask questions about the uploaded documents', disabled=not st.session_state.is_api_key_valid)

if __name__ == '__main__':
    main()