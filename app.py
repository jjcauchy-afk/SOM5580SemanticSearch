import streamlit as st

# Use st.write for a simple text message
st.write("Hello World2!")

# Or use st.title for a larger, bold header
st.title("👋 Hello Streamlit!")

name = st.text_input("Enter your name:")

if st.button("Submit"):
    st.success(f"Hello, {name}!")
