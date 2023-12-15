import streamlit as st
from similairty_calculator import calculate_similarity

def main():
    st.title("Language Similarity Calculator")
    
    col1, col2 = st.columns(2)
    
    # Text areas for entering sentences in both languages
    with col1:
        st.header("Enter text in Language 1")
        sentences_lang1 = st.text_area("Language 1 Sentences", height=200)
    
    with col2:
        st.header("Enter text in Language 2")
        sentences_lang2 = st.text_area("Language 2 Sentences", height=200)
    
    # Button to trigger similarity computation
    if st.button("Find Similarity"):
        if sentences_lang1 and sentences_lang2:
            # Split input into separate sentences for processing
            sentences_lang1 = sentences_lang1.split('\n')
            sentences_lang2 = sentences_lang2.split('\n')
            
            # Calculate similarity
            similarity_scores,plt = calculate_similarity(sentences_lang1, sentences_lang2)
            # similarity_scores = "Not yet implemented.."
            print(similarity_scores)
            
            # Display similarity result
            st.header("Similarity Result")
            st.dataframe(similarity_scores)
            st.pyplot(plt)
        else:
            st.warning("Please enter sentences in both languages.")

if __name__ == "__main__":
    main()