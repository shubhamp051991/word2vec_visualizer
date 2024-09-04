import streamlit as st
from src.word2vec_model import CBOW, SkipGram
from src.visualizations import plot_word_vectors

def main():
    st.title("Word2Vec Visualization")

    # User input
    user_input = st.text_area("Enter text (one sentence per line):", "The cat sat on the mat.\nDogs chase cats.\nBirds fly in the sky.")

    # Model parameters
    st.sidebar.header("Model Parameters")
    vector_size = st.sidebar.slider("Vector Size", min_value=50, max_value=300, value=100, step=50)
    window = st.sidebar.slider("Window Size", min_value=2, max_value=10, value=5)
    min_count = st.sidebar.slider("Min Count", min_value=1, max_value=5, value=1)

    # Model selection
    model_type = st.radio("Select Word2Vec architecture:", ("CBOW", "Skip-gram"))

    if st.button("Train and Visualize"):
        # Prepare input
        sentences = user_input.split('\n')

        # Initialize and train the selected model
        if model_type == "CBOW":
            model = CBOW(sentences, vector_size=vector_size, window=window, min_count=min_count)
        else:
            model = SkipGram(sentences, vector_size=vector_size, window=window, min_count=min_count)

        model.train()

        # Get word vectors
        word_vectors = model.get_word_vectors()

        # Visualize word vectors
        fig = plot_word_vectors(word_vectors)
        st.pyplot(fig)

        # Display vector arithmetic example
        st.subheader("Vector Arithmetic Example")
        words = list(word_vectors.keys())
        if len(words) >= 3:
            word1, word2, word3 = st.selectbox("Word 1", words), st.selectbox("Word 2", words), st.selectbox("Word 3", words)
            if st.button("Calculate"):
                result = model.vector_arithmetic(word1, word2, word3)
                st.write(f"{word1} - {word2} + {word3} ≈ {result}")

if __name__ == "__main__":
    main()
