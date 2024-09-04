import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_word_vectors(word_vectors, n_components=2):
    """
    Plot word vectors in 2D space using PCA for dimensionality reduction.
    
    :param word_vectors: dict of word: vector pairs
    :param n_components: number of PCA components to use (default 2)
    :return: matplotlib figure
    """
    # Extract words and vectors
    words = list(word_vectors.keys())
    vectors = list(word_vectors.values())

    # Perform PCA
    pca = PCA(n_components=n_components)
    vectors_2d = pca.fit_transform(vectors)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    # Add word labels
    for i, word in enumerate(words):
        ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))

    ax.set_title("Word Vectors Visualization")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")

    return fig
