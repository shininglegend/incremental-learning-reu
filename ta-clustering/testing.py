if __name__ == "__main__":
    import mnist
    import numpy as np

    import clustering_gemini
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import plotly.express as px


    # Load the MNIST dataset
    train_images, train_labels = mnist.x_train, mnist.y_train
    test_images, test_labels = mnist.x_test, mnist.y_test

    # Flatten the images and normalize the pixel values
    train_images = np.array(train_images)
    train_images = train_images.reshape(-1, 784) / 255.0

    # TODO Perform clustering using k-means. After each 100 are saved in the cluster_storage,
    # retrain on it, test and save the accuracy.
    accuracy = []
    cluster_storage = clustering_gemini.ClusteringMechanism(Q=10, P=50)
    for i in range(0, len(train_images), 100):
        # Add that batch
        cluster_storage.add_multi(train_images[i:i+100])
        cluster_storage.visualize()
        # Train
        model = KMeans(n_clusters=10)
        X = cluster_storage.get_clusters_for_training()
        model.fit(X)
        # Test on numbers to that point
        returned_labels = model.predict(test_images)
        # Save the accuracy
        accuracy.append(np.mean(returned_labels == test_labels))
        print("Accuracy:", accuracy)

    # Visualize the accuracy as a function of time
    fig = px.line(x=range(len(accuracy)), y=accuracy)
    fig.show()
