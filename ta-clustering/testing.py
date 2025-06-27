if __name__ == "__main__":
    import mnist
    import numpy as np

    import clustering_gemini
    from sklearn.cluster import KMeans
    import plotly.express as px
    from dim_reduction import create_pca_reducer, create_random_projection_reducer, create_identity_reducer, create_autoencoder_reducer

    STEP = 500

    # Load the MNIST dataset
    train_images, train_labels = mnist.x_train, mnist.y_train
    test_images, test_labels = mnist.x_test, mnist.y_test

    # Flatten the images and normalize the pixel values
    train_images = np.array(train_images)
    train_images = train_images.reshape(-1, 784) / 255.0
    test_images = np.array(test_images)
    test_images = test_images.reshape(-1, 784) / 255.0

    # TODO Perform clustering using k-means. After each 100 are saved in the cluster_storage,
    # retrain on it, test and save the accuracy.
    accuracy = []

    # Easy switching between reducers - uncomment the one you want:
    # reducer = create_pca_reducer(10)                    # PCA with 10 components
    # reducer = create_random_projection_reducer(100)   # Random projection with 100 components
    # reducer = create_identity_reducer()               # No dimensionality reduction
    reducer = create_autoencoder_reducer(10, hidden_layers=[128, 64], epochs=50)  # Autoencoder with 10 components

    cluster_storage = clustering_gemini.ClusteringMechanism(Q=10, P=100, dimensionality_reducer=reducer)

    # Fit reducer on initial batch
    print("Fitting reducer on initial batch...")
    cluster_storage.fit_reducer(train_images[:(STEP*2)])
    print("Fitted!")

    
    for i in range(0, len(train_images), STEP):
        # Add that batch
        cluster_storage.add_multi(train_images[i:i+STEP])
        if i == STEP:
            cluster_storage.visualize()
        # Train
        model = KMeans(n_clusters=10)
        X = cluster_storage.get_clusters_for_training()
        model.fit(X)
        # Test on numbers to that point - transform test data with same PCA
        test_images_transformed = cluster_storage.transform(test_images)
        returned_labels = model.predict(test_images_transformed)
        # Save the accuracy
        accuracy.append(np.mean(returned_labels == test_labels))
        if i%2500==0:
            print("Accuracy:", accuracy[int(i/STEP)])
    # Visualize the accuracy as a function of time
    fig = px.line(x=range(len(accuracy)), y=accuracy, range_y=[0.0, 1.0])
    fig.show()
