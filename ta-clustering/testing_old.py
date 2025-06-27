if __name__ == "__main__":
    import mnist
    import numpy as np

    import clustering_gemini
    from sklearn.cluster import KMeans, HDBSCAN
    import plotly.express as px
    from dim_reduction import create_pca_reducer, create_random_projection_reducer, create_identity_reducer, create_autoencoder_reducer

    STEP = 100

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
    reducer = create_pca_reducer(50)                    # PCA with 50 components
    # reducer = create_random_projection_reducer(100)   # Random projection with 100 components
    # reducer = create_identity_reducer()               # No dimensionality reduction
    # reducer = create_autoencoder_reducer(10, hidden_layers=[32, 32], epochs=20)  # Autoencoder with 10 components

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
        # model = HDBSCAN(min_cluster_size=10)
        model = KMeans(n_clusters=10)
        X = cluster_storage.get_clusters_for_training()
        model.fit(X)

        # Create mapping using TRAINING data - use all training data seen so far
        train_data_so_far = train_images[:i+STEP]
        train_labels_so_far = np.array(train_labels[:i+STEP])
        train_transformed = cluster_storage.transform(train_data_so_far)
        train_predictions = model.predict(train_transformed)

        # Map cluster IDs to most frequent digit labels using training data
        cluster_to_digit = {}
        for cluster_id in np.unique(train_predictions):
            mask = train_predictions == cluster_id
            cluster_train_labels = train_labels_so_far[mask]
            if len(cluster_train_labels) > 0:
                cluster_to_digit[cluster_id] = np.bincount(cluster_train_labels.astype(int)).argmax()

        # Test on test data
        test_images_transformed = cluster_storage.transform(test_images)
        returned_labels = model.predict(test_images_transformed)

        # Convert cluster predictions to digit predictions
        mapped_predictions = np.array([cluster_to_digit.get(label, -1) for label in returned_labels])

        # Calculate accuracy excluding noise points
        valid_mask = mapped_predictions != -1
        test_labels_np = np.array(test_labels)
        if np.sum(valid_mask) > 0:
            acc = np.mean(mapped_predictions[valid_mask] == test_labels_np[valid_mask])
        else:
            acc = 0.0

        # Save the accuracy
        accuracy.append(acc)
        if i%2500==0:
            print("Accuracy:", accuracy[int(i/STEP)])

    # Visualize final clustering results with labeled data points
    print("Visualizing final clustering results...")
    final_model = KMeans(n_clusters=10)
    X_final = cluster_storage.get_clusters_for_training()
    final_model.fit(X_final)

    # Get test data predictions for visualization
    test_subset = test_images[:1000]  # Use subset for cleaner visualization
    test_labels_subset = np.array(test_labels[:1000])
    test_transformed = cluster_storage.transform(test_subset)
    final_predictions = final_model.predict(test_transformed)

    # Map cluster IDs to digits for final model
    final_cluster_to_digit = {}
    for cluster_id in np.unique(final_predictions):
        mask = final_predictions == cluster_id
        cluster_test_labels = test_labels_subset[mask]
        if len(cluster_test_labels) > 0:
            final_cluster_to_digit[cluster_id] = np.bincount(cluster_test_labels.astype(int)).argmax()

    # Create 2D projection for visualization using first 2 PCA components
    if test_transformed.shape[1] >= 2:
        fig_clusters = px.scatter(
            x=test_transformed[:, 0],
            y=test_transformed[:, 1],
            color=[str(final_cluster_to_digit.get(pred, pred)) for pred in final_predictions],
            symbol=[str(label) for label in test_labels_subset],
            title="Final Clustering Results (Color=Predicted Digit, Symbol=True Digit)",
            labels={'color': 'Predicted Digit', 'symbol': 'True Digit'}
        )
        fig_clusters.show()

    # Also visualize the accuracy as a function of time
    fig = px.line(x=range(len(accuracy)), y=accuracy, range_y=[0.0, 1.0])
    fig.show()