if __name__ == "__main__":
    import mnist
    import numpy as np

    import clustering_gemini
    from sklearn.neural_network import MLPClassifier
    import plotly.express as px
    from dim_reduction import create_pca_reducer, create_random_projection_reducer, create_identity_reducer, create_autoencoder_reducer

    STEP = 200

    # Load the MNIST dataset
    train_images, train_labels = mnist.x_train, mnist.y_train
    test_images, test_labels = mnist.x_test, mnist.y_test

    # Flatten the images and normalize the pixel values
    train_images = np.array(train_images)
    train_images = train_images.reshape(-1, 784) / 255.0
    test_images = np.array(test_images)
    test_images = test_images.reshape(-1, 784) / 255.0

    # Sort training data by digit class for incremental learning
    train_labels = np.array(train_labels)
    sorted_indices = np.argsort(train_labels)
    train_images = train_images[sorted_indices]
    train_labels = train_labels[sorted_indices]

    # Sort test data by digit class for incremental evaluation
    test_labels = np.array(test_labels)
    test_sorted_indices = np.argsort(test_labels)
    test_images = test_images[test_sorted_indices]
    test_labels = test_labels[test_sorted_indices]

    # Perform clustering using a model. After each 100 are saved in the cluster_storage,
    # retrain on it, test and save the accuracy.
    accuracy = []
    per_digit_accuracy = {i: [] for i in range(10)}  # Track accuracy for each digit over time

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

    # Track which digit we're currently learning
    current_digit = 0
    digit_counts = np.bincount(train_labels)

    for i in range(0, len(train_images), STEP):
        # Add that batch with labels
        batch_labels = train_labels[i:i+STEP]
        cluster_storage.add_multi(train_images[i:i+STEP], batch_labels)

        # Check if we've moved to a new digit
        if len(batch_labels) > 0:
            batch_digit = batch_labels[0]
            if batch_digit != current_digit:
                print(f"\nNow learning digit {batch_digit}")
                current_digit = batch_digit

        # Train MLP directly on stored cluster data with stored labels
        stored_points, stored_labels = cluster_storage.get_clusters_with_labels()
        stored_labels = np.array(stored_labels)

        # Train MLP with two hidden layers of 200 nodes each
        model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=300, random_state=42)
        model.fit(stored_points, stored_labels)

        # Test only on digits seen so far
        max_digit_seen = max(train_labels[:i+STEP]) if i+STEP > 0 else 0
        test_mask = test_labels <= max_digit_seen
        test_subset = test_images[test_mask]
        test_labels_subset = test_labels[test_mask]

        # Test on relevant test data
        test_images_transformed = cluster_storage.transform(test_subset)
        mapped_predictions = model.predict(test_images_transformed)

        # Calculate overall accuracy
        acc = np.mean(mapped_predictions == test_labels_subset)

        # Calculate per-digit accuracy
        digit_accuracies = {}
        for digit in range(max_digit_seen + 1):
            digit_mask = test_labels_subset == digit
            if np.any(digit_mask):
                digit_acc = np.mean(mapped_predictions[digit_mask] == test_labels_subset[digit_mask])
                digit_accuracies[digit] = digit_acc
                per_digit_accuracy[digit].append(digit_acc)
            else:
                per_digit_accuracy[digit].append(0.0)

        # Fill in zeros for digits not yet seen
        for digit in range(max_digit_seen + 1, 10):
            per_digit_accuracy[digit].append(0.0)

        # Save the overall accuracy
        accuracy.append(acc)
        if i%2500==0:
            print(f"Step {i//STEP}: Overall Accuracy: {accuracy[int(i/STEP)]:.4f} (testing digits 0-{max_digit_seen})")
            for digit in range(max_digit_seen + 1):
                if digit in digit_accuracies:
                    print(f"Accuracy for digit {digit}: {digit_accuracies[digit]:.4f}")

    # # Visualize final neural network results with labeled data points
    # print("Visualizing final neural network results...")
    # stored_points, final_stored_labels = cluster_storage.get_clusters_with_labels()
    # final_stored_labels = np.array(final_stored_labels)

    # # Train final model directly on stored cluster data with stored labels
    # final_model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=300, random_state=42)
    # final_model.fit(stored_points, final_stored_labels)

    # # Get test data predictions for visualization
    # test_subset = test_images[:1000]  # Use subset for cleaner visualization
    # test_labels_subset = np.array(test_labels[:1000])
    # test_transformed = cluster_storage.transform(test_subset)
    # final_predictions = final_model.predict(test_transformed)

    cluster_storage.visualize()

    # # Create 2D projection for visualization using first 2 PCA components
    # if test_transformed.shape[1] >= 2:
    #     fig_clusters = px.scatter(
    #         x=test_transformed[:, 0],
    #         y=test_transformed[:, 1],
    #         color=[str(pred) for pred in final_predictions],
    #         symbol=[str(label) for label in test_labels_subset],
    #         title="Final Neural Network Results (Color=Predicted Digit, Symbol=True Digit)",
    #         labels={'color': 'Predicted Digit', 'symbol': 'True Digit'}
    #     )
    #     fig_clusters.show()

    # Also visualize the accuracy as a function of time
    import pandas as pd

    # Create dataframe for per-digit accuracy visualization
    steps = list(range(len(accuracy)))
    plot_data = []

    for digit in range(10):
        for step, acc in enumerate(per_digit_accuracy[digit]):
            plot_data.append({
                'Step': step,
                'Accuracy': acc,
                'Digit': f'Digit {digit}'
            })

    df = pd.DataFrame(plot_data)

    # Create multi-line plot showing accuracy per digit over time
    fig = px.line(df, x='Step', y='Accuracy', color='Digit',
                  title='Per-Digit Accuracy Over Time',
                  range_y=[0.0, 1.0])
    fig.show()

    # Also show overall accuracy
    fig_overall = px.line(x=range(len(accuracy)), y=accuracy,
                         title='Overall Accuracy Over Time',
                         range_y=[0.0, 1.0])
    fig_overall.show()
