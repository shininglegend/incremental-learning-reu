import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA as SklearnPCA

class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction techniques."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.fitted = False

    @abstractmethod
    def fit(self, X):
        """Fit the dimensionality reducer on training data."""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform data using the fitted reducer."""
        pass

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class PCAReducer(DimensionalityReducer):
    """PCA-based dimensionality reduction using sklearn."""

    def __init__(self, n_components=None):
        super().__init__(n_components)
        self.pca = SklearnPCA(n_components=n_components) if n_components else None

    def fit(self, X):
        """Fit PCA on training data."""
        if self.pca is None:
            return

        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.pca.fit(X)
        self.fitted = True

    def transform(self, X):
        """Transform data using fitted PCA."""
        if self.pca is None:
            return X

        if not self.fitted:
            raise ValueError("PCA must be fitted before transforming data. Call fit() first.")

        if isinstance(X, list):
            X = np.array(X)

        # Handle both single samples and batches
        if len(X.shape) == 1:
            X_reshaped = X.reshape(1, -1)
            return self.pca.transform(X_reshaped).flatten()
        else:
            return self.pca.transform(X)


class IdentityReducer(DimensionalityReducer):
    """No-op dimensionality reducer that returns data unchanged."""

    def __init__(self, n_components=None):
        super().__init__(n_components)

    def fit(self, X):
        """No-op fit."""
        self.fitted = True

    def transform(self, X):
        """Return data unchanged."""
        return X


class RandomProjectionReducer(DimensionalityReducer):
    """Random projection dimensionality reduction."""

    def __init__(self, n_components):
        super().__init__(n_components)
        self.projection_matrix = None
        self.input_dim = None

    def fit(self, X):
        """Fit random projection matrix."""
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.input_dim = X.shape[1]
        # Generate random projection matrix with Gaussian distribution
        self.projection_matrix = np.random.normal(0, 1, (self.input_dim, self.n_components))
        # Normalize columns
        self.projection_matrix = self.projection_matrix / np.sqrt(self.n_components)
        self.fitted = True

    def transform(self, X):
        """Transform data using random projection."""
        if not self.fitted:
            raise ValueError("Random projection must be fitted before transforming data. Call fit() first.")

        if isinstance(X, list):
            X = np.array(X)

        # Handle both single samples and batches
        if len(X.shape) == 1:
            return np.dot(X, self.projection_matrix)
        else:
            return np.dot(X, self.projection_matrix)


class AutoencoderReducer(DimensionalityReducer):
    """Autoencoder-based dimensionality reduction using neural networks."""

    def __init__(self, n_components, hidden_layers=None, epochs=100, batch_size=32, learning_rate=0.001):
        super().__init__(n_components)
        self.hidden_layers = hidden_layers or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.encoder = None
        self.input_dim = None

    def _build_autoencoder(self, input_dim):
        """Build the autoencoder architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))

        # Encoder layers
        encoded = input_layer
        for hidden_dim in self.hidden_layers:
            encoded = layers.Dense(hidden_dim, activation='relu')(encoded)

        # Bottleneck layer
        bottleneck = layers.Dense(self.n_components, activation='relu', name='bottleneck')(encoded)

        # Decoder layers
        decoded = bottleneck
        for hidden_dim in reversed(self.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation='relu')(decoded)

        # Output layer
        output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

        # Create models
        self.autoencoder = keras.Model(input_layer, output_layer)
        self.encoder = keras.Model(input_layer, bottleneck)

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

    def fit(self, X):
        """Train the autoencoder on the data."""
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        self.input_dim = X.shape[1]
        self._build_autoencoder(self.input_dim)

        # Train the autoencoder
        self.autoencoder.fit(X, X,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           shuffle=True,
                           verbose=0)

        self.fitted = True

    def transform(self, X):
        """Transform data using the trained encoder."""
        if not self.fitted:
            raise ValueError("Autoencoder must be fitted before transforming data. Call fit() first.")

        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            return self.encoder.predict(X, verbose=0).flatten()
        else:
            return self.encoder.predict(X, verbose=0)


# Factory functions for easy reducer creation
def create_pca_reducer(n_components):
    """Create a PCA reducer with specified components."""
    return PCAReducer(n_components)

def create_random_projection_reducer(n_components):
    """Create a random projection reducer with specified components."""
    return RandomProjectionReducer(n_components)

def create_identity_reducer():
    """Create an identity reducer (no dimensionality reduction)."""
    return IdentityReducer()

def create_autoencoder_reducer(n_components, hidden_layers=None, epochs=100, batch_size=32, learning_rate=0.001):
    """Create an autoencoder reducer with specified components and architecture."""
    return AutoencoderReducer(n_components, hidden_layers, epochs, batch_size, learning_rate)
