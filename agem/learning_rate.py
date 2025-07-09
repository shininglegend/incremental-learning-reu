# This code gives the learning rate scheduler for TA-A-GEM method.
# It is not required per se, but it can be helpful.


class TALearningRateScheduler:
    """
    Learning rate scheduler for TA-* method.

    Lowers learning rate when loss plateaus and resets to initial value
    when loss spikes significantly above best observed loss.
    """

    def __init__(
        self,
        lr_init: float = 1e-3,
        factor: float = 0.9999,
        min_lr: float = 1e-5,
        patience: int = 5,
        threshold: float = 1e-4,
        reset_threshold: float = 1.0,
    ):
        """
        Initialize the learning rate scheduler.

        Args:
            lr_init: Initial learning rate
            factor: Factor to multiply LR when stagnant (< 1)
            min_lr: Minimum learning rate threshold
            patience: Number of stagnant/spike steps before action
            threshold: Relative improvement threshold for stagnation
            reset_threshold: Absolute spike threshold for reset
        """
        self.lr_init = lr_init
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.threshold = threshold
        self.reset_threshold = reset_threshold

        # State variables
        self.current_lr = lr_init
        self.best_loss = float("inf")
        self.n_stagnant = 0
        self.n_spike = 0

    def step(self, loss: float) -> float:
        """
        Update learning rate based on current loss value.

        Args:
            loss: Current loss value

        Returns:
            Updated learning rate
        """
        # Check for improvement
        if loss < self.best_loss * (1 - self.threshold):
            self.best_loss = loss
            self.n_stagnant = 0
        else:
            self.n_stagnant += 1

        # Check for spike
        if loss > self.best_loss + self.reset_threshold:
            self.n_spike += 1
        else:
            self.n_spike = 0

        # Reset LR on sustained spike
        if self.n_spike > self.patience:
            self.current_lr = self.lr_init
            self.best_loss = loss
            self.n_spike = 0

        # Reduce LR on stagnation
        if self.n_stagnant > self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.n_stagnant = 0

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.lr_init
        self.best_loss = float("inf")
        self.n_stagnant = 0
        self.n_spike = 0
