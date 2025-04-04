import numpy as np

class TokenBuffer:
    def __init__(self):
        """Initialize an empty buffer to store tokens."""
        self.buffer = []

    def add_tokens(self, tokens):
        """
        Add a batch of tokens to the buffer.
        :param tokens: A numpy array of tokens (shape: [batch_size, seq_len, 6]).
        """
        self.buffer.append(tokens)

    def save_all(self, filepath):
        """
        Save all tokens (as received) to a file.
        :param filepath: Path to save the tokens.
        """
        all_tokens = np.concatenate(self.buffer, axis=0)  # Combine all batches
        np.save(filepath, all_tokens)
        print(f"All tokens saved to {filepath}")

    def save_notes_only(self, filepath):
        """
        Save only the tokens of type 3 (Note) to a file.
        :param filepath: Path to save the filtered tokens.
        """
        all_tokens = np.concatenate(self.buffer, axis=0)  # Combine all batches
        notes_only = all_tokens[all_tokens[:, :, 0] == 3]  # Filter tokens of type 3
        np.save(filepath, notes_only)
        print(f"Note tokens saved to {filepath}")