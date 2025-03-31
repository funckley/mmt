class LongBuffer:
    def __init__(self):
        """Initialize the buffer to store MIDI sequences."""
        self.buffer = []

    def add_to_buffer(self, midi_data):
        """Add a new MIDI sequence to the buffer.
        
        Args:
            midi_data: The MIDI sequence to add to the buffer.
        """
        self.buffer.append(midi_data)

    def get_from_buffer(self):
        """Retrieve a MIDI sequence from the buffer.
        
        Returns:
            The next MIDI sequence from the buffer, or None if the buffer is empty.
        """
        if self.buffer:
            return self.buffer.pop(0)
        return None

    def clear_buffer(self):
        """Clear all MIDI sequences from the buffer."""
        self.buffer.clear()