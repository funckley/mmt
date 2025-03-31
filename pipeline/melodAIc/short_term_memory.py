class ShortTermMemory:
    def __init__(self):
        """Initialize the short-term memory."""
        self.queue = []

    def add_to_queue(self, midi_data):
        """Add a new MIDI sequence to the queue."""
        self.queue.append(midi_data)

    def get_from_queue(self):
        """Retrieve a MIDI sequence from the queue."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def clear_queue(self):
        """Clear all MIDI sequences from the queue."""
        self.queue.clear()