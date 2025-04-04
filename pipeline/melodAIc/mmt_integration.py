import mido
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# Configuration
STREAM_DEVICE_IP = "192.168.1.100"  # Replace with the IP of the device running stream.py
STREAM_DEVICE_PORT = 5005          # Port where stream.py's OSC server is listening
LOCAL_IP = "192.168.1.101"         # Replace with the IP of this device
LOCAL_PORT = 5006                  # Port where this device's OSC server will listen

# Instrument code map
instrument_code_map = {
    "piano": 1,
    "electric-piano": 2,
    "harpsichord": 3,
    "clavinet": 4,
    "celesta": 5,
    "glockenspiel": 6,
    "music-box": 7,
    "vibraphone": 8,
    "marimba": 9,
    "xylophone": 10,
    "tubular-bells": 11,
    "dulcimer": 12,
    "organ": 13,
    "church-organ": 14,
    "accordion": 15,
    "harmonica": 16,
    "bandoneon": 17,
    "nylon-string-guitar": 18,
    "steel-string-guitar": 19,
    "electric-guitar": 20,
    "bass": 21,
    "electric-bass": 22,
    "slap-bass": 23,
    "synth-bass": 24,
    "violin": 25,
    "viola": 26,
    "cello": 27,
    "contrabass": 28,
    "harp": 29,
    "timpani": 30,
    "strings": 31,
    "synth-strings": 32,
    "voices": 33,
    "orchestra-hit": 34,
    "trumpet": 35,
    "trombone": 36,
    "tuba": 37,
    "horn": 38,
    "brasses": 39,
    "synth-brasses": 40,
    "soprano-saxophone": 41,
    "alto-saxophone": 42,
    "tenor-saxophone": 43,
    "baritone-saxophone": 44,
    "oboe": 45,
    "english-horn": 46,
    "bassoon": 47,
    "clarinet": 48,
    "piccolo": 49,
    "flute": 50,
    "recorder": 51,
    "pan-flute": 52,
    "ocarina": 53,
    "lead": 54,
    "pad": 55,
    "sitar": 56,
    "banjo": 57,
    "shamisen": 58,
    "koto": 59,
    "kalimba": 60,
    "bag-pipe": 61,
    "shehnai": 62,
    "melodic-tom": 63,
    "synth-drums": 64,
    "null": 0
}

# Map MIDI notes to instruments (specific to your MIDI controller)
midi_note_to_instrument = {
    48: "piano",                # MIDI note 48 → piano
    50: "clarinet",             # MIDI note 50 → clarinet
    52: "brasses",              # MIDI note 52 → brasses
    53: "marimba",              # MIDI note 53 → marimba
    55: "nylon-string-guitar",  # MIDI note 55 → nylon-string-guitar
    57: "violin",               # MIDI note 57 → violin
    59: "cello",                # MIDI note 59 → cello
    60: "flute"                 # MIDI note 60 → flute
}

# Special MIDI note to send the `/new_instruments` message
SEND_MESSAGE_NOTE = 72  # C5: Send the `/new_instruments` message
START_MESSAGE_NOTE = 71  # B4: Send the `/start` message

# Instrument state (0 = off, 1 = on)
instrument_state = {name: 0 for name in instrument_code_map.keys()}

# Set up the OSC client to send messages to stream.py
def setup_osc_client(ip, port):
    client = SimpleUDPClient(ip, port)
    print(f"OSC client set up to send messages to {ip}:{port}")
    return client

# Set up the OSC server to receive messages from stream.py
def setup_osc_server(ip, port):
    dispatcher = Dispatcher()

    # Map OSC addresses to handlers
    dispatcher.map("/tokens", handle_tokens)  # Handle generated tokens from stream.py

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC server listening on {ip}:{port}")
    return server

# Handler for incoming tokens
def handle_tokens(address, *args):
    print(f"Received tokens from {address}: {args}")
    # Process the tokens (e.g., update buffer, apply rule-based logic)


# Toggle instrument state
def toggle_instrument(osc_client, instrument_name):
    if instrument_name not in instrument_state:
        print(f"Instrument '{instrument_name}' not found.")
        return

    # Toggle the instrument state
    instrument_state[instrument_name] = 1 - instrument_state[instrument_name]
    state = instrument_state[instrument_name]
    print(f"Toggled instrument '{instrument_name}' to state {state}.")

    # Get the list of active instruments (numbers for OSC, names for printing)
    active_instruments = [
        instrument_code_map[name]
        for name, is_on in instrument_state.items() if is_on == 1
    ]
    active_instrument_names = [
        name for name, is_on in instrument_state.items() if is_on == 1
    ]

    # Print the list of active instrument names
    print(f"Active instruments: {active_instrument_names}")

    return active_instruments

# Update instrument state and send new instruments
def update_instruments(osc_client, instrument_name, state):
    if instrument_name not in instrument_state:
        print(f"Instrument '{instrument_name}' not found.")
        return

    # Update the instrument state
    instrument_state[instrument_name] = state
    print(f"Updated instrument '{instrument_name}' to state {state}.")

    # Get the list of active instruments
    active_instruments = [
        instrument_code_map[name]
        for name, is_on in instrument_state.items() if is_on == 1
    ]

    # Send the updated instruments to stream.py
    osc_client.send_message("/new_instruments", active_instruments)
    print(f"Sent /new_instruments message with instruments: {active_instruments}")

# Handle MIDI messages
def handle_midi_message(osc_client, message):
    if message.type == 'note_on' and message.velocity > 0:
        if message.note in midi_note_to_instrument:
            # Toggle the corresponding instrument
            instrument_name = midi_note_to_instrument[message.note]
            toggle_instrument(osc_client, instrument_name)
        elif message.note == SEND_MESSAGE_NOTE:
            # Send the `/new_instruments` message
            active_instruments = [
                instrument_code_map[name]
                for name, is_on in instrument_state.items() if is_on == 1
            ]
            osc_client.send_message("/new_instruments", active_instruments)
            print(f"Sent /new_instruments message with instruments: {active_instruments}")
        elif message.note == START_MESSAGE_NOTE:
            # Send the `/start` message
            osc_client.send_message("/start", [])
            print("Sent /start message to stream.py")


# Main function
def main():
    # Set up OSC client
    osc_client = setup_osc_client(STREAM_DEVICE_IP, STREAM_DEVICE_PORT)

    # List available MIDI devices
    midi_devices = mido.get_input_names()
    print("Available MIDI devices:")
    for idx, device in enumerate(midi_devices):
        print(f"{idx}: {device}")

    # Ask the user to select a MIDI device by index
    while True:
        try:
            midi_input_index = int(input("Enter the index of your MIDI device: "))
            if 0 <= midi_input_index < len(midi_devices):
                break
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    midi_input_name = midi_devices[midi_input_index]
    print(f"Selected MIDI device: {midi_input_name}")

    # Open MIDI input
    with mido.open_input(midi_input_name) as midi_input:
        print(f"Listening for MIDI messages on {midi_input_name}...")

        # Listen for MIDI messages
        for message in midi_input:
            print(f"Received MIDI message: {message}")
            handle_midi_message(osc_client, message)

if __name__ == "__main__":
    main()