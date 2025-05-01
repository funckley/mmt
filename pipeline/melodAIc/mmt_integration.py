import mido
import json
import os
import time
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
from queue import Queue

from representation import decode_notes, load_encoding


#########################################################################################################################################################
# DECODING & MIDI
#########################################################################################################################################################
tempo = 120  # Default tempo in BPM
velocity = 64  # Default velocity (0–127)

# Token queue for buffering tokens
token_queue = Queue()

# Consumer thread to process tokens from the queue
def token_consumer():
    while True:
        tokens = token_queue.get()
        if tokens is None:
            break
        notes = decode_notes(tokens, encoding)
        notes_to_midi(notes, resolution)

# def token_consumer():
#     while True:
#         chunks = token_queue.get()
#         if chunks is None:
#             break
#         print(f"Chunks received for decoding: {chunks}")  # Debugging input
#         # Decode all notes in the chunks
#         notes = decode_notes(chunks, encoding)
#         print(f"Decoded notes: {notes}")
#         # Send the notes to MIDI playback
#         notes_to_midi(notes, resolution, tempo=120)

# Start the consumer thread
consumer_thread = threading.Thread(target=token_consumer)
consumer_thread.daemon = True  # Ensure the thread exits when the main program exits
consumer_thread.start()

encoding_file = "encoding.json"
encoding = load_encoding(encoding_file)

# Get the resolution (ticks per beat) from the encoding
resolution = encoding.get("resolution", 12)  # Default to 12 if not specified

print("Available MIDI output ports:")
for port in mido.get_output_names():
    print(port)

midi_output = mido.open_output("IAC Driver PythonMIDIOut")  # Replace with your virtual MIDI port name

# def notes_to_midi(notes, resolution, tempo=120):
#     """
#     Convert notes into MIDI messages and send them in real-time.
#     :param notes: List of notes (beat, position, pitch, duration, program).
#     :param resolution: Ticks per beat (e.g., 12).
#     :param tempo: Tempo in beats per minute.
#     """
#     seconds_per_tick = 60 / (tempo * resolution)  # Convert ticks to seconds

#     for note in notes:
#         beat, position, pitch, duration, program = note

#         # Calculate the start time and duration in seconds
#         start_time = (beat * resolution + position) * seconds_per_tick
#         duration_in_seconds = duration * seconds_per_tick

#         # Send a program change message for the instrument
#         program_change = mido.Message('program_change', program=program)
#         midi_output.send(program_change)

#         # Send a note_on message
#         note_on = mido.Message('note_on', note=pitch, velocity=64)
#         midi_output.send(note_on)
#         print(f"Sent MIDI note_on: {note_on}")

#         # Wait for the duration of the note
#         time.sleep(duration_in_seconds)

#         # Send a note_off message
#         note_off = mido.Message('note_off', note=pitch, velocity=64)
#         midi_output.send(note_off)
#         print(f"Sent MIDI note_off: {note_off}")






# def notes_to_midi(notes, resolution):
#     """
#     Convert notes into MIDI messages and send them in real-time.
#     :param notes: List of notes (beat, position, pitch, duration, program).
#     :param resolution: Ticks per beat (e.g., 12).
#     """
#     global tempo  # Use the global tempo variable for dynamic tempo control
#     global velocity  # Use the global velocity variable for dynamic velocity control

#     for note in notes:
#         beat, position, pitch, duration, program = note

#         # Calculate the start time and duration in seconds based on the current tempo
#         seconds_per_tick = 60 / (tempo * resolution)  # Convert ticks to seconds
#         start_time = (beat * resolution + position) * seconds_per_tick
#         duration_in_seconds = duration * seconds_per_tick

#         # Send a program change message for the instrument
#         program_change = mido.Message('program_change', program=program)
#         midi_output.send(program_change)

#         # Send a note_on message with the current velocity
#         note_on = mido.Message('note_on', note=pitch, velocity=velocity)
#         midi_output.send(note_on)
#         print(f"Sent MIDI note_on: {note_on}")

#         # Wait for the duration of the note
#         time.sleep(duration_in_seconds)

#         # Send a note_off message
#         note_off = mido.Message('note_off', note=pitch, velocity=64)
#         midi_output.send(note_off)
#         print(f"Sent MIDI note_off: {note_off}")




def notes_to_midi(notes, resolution):
    """
    Convert notes into MIDI messages and send them in real-time.
    :param notes: List of notes (beat, position, pitch, duration, program).
    :param resolution: Ticks per beat (e.g., 12).
    """
    global tempo  # Use the global tempo variable for dynamic tempo control
    global velocity  # Use the global velocity variable for dynamic velocity control

    # Group notes by (beat, position)
    grouped_notes = {}
    for note in notes:
        beat, position, pitch, duration, program = note
        key = (beat, position)
        if key not in grouped_notes:
            grouped_notes[key] = []
        grouped_notes[key].append((pitch, duration, program))

    # Sort groups by beat and position
    sorted_groups = sorted(grouped_notes.items(), key=lambda x: (x[0][0], x[0][1]))

    for (beat, position), group in sorted_groups:
        # Calculate the start time and duration in seconds based on the current tempo
        seconds_per_tick = 60 / (tempo * resolution)  # Convert ticks to seconds
        start_time = (beat * resolution + position) * seconds_per_tick
        duration_in_seconds = max(note[1] * seconds_per_tick for note in group)  # Longest duration in the group

        # Send program change messages for all instruments in the group
        for pitch, duration, program in group:
            program_change = mido.Message('program_change', program=program)
            midi_output.send(program_change)

        # Send note_on messages for all notes in the group
        for pitch, duration, program in group:
            note_on = mido.Message('note_on', note=pitch, velocity=velocity)
            midi_output.send(note_on)
            print(f"Sent MIDI note_on: {note_on}")

        # Wait for the duration of the notes
        time.sleep(duration_in_seconds)

        # Send note_off messages for all notes in the group
        for pitch, duration, program in group:
            note_off = mido.Message('note_off', note=pitch, velocity=64)
            midi_output.send(note_off)
            print(f"Sent MIDI note_off: {note_off}")





# def notes_to_midi(notes, resolution, tempo=120):
#     """
#     Convert notes into MIDI messages and send them in real-time.
#     :param notes: List of notes (beat, position, pitch, duration, program).
#     :param resolution: Ticks per beat (e.g., 12).
#     :param tempo: Tempo in beats per minute.
#     """
#     seconds_per_tick = 60 / (tempo * resolution)  # Convert ticks to seconds

#     # Create a new list of notes with calculated start_time
#     notes_with_time = []
#     for note in notes:
#         beat, position, pitch, duration, program = note
#         start_time = (beat * resolution + position) * seconds_per_tick
#         duration_in_seconds = duration * seconds_per_tick
#         notes_with_time.append((beat, position, pitch, duration, program, start_time))

#     # Sort notes by start_time
#     notes_with_time.sort(key=lambda x: x[-1])  # Sort by the appended start_time

#     # Send MIDI messages in real-time
#     start_time_reference = time.time()  # Reference time for real-time playback
#     for note in notes_with_time:
#         beat, position, pitch, duration, program, start_time = note
#         duration_in_seconds = duration * seconds_per_tick

#         # Wait until the correct time to send the note
#         current_time = time.time()
#         time_to_wait = start_time - (current_time - start_time_reference)
#         if time_to_wait > 0:
#             time.sleep(time_to_wait)

#         # Send a program change message for the instrument
#         program_change = mido.Message('program_change', program=program)
#         midi_output.send(program_change)

#         # Send a note_on message
#         note_on = mido.Message('note_on', note=pitch, velocity=64)
#         midi_output.send(note_on)
#         print(f"Sent MIDI note_on: {note_on}")

#         # Wait for the duration of the note
#         time.sleep(duration_in_seconds)
#         # time.sleep(0)

#         # Send a note_off message
#         note_off = mido.Message('note_off', note=pitch, velocity=64)
#         midi_output.send(note_off)
#         print(f"Sent MIDI note_off: {note_off}")

# def notes_to_midi(notes, resolution, tempo=120):
#     """
#     Convert notes into MIDI messages and send them in real-time.
#     :param notes: List of notes (beat, position, pitch, duration, program).
#     :param resolution: Ticks per beat (e.g., 12).
#     :param tempo: Tempo in beats per minute.
#     """
#     seconds_per_tick = 60 / (tempo * resolution)  # Convert ticks to seconds

#     # Create a new list of notes with calculated start_time
#     notes_with_time = []
#     for note in notes:
#         beat, position, pitch, duration, program = note
#         start_time = (beat * resolution + position) * seconds_per_tick
#         duration_in_seconds = duration * seconds_per_tick
#         notes_with_time.append((beat, position, pitch, duration, program, start_time))

#     # Sort notes by start_time
#     notes_with_time.sort(key=lambda x: x[-1])  # Sort by the appended start_time

#     # Send MIDI messages in real-time
#     start_time_reference = time.time()  # Reference time for real-time playback
#     for note in notes_with_time:
#         beat, position, pitch, duration, program, start_time = note
#         duration_in_seconds = duration * seconds_per_tick

#         # Get the MIDI channel for the instrument
#         instrument_name = next(
#             (name for name, code in instrument_code_map.items() if code == program), "null"
#         )
#         midi_channel = instrument_to_channel.get(instrument_name, 1) - 1  # Convert to 0-based channel

#         # Wait until the correct time to send the note
#         current_time = time.time()
#         time_to_wait = start_time - (current_time - start_time_reference)
#         if time_to_wait > 0:
#             time.sleep(time_to_wait)

#         # Send a program change message for the instrument
#         program_change = mido.Message('program_change', program=program, channel=midi_channel)
#         midi_output.send(program_change)

#         # Send a note_on message
#         note_on = mido.Message('note_on', note=pitch, velocity=64, channel=midi_channel)
#         midi_output.send(note_on)
#         # print(f"Sent MIDI note_on: {note_on}")

#         # Wait for the duration of the note
#         time.sleep(duration_in_seconds)

#         # Send a note_off message
#         note_off = mido.Message('note_off', note=pitch, velocity=64, channel=midi_channel)
#         midi_output.send(note_off)
#         # print(f"Sent MIDI note_off: {note_off}")

#########################################################################################################################################################
# OSC and MMT INTEGRATION
#########################################################################################################################################################

# Configuration
STREAM_DEVICE_IP = "184.105.238.175"  # Replace with the IP of the device running stream.py
STREAM_DEVICE_PORT = 5005          # Port where stream.py's OSC server is listening
LOCAL_IP = "35.3.44.33"         # Replace with the IP of this device
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
    52: "violin",              # MIDI note 52 → brasses
    53: "cello",              # MIDI note 53 → voices
    55: "voices",  # MIDI note 55 → nylon-string-guitar
    57: "nylon-string-guitar",               # MIDI note 57 → violin
    59: "strings",                # MIDI note 59 → cello
    60: "flute"                 # MIDI note 60 → flute
}

# Map instruments to MIDI channels (1-16)
instrument_to_channel = {
    "piano": 1,
    "clarinet": 2,
    "violin": 3,
    "cello": 4,
    "voices": 5,
    "nylon-string-guitar": 6,
    "strings": 7,
    "flute": 8,
    # Add more instruments as needed
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

    # Define a handler for the /tokens message
    def tokens_handler(address, *args):
        print(f"Received tokens: {args}")

    # Map OSC addresses to handlers
    dispatcher.map("/tokens", handle_tokens)  # Handle generated tokens from stream.py

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC server listening on {ip}:{port}")
    return server

# Handler for incoming tokens
def handle_tokens(address, *args):
    print(f"Received tokens from {address}: {args}")
    # Process the tokens (e.g., update buffer, apply rule-based logic)

    tokens = args[0]  # Assuming tokens are sent as the first argument

    # Ensure tokens are wrapped in a list if necessary
    # if not isinstance(tokens[0], list):
    #     tokens = [tokens]

    # Add tokens to the queue for processing
    token_queue.put(tokens)

    # # Decode the tokens into notes
    # notes = decode_notes(tokens, encoding)  # Use the imported function
    # print(f"Decoded notes: {notes}")

    # # Further processing or sending MIDI messages
    # notes_to_midi(notes, resolution, tempo=120)  # Convert notes to MIDI


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
    global tempo  # Use the global tempo variable
    global velocity  # Use the global velocity variable

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
            # osc_client.send_message("/start", [])
            # print("Sent /start message to stream.py")
            active_instruments = [
                instrument_code_map[name]
                for name, is_on in instrument_state.items() if is_on == 1
            ]

            if not active_instruments:
                print("Error: You must activate at least one instrument before starting.")
                return

            # Send the /start message with the active instruments
            osc_client.send_message("/start", active_instruments)
            print(f"Sent /start message with instruments: {active_instruments}")

    elif message.type == 'control_change' and message.control == 82:
        # Map the slider value (0–127) to a tempo range (e.g., 40–200 BPM)
        min_tempo = 40
        max_tempo = 300
        tempo = int(min_tempo + (message.value / 127) * (max_tempo - min_tempo))

        print(f"Tempo updated to: {tempo} BPM")
    
    # Handle velocity control (slider on control 83)
    elif message.type == 'control_change' and message.control == 83:
        # Map the slider value (0–127) to a velocity range (0–127)
        velocity = int(message.value)
        print(f"Velocity updated to: {velocity}")

# Example in mmt_integration.py
def send_start_message(osc_client, active_instruments):
    if not active_instruments:
        print("Error: You must activate at least one instrument before starting.")
        return

    # Send the /start message with the active instruments
    osc_client.send_message("/start", active_instruments)
    print(f"Sent /start message with instruments: {active_instruments}")




# Main function
def main():
    # Set up OSC client
    osc_client = setup_osc_client(STREAM_DEVICE_IP, STREAM_DEVICE_PORT)


    # Set up OSC server in a separate thread
    osc_server = setup_osc_server(LOCAL_IP, LOCAL_PORT)
    osc_thread = threading.Thread(target=osc_server.serve_forever)
    osc_thread.daemon = True  # Ensure the thread exits when the main program exits
    osc_thread.start()
    print("OSC server is running in a separate thread.")

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