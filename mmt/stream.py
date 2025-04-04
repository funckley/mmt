import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
import representation
import utils
import time

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

debug = True  # Set to True for debugging mode


def setup_osc_server(ip="0.0.0.0", port=5005):
    """
    Set up the OSC server to receive messages.
    :param ip: IP address to bind the server.
    :param port: Port to bind the server.
    :return: Configured OSC server.
    """
    dispatcher = Dispatcher()

    # Map OSC addresses to handlers
    dispatcher.map("/start", lambda addr, *args: {"type": "start", "data": args})
    dispatcher.map("/new_instruments", lambda addr, *args: {"type": "new_instruments", "data": args})
    dispatcher.map("/modified_n_beats", lambda addr, *args: {"type": "modified_n_beats", "data": args})

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"OSC server listening on {ip}:{port}")
    return server


def setup_osc_client(ip="127.0.0.1", port=5006):
    """
    Set up the OSC client to send messages.
    :param ip: IP address of the receiver.
    :param port: Port of the receiver.
    :return: Configured OSC client.
    """
    client = SimpleUDPClient(ip, port)
    print(f"OSC client set up to send messages to {ip}:{port}")
    return client

def wait_for_osc_message(osc_server):
    """
    Wait for and process incoming OSC messages.
    :param osc_server: The OSC server instance.
    :return: Parsed OSC message.
    """
    print("Waiting for OSC message...")
    osc_server.handle_request()  # Blocks until a message is received

def create_instrument_informed_tokens(instruments, sos_token, device):
    """
    Create initial tokens with the specified instruments and the correct format.
    :param instruments: List of instrument IDs.
    :param sos_token: Start-of-song token.
    :param device: The device to move the tokens to (e.g., 'cuda' or 'cpu').
    :return: Initial tokens as a torch tensor.
    """
    # Calculate the total size: <start-of-song>, instruments, <start-of-notes>
    total_size = 1 + len(instruments) + 1  # 1 for <start-of-song>, len(instruments), 1 for <start-of-notes>

    # Create a tensor for the start tokens
    tokens = torch.zeros((1, total_size, 6), dtype=torch.long, device=device)

    # Add the <start-of-song> token
    tokens[0, 0, 0] = sos_token  # Type 0 for <start-of-song>

    # Add the instrument tokens
    for i, instrument in enumerate(instruments):
        tokens[0, i + 1, 0] = 1  # Type 1 for instrument
        tokens[0, i + 1, 5] = instrument  # Instrument ID

    # Add the <start-of-notes> token
    tokens[0, len(instruments) + 1, 0] = 2  # Type 2 for <start-of-notes>

    return tokens

def extract_last_n_beats(generated_tokens, n=16, device="cpu"):
    """
    Extract the last N beats from the generated tokens.
    :param generated_tokens: List of generated token chunks.
    :param n: Number of beats to extract.
    :param device: The device to move the tokens to (e.g., 'cuda' or 'cpu').
    :return: Last N beats as a torch tensor.
    """
    all_tokens = torch.cat([torch.tensor(chunk, device=device) for chunk in generated_tokens], dim=1)
    # Filter tokens for the last N beats
    last_n_beats = all_tokens[:, -n:, :]

    # Ensure valid context for the extracted beats
    if last_n_beats.size(1) < n:
        # Pad with default values if there are not enough beats
        padding = torch.zeros((1, n - last_n_beats.size(1), 6), dtype=torch.long, device=device)
        last_n_beats = torch.cat((padding, last_n_beats), dim=1)

    # Update beat values to be incremental
    last_n_beats[0, :, 1] = torch.arange(last_n_beats.size(1), device=device)

    return last_n_beats


def update_instruments(tokens, new_instruments, device):
    """
    Update the instruments in the tokens.
    :param tokens: Tokens to update.
    :param new_instruments: List of new instrument IDs.
    :param device: The device to move the tokens to (e.g., 'cuda' or 'cpu').
    :return: Updated tokens.
    """
    # Ensure valid instrument updates
    for i, instrument in enumerate(new_instruments):
        tokens[0, i, 0] = 1  # Type 1 for instrument
        tokens[0, i, 5] = instrument  # Update the instrument ID

    # Replace any EOS tokens (type 4) with valid tokens
    for i in range(len(tokens[0])):
        if tokens[0, i, 0] == 4:  # If the token type is EOS
            tokens[0, i, 0] = 3  # Replace with a valid type (e.g., 3 for note)
            tokens[0, i, 1] = i  # Incremental beat values
            tokens[0, i, 2] = 0  # Reset position
            tokens[0, i, 3] = 60  # Default pitch
            tokens[0, i, 4] = 6  # Default duration
            tokens[0, i, 5] = 0  # Reset instrument

    # Ensure all tokens have valid beat values
    tokens[0, :, 1] = torch.arange(tokens.size(1), device=device)  # Incremental beat values

    return tokens.to(device)

def wait_for_modified_n_beats(osc_server):
    """
    Wait for the modified N-beats message from the rule-based system.
    :param osc_server: The OSC server instance.
    :return: Modified N-beats as a list or tensor.
    """
    print("Waiting for modified N-beats...")
    while True:
        message = wait_for_osc_message(osc_server)  # Wait for an OSC message
        if message["type"] == "modified_n_beats":
            print("Received modified N-beats.")
            return message["data"]  # Return the modified N-beats
        

def modify_tokens_for_n_beat_continuation(tokens, new_instruments, device):
    """
    Modify tokens for N-beat continuation and ensure the correct input format.
    Map old instrument IDs in the note tokens to the new instrument IDs.
    :param tokens: The tokens to modify.
    :param new_instruments: List of new instrument IDs.
    :param device: The device to move the tokens to (e.g., 'cuda' or 'cpu').
    :return: Modified tokens with the correct format.
    """
    # Extract only the tokens where the first index (type) is 3 (notes)
    note_tokens = tokens[0][tokens[0, :, 0] == 3]

    # Get the unique instrument IDs from the note tokens
    old_instruments = note_tokens[:, 5].unique().tolist()

    # Create a mapping from old instrument IDs to new instrument IDs
    instrument_mapping = {old: new for old, new in zip(old_instruments, new_instruments)}

    # Replace the instrument IDs in the note tokens using the mapping
    for i in range(note_tokens.size(0)):
        old_instrument = note_tokens[i, 5].item()
        if old_instrument in instrument_mapping:
            note_tokens[i, 5] = instrument_mapping[old_instrument]

    # Calculate the total size of the new input
    total_size = 1 + len(new_instruments) + 1 + note_tokens.size(0)  # <start-of-song>, instruments, <start-of-notes>, notes

    # Create the new input pattern
    tgt_start = torch.zeros((1, total_size, 6), dtype=torch.long, device=device)

    # Add the <start-of-song> token
    tgt_start[0, 0, 0] = 0  # Type 0 for <start-of-song>

    # Add the instrument tokens
    for i, instrument in enumerate(new_instruments):
        tgt_start[0, i + 1, 0] = 1  # Type 1 for instrument
        tgt_start[0, i + 1, 5] = instrument  # Instrument ID

    # Add the <start-of-notes> token
    tgt_start[0, len(new_instruments) + 1, 0] = 2  # Type 2 for <start-of-notes>

    # Add the updated note tokens
    tgt_start[0, len(new_instruments) + 2:, :] = note_tokens

    return tgt_start

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        default=50,
        type=int,
        help="number of samples to generate",
    )
    # Model
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="whether to shuffle the test data",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    # Sampling
    parser.add_argument(
        "--seq_len", default=1024, type=int, help="sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="enable streaming mode for generating MIDI data in chunks",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run the script in test mode (no OSC communication)",
    )
    return parser.parse_args(args=args, namespace=namespace)


def save_result(filename, data, sample_dir, encoding):
    """Save the results in essential formats only."""
    # Save as a numpy array (tokens)
    np.save(sample_dir / "npy" / f"{filename}.npy", data)

    # Save as a TXT file (tokens for readability)
    representation.save_txt(
        sample_dir / "txt" / f"{filename}.txt", data, encoding
    )

    # Save as a MIDI file
    music = representation.decode(data, encoding)
    music.write(sample_dir / "mid" / f"{filename}.mid")

# def save_result(data, encoding):
#     """Return the results as tokens and MIDI."""
#     # Convert tokens to MIDI
#     music = representation.decode(data, encoding)

#     # Return tokens and MIDI object
#     return data, music

def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"../data/{args.dataset}/processed/test-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"../data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"../exp/test_{args.dataset}")

    # Ensure the output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "generate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'generate-args.json'}")
    utils.save_args(args.out_dir / "generate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "npy").mkdir(exist_ok=True)
    (sample_dir / "csv").mkdir(exist_ok=True)
    (sample_dir / "txt").mkdir(exist_ok=True)
    (sample_dir / "json").mkdir(exist_ok=True)
    (sample_dir / "png").mkdir(exist_ok=True)
    (sample_dir / "mid").mkdir(exist_ok=True)
    (sample_dir / "wav").mkdir(exist_ok=True)
    (sample_dir / "mp3").mkdir(exist_ok=True)
    (sample_dir / "png-trimmed").mkdir(exist_ok=True)
    (sample_dir / "wav-trimmed").mkdir(exist_ok=True)
    (sample_dir / "mp3-trimmed").mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device("cuda" if args.gpu >= 0 else "cpu")
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )
    if debug:
        print(f"Dataset loaded successfully. Number of samples: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.info(f"Creating the model...")
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    print("Model created successfully.")  # Debug print

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    if debug:
        print(f"Loading model checkpoint from: {checkpoint_filename}")  # Debug print
    checkpoint = torch.load(checkpoint_filename, map_location=torch.device("cpu"))
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    if debug:
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")   
    # model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    if debug:
        print("Model checkpoint loaded successfully.")  # Debug print
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get special tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    # # Test mode
    # if args.test:
    #     print("Running in test mode...")
    #     generated_tokens = []
    #     token_count = 0

    #     # Simulate a "start" message
    #     instruments = [1, 48, 50]  # Example instrument IDs
    #     tgt_start = create_instrument_informed_tokens(instruments, sos, device)

    #     # Print the initial input to the model
    #     print("Initial input to the model:")
    #     print(tgt_start)

    #     while token_count < 1000:
    #         # Generate tokens in chunks
    #         for chunk in model.generate(
    #             tgt_start,
    #             args.seq_len,
    #             eos_token=eos,
    #             temperature=args.temperature,
    #             filter_logits_fn=args.filter,
    #             filter_thres=args.filter_threshold,
    #             monotonicity_dim=("type", "beat"),
    #             streaming=True,
    #             chunk_size=10,
    #         ):
    #             # Log the generated chunk
    #             chunk_np = chunk.cpu().numpy()
    #             print(f"Generated chunk: {chunk_np.tolist()}")
    #             generated_tokens.append(chunk_np)
    #             token_count += len(chunk_np)

    #             # if token_count % 200 == 0:  # Reset every 300 tokens
    #             #     print("Resetting input context to maintain diversity...")
    #             #     last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)
    #             #     tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, instruments, device)
    #             #     print(f"New input context: {tgt_start}")
    #             #     break  # Exit the inner loop to start a new `model.generate` call


    #             # Simulate an instrument change
    #             if token_count > 100:  # Example condition for instrument change
    #                 print("Simulating instrument change...")
    #                 new_instruments = [3, 25, 38]  # Example new instruments

    #                 # Extract the last generated tokens and modify them for N-beat continuation
    #                 last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)
    #                 tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, new_instruments, device)

    #                 print(f"Modified tokens for N-beat continuation: {tgt_start}")

    #                 # Print the input to the model
    #                 print("Input to the model after instrument change:")
    #                 print(tgt_start)

    #                 # Start a new model.generate(...) for N-beat continuation
    #                 for chunk in model.generate(
    #                     tgt_start,
    #                     args.seq_len,
    #                     eos_token=eos,
    #                     temperature=args.temperature,
    #                     filter_logits_fn=args.filter,
    #                     filter_thres=args.filter_threshold,
    #                     monotonicity_dim=("type", "beat"),
    #                     streaming=True,
    #                     chunk_size=10,
    #                 ):
    #                     # Log the generated chunk
    #                     chunk_np = chunk.cpu().numpy()
    #                     print(f"Generated chunk (N-beat continuation): {chunk_np.tolist()}")
    #                     generated_tokens.append(chunk_np)
    #                     token_count += len(chunk_np)

    #                 break  # Exit the inner loop after handling the instrument change

    #     print("Test mode completed.")
    #     return
    
    # Initialize the OSC server
    osc_server = setup_osc_server(ip="0.0.0.0", port=5005)  # Replace with the actual IP and port

    # Initialize the OSC client
    osc_client = setup_osc_client(ip="127.0.0.1", port=5006)  # Replace with the actual IP and port

    # Iterate over the dataset
    with torch.no_grad():
        data_iter = iter(test_loader)
        batch = next(data_iter)  # Load the first batch
        generated_tokens = []  # Store all generated tokens
        token_count = 0  # Track the total number of generated tokens

         # Test mode
        if args.test:
            print("Running in test mode...")
            generated_tokens = []
            token_count = 0

            # Simulate a "start" message
            instruments = [1, 48, 50]  # Example instrument IDs
            tgt_start = create_instrument_informed_tokens(instruments, sos, device)

            # Print the initial input to the model
            print("Initial input to the model:")
            print(tgt_start)

            while token_count < 1000:
                # Generate tokens in chunks
                for chunk in model.generate(
                    tgt_start,
                    args.seq_len,
                    eos_token=eos,
                    temperature=args.temperature,
                    filter_logits_fn=args.filter,
                    filter_thres=args.filter_threshold,
                    monotonicity_dim=("type", "beat"),
                    streaming=True,
                    chunk_size=10,
                ):
                    # Log the generated chunk
                    chunk_np = chunk.cpu().numpy()
                    print(f"Generated chunk: {chunk_np.tolist()}")
                    generated_tokens.append(chunk_np)
                    token_count += len(chunk_np)

                    # if token_count % 200 == 0:  # Reset every 300 tokens
                    #     print("Resetting input context to maintain diversity...")
                    #     last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)
                    #     tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, instruments, device)
                    #     print(f"New input context: {tgt_start}")
                    #     break  # Exit the inner loop to start a new `model.generate` call


                    # Simulate an instrument change
                    if token_count > 100:  # Example condition for instrument change
                        print("Simulating instrument change...")
                        new_instruments = [3, 25, 38]  # Example new instruments

                        # Extract the last generated tokens and modify them for N-beat continuation
                        last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)
                        tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, new_instruments, device)

                        print(f"Modified tokens for N-beat continuation: {tgt_start}")

                        # Print the input to the model
                        print("Input to the model after instrument change:")
                        print(tgt_start)

                        # Start a new model.generate(...) for N-beat continuation
                        for chunk in model.generate(
                            tgt_start,
                            args.seq_len,
                            eos_token=eos,
                            temperature=args.temperature,
                            filter_logits_fn=args.filter,
                            filter_thres=args.filter_threshold,
                            monotonicity_dim=("type", "beat"),
                            streaming=True,
                            chunk_size=10,
                        ):
                            # Log the generated chunk
                            chunk_np = chunk.cpu().numpy()
                            print(f"Generated chunk (N-beat continuation): {chunk_np.tolist()}")
                            generated_tokens.append(chunk_np)
                            token_count += len(chunk_np)

                        break  # Exit the inner loop after handling the instrument change

            print("Test mode completed.")
            return

        while True:
            # Wait for user input via OSC
            message = wait_for_osc_message(osc_server)  # Replace with your OSC message handling function

            if message.type == "start":
                # Instrument-informed generation
                instruments = message["data"]
                tgt_start = create_instrument_informed_tokens(instruments, sos, device)  # Create start tokens

                while token_count < 1100:
                    # Generate tokens in chunks
                    for chunk in model.generate(
                        tgt_start,
                        args.seq_len,
                        eos_token=eos,
                        temperature=args.temperature,
                        filter_logits_fn=args.filter,
                        filter_thres=args.filter_threshold,
                        monotonicity_dim=("type", "beat"),
                        streaming=True,
                        chunk_size=10,
                    ):
                        
                        # Send chunk via OSC
                        chunk_np = chunk.cpu().numpy()
                        osc_client.send_message("/tokens", chunk_np.tolist())
                        generated_tokens.append(chunk_np)
                        token_count += len(chunk_np)

                         # Check for instrument change
                        if osc_server.has_message("new_instruments"):
                            # Get the new instruments from the OSC message
                            new_instruments = osc_server.get_message("new_instruments")
                            print(f"Received new instruments: {new_instruments}")

                            # Extract the last generated tokens
                            last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)

                            # Modify the tokens for N-beat continuation with new instruments
                            tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, new_instruments, device)

                            print(f"Updated tgt_start with new instruments: {tgt_start}")

                            break  # Exit the inner loop to handle the instrument change

            if token_count >= 1100:
                # Switch to N-beat continuation
                last_16_beats = extract_last_n_beats(generated_tokens, n=16, device=device)
                tgt_start = last_16_beats

                while token_count < 1200:
                    # Generate tokens in chunks
                    for chunk in model.generate(
                        tgt_start,
                        args.seq_len,
                        eos_token=eos,
                        temperature=args.temperature,
                        filter_logits_fn=args.filter,
                        filter_thres=args.filter_threshold,
                        monotonicity_dim=("type", "beat"),
                        streaming=True,
                        chunk_size=10,
                    ):
                        # Send chunk via OSC
                        chunk_np = chunk.cpu().numpy()
                        osc_client.send_message("/tokens", chunk_np.tolist())
                        generated_tokens.append(chunk_np)
                        token_count += len(chunk_np)

                        
                        # Check for instrument change
                        if osc_server.has_message("new_instruments"):
                            # Get the new instruments from the OSC message
                            new_instruments = osc_server.get_message("new_instruments")
                            print(f"Received new instruments: {new_instruments}")

                            # Extract the last generated tokens
                            last_tokens = extract_last_n_beats(generated_tokens, n=16, device=device)

                            # Modify the tokens for N-beat continuation with new instruments
                            tgt_start = modify_tokens_for_n_beat_continuation(last_tokens, new_instruments, device)

                            print(f"Updated tgt_start with new instruments: {tgt_start}")

                            break  # Exit the inner loop to handle the instrument change


if __name__ == "__main__":
    main()