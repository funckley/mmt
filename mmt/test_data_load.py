import torch
import music_x_transformers
import json

checkpoint_path = "/Users/funckley/Documents/GitHub/mmt/exp/sod/ape/checkpoints/best_model.pt"
encoding_path = "/Users/funckley/Documents/GitHub/mmt/data/sod/processed/notes/encoding.json"

print("Testing checkpoint loading...")

try:
    # Load the encoding
    with open(encoding_path, "r") as f:
        encoding = json.load(f)
    print("Encoding loaded successfully.")

    # Define a smaller model for testing
    model = music_x_transformers.MusicXTransformer(
        dim=512,  # Match the checkpoint's embedding size
        encoding=encoding,
        depth=6,  # Match the checkpoint's depth
        heads=8,  # Match the checkpoint's number of attention heads
        max_seq_len=1024,
        max_beat=256,
        rotary_pos_emb=False,
        use_abs_pos_emb=True,
        emb_dropout=0.1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")