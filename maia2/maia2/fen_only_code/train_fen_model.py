import argparse
import os
from multiprocessing import Process, Queue, cpu_count
import time
from .utils import seed_everything, readable_time, readable_num, count_parameters
from .utils import get_all_possible_moves, create_elo_dict
from .utils import decompress_zst, read_or_create_chunks
# from .main import MAIA2Model, preprocess_thread, train_chunks, read_monthly_data_path
from .fen_model import MAIA2Transformer, preprocess_thread, train_chunks, read_monthly_data_path
import torch
import torch.nn as nn
import pdb
import random


def run(cfg):
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    num_processes = cpu_count() - cfg.num_cpu_left

    save_root = f'../fen_transformer_saves/{cfg.lr}_{cfg.batch_size}_{cfg.wd}_{cfg.num_blocks_vit}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Transformer(len(all_moves), cfg)
    print(model, flush=True)
    model = model.cuda()
    model = nn.DataParallel(model)

    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()
    criterion_move_time = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N_params = count_parameters(model)
    print(f'Trainable Parameters: {N_params}', flush=True)

    accumulated_samples = 0
    accumulated_games = 0

    # Optional checkpoint loading
    if cfg.from_checkpoint:
        formatted_month = f"{cfg.checkpoint_month:02d}"
        checkpoint_path = os.path.join(
            save_root, f'epoch_{cfg.checkpoint_epoch}_{cfg.checkpoint_year}-{formatted_month}.pgn.pt'
        )
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        accumulated_samples = checkpoint['accumulated_samples']
        accumulated_games = checkpoint['accumulated_games']

    # ----------------------------------------------------------
    # Dictionary: (pgn_path, num_chunk) -> a random seed (int)
    # so we pick the same validation subset across epochs.
    # ----------------------------------------------------------
    chunk_seeds_dict = {}

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}', flush=True)
        pgn_paths = read_monthly_data_path(cfg)
        
        num_file = 0
        for pgn_path in pgn_paths:
            start_time = time.time()
            decompress_zst(pgn_path + '.zst', pgn_path)
            print(f'Decompressing {pgn_path} took {readable_time(time.time() - start_time)}', flush=True)

            pgn_chunks = read_or_create_chunks(pgn_path, cfg)
            print(f'Training {pgn_path} with {len(pgn_chunks)} chunks.', flush=True)
            
            queue = Queue(maxsize=cfg.queue_length)
            
            # Split chunks into sublists of size 'num_processes'
            pgn_chunks_sublists = []
            for i in range(0, len(pgn_chunks), num_processes):
                pgn_chunks_sublists.append(pgn_chunks[i : i + num_processes])
            
            # Start first worker
            pgn_chunks_sublist = pgn_chunks_sublists[0]
            worker = Process(
                target=preprocess_thread, 
                args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict)
            )
            worker.start()
            
            num_chunk = 0
            offset = 0

            while True:
                if not queue.empty():
                    # Launch next worker if any sublists remain
                    if offset + 1 < len(pgn_chunks_sublists):
                        pgn_chunks_sublist = pgn_chunks_sublists[offset + 1]
                        worker = Process(
                            target=preprocess_thread, 
                            args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict)
                        )
                        worker.start()
                        offset += 1

                    data, game_count, chunk_count = queue.get()

                    # Key to identify which chunk this is
                    chunk_id = (pgn_path, num_chunk)

                    # If we haven't assigned a seed for this chunk yet, do so
                    if chunk_id not in chunk_seeds_dict:
                        chunk_seeds_dict[chunk_id] = random.getrandbits(64)  # 64-bit seed

                    # Retrieve the seed for this chunk
                    seed_for_chunk = chunk_seeds_dict[chunk_id]

                    # Use that seed to pick the same random 10% across epochs
                    random.seed(seed_for_chunk)
                    val_size = int(0.1 * len(data))
                    val_indices = set(random.sample(range(len(data)), val_size))

                    data_val = [data[i] for i in val_indices]
                    data_train = [data[i] for i in range(len(data)) if i not in val_indices]

                    # Train + validate
                    (
                        loss,
                        loss_maia,
                        loss_side_info,
                        loss_value,
                        loss_move_time,
                        val_loss,
                        val_loss_maia,
                        val_loss_side_info,
                        val_loss_value,
                        val_loss_move_time,
                        val_accuracy
                    ) = train_chunks(
                        cfg,
                        data_train,
                        data_val,
                        model,
                        optimizer,
                        all_moves_dict,
                        criterion_maia,
                        criterion_side_info,
                        criterion_value,
                        criterion_move_time
                    )

                    num_chunk += chunk_count
                    accumulated_samples += len(data)
                    accumulated_games += game_count

                    print(f'[{num_chunk}/{len(pgn_chunks)}]', flush=True)
                    print(f'[# Positions]: {readable_num(accumulated_samples)}', flush=True)
                    print(f'[# Games]: {readable_num(accumulated_games)}', flush=True)

                    # Print training losses
                    print(
                        f"[# Train Loss]: {loss} | "
                        f"[# Train Loss MAIA]: {loss_maia} | "
                        f"[# Train Loss Side Info]: {loss_side_info} | "
                        f"[# Train Loss Value]: {loss_value} | "
                        f"[# Train Loss Time]: {loss_move_time}",
                        flush=True
                    )

                    # Print validation losses + accuracy
                    print(
                        f"[# Val Loss]: {val_loss} | "
                        f"[# Val Loss MAIA]: {val_loss_maia} | "
                        f"[# Val Loss Side Info]: {val_loss_side_info} | "
                        f"[# Val Loss Value]: {val_loss_value} | "
                        f"[# Val Loss Time]: {val_loss_move_time} | "
                        f"[# Val Accuracy]: {val_accuracy}",
                        flush=True
                    )

                    # If all chunks are processed, break
                    if num_chunk == len(pgn_chunks):
                        break

            num_file += 1
            elapsed_str = readable_time(time.time() - start_time)
            print(
                f'### ({num_file} / {len(pgn_paths)}) Took {elapsed_str} to train '
                f'{pgn_path} with {len(pgn_chunks)} chunks.', 
                flush=True
            )

            # Remove the decompressed PGN to save space
            os.remove(pgn_path)
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accumulated_samples': accumulated_samples,
                'accumulated_games': accumulated_games
            }, f'{save_root}epoch_{epoch + 1}_{pgn_path[-11:]}.pt')
