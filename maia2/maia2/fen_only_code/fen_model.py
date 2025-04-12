import chess.pgn
import chess
import signal
import pdb
from multiprocessing import Pool, cpu_count, Queue, Process
import torch
import tqdm
import argparse
from .utils import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map
import os
import pandas as pd
import time
import sys
from einops import rearrange
from . import train_fen_model

def process_chunks(cfg, pgn_path, pgn_chunks, elo_dict):
    # process_per_chunk((pgn_chunks[0][0], pgn_chunks[0][1], pgn_path, elo_dict, cfg))

    if cfg.verbose:
        results = process_map(process_per_chunk, [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks],
                              max_workers=len(pgn_chunks), chunksize=1)
    else:
        with Pool(processes=len(pgn_chunks)) as pool:
            results = pool.map(process_per_chunk, [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks])

    ret = []
    count = 0
    list_of_dicts = []
    for result, game_count, frequency in results:
        ret.extend(result)
        count += game_count
        list_of_dicts.append(frequency)

    total_counts = {}

    for d in list_of_dicts:
        for key, value in d.items():
            total_counts[key] = total_counts.get(key, 0) + value

    # print(total_counts, flush=True)

    return ret, count, len(pgn_chunks)


def process_per_game(game, white_elo, black_elo, white_win, cfg):
    ret = []

    board = game.board()
    moves = list(game.mainline_moves())
    time_control = parse_time_control(game.headers.get('TimeControl', '1000+10'))

    mainline = []

    for i, node in enumerate(game.mainline()):
        mainline.append(node)


    for i, node in enumerate(mainline):

        move = moves[i]

        if i >= cfg.first_n_moves:

            comment = node.comment
            clock_info = extract_clock_time(comment)

            if i+2 < len(mainline):
                next_comment = mainline[i+2].comment
                clock_delta = clock_info - extract_clock_time(next_comment)
            
            else:
                clock_delta = clock_info

            if i % 2 == 0:
                board_input = board.fen()
                move_input = move.uci()
                elo_self = white_elo
                elo_oppo = black_elo
                active_win = white_win

            else:
                board_input = board.fen()
                move_input = move.uci()
                elo_self = black_elo
                elo_oppo = white_elo
                active_win = - white_win

            if clock_info > cfg.clock_threshold:
                ret.append((board_input, move_input, elo_self, elo_oppo, active_win, time_control, clock_info, clock_delta))

        board.push(move)
        if i == cfg.max_ply:
            break

    return ret


def game_filter(game):
    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    time_control = game.headers.get("TimeControl", "?")
    result = game.headers.get("Result", "?")
    event = game.headers.get("Event", "?")

    if white_elo == "?" or black_elo == "?" or time_control == "?" or result == "?" or event == "?":
        return

    if 'Rated' not in event:
        return

    if 'Rapid' not in event:
        return

    for _, node in enumerate(game.mainline()):
        if 'clk' not in node.comment:
            return

    white_elo = int(white_elo)
    black_elo = int(black_elo)

    if result == '1-0':
        white_win = 1
    elif result == '0-1':
        white_win = -1
    elif result == '1/2-1/2':
        white_win = 0
    else:
        return

    return game, white_elo, black_elo, white_win

def terminate_all():
    """
    Forcefully terminate the entire script and all subprocesses.
    """
    print("Terminating all processes...", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)  # Send SIGTERM to the entire process tree

def process_per_chunk(args):
    start_pos, end_pos, pgn_path, elo_dict, cfg = args

    ret = []
    game_count = 0

    frequency = {}

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:

        pgn_file.seek(start_pos)

        while pgn_file.tell() < end_pos:

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            filtered_game = game_filter(game)
            if filtered_game:
                game, white_elo, black_elo, white_win = filtered_game
                white_elo = extract_elo(white_elo)
                black_elo = extract_elo(black_elo)

                if white_elo < black_elo:
                    range_1, range_2 = map_to_category(black_elo, elo_dict), map_to_category(white_elo, elo_dict)
                else:
                    range_1, range_2 = map_to_category(white_elo, elo_dict), map_to_category(black_elo, elo_dict)

                freq = frequency.get((range_1, range_2), 0)
                if freq >= cfg.max_games_per_elo_range:
                    continue

                ret_per_game = process_per_game(game, white_elo, black_elo, white_win, cfg)
                ret.extend(ret_per_game)
                if len(ret_per_game):

                    if (range_1, range_2) in frequency:
                        frequency[(range_1, range_2)] += 1
                    else:
                        frequency[(range_1, range_2)] = 1

                    game_count += 1

    return ret, game_count, frequency


class MAIA1Dataset(torch.utils.data.Dataset):

    def __init__(self, data, all_moves_dict, elo_dict, cfg):

        self.all_moves_dict = all_moves_dict
        self.cfg = cfg
        self.data = data.values.tolist()
        self.elo_dict = elo_dict

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        fen, move, elo_self, elo_oppo, white_active = self.data[idx]

        if white_active:
            board = chess.Board(fen)
        else:
            board = chess.Board(fen).mirror()
            move = mirror_move(move)

        board_input = tokenize_board(board)
        move_input = self.all_moves_dict[move]

        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)

        legal_moves, side_info = get_side_info(board, move, self.all_moves_dict)

        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info


class MAIA2Dataset(torch.utils.data.Dataset):

    def __init__(self, data, all_moves_dict, cfg):
        self.all_moves_dict = all_moves_dict
        # print(all_moves_dict)
        self.data = data
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_input, move_uci, elo_self, elo_oppo, active_win, time_control, clock_info, clock_delta = self.data[idx]

        board = chess.Board(board_input)
        board_input = tokenize_board(board)

        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)

        move_input = self.all_moves_dict[move_uci]

        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, active_win, list(time_control), clock_info, clock_delta


class BasicBlock(torch.nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        mid_planes = planes

        self.conv1 = torch.nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)

        return out


class ChessResNet(torch.nn.Module):

    def __init__(self, block, cfg):
        super(ChessResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(cfg.input_channels, cfg.dim_cnn, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EloAwareAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., elo_dim=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, elo_emb):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., elo_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(nn.ModuleList([
                EloAwareAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, elo_dim=elo_dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, elo_emb):
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x

        return self.norm(x)

class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", batch_first=True):
        super().__init__()
        # Self-attention only (no cross-attention)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )

        # Feed-forward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropouts for the residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, seq_len, d_model] if batch_first=True
        attn_mask: optional attention mask
            - can be 2D [seq_len, seq_len] => broadcast
            - or 3D [batch_size, seq_len, seq_len], which we might expand if needed
        """
        # ---- Self-Attention ----
        x2, _ = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask)
        x = x + self.dropout1(x2)  # residual connection
        x = self.norm1(x)

        # ---- Feed Forward ----
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual connection
        x = self.norm2(x)

        return x


class CustomTransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1, activation="relu"):
        """
        Args:
            dim (int): model dimension (d_model).
            depth (int): number of decoder blocks.
            heads (int): number of self-attention heads in each block.
            dim_head (int): not used directly here, included for API compatibility.
            mlp_dim (int): dimension of the feed-forward 'inner' layer.
            dropout (float): dropout probability.
            activation (str): activation function ("relu", "gelu", etc.).
        """
        super().__init__()
        self.heads = heads

        # Create `depth` layers of DecoderOnlyBlock
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, seq_len, dim] if batch_first=True
        attn_mask: optional attention mask
            - 2D shape [seq_len, seq_len], broadcast to all samples/heads
            - or 3D shape [batch_size, seq_len, seq_len]
        """
        # (Optional) expand 3D mask to [batch_size * n_heads, seq_len, seq_len] if needed
        if attn_mask is not None and attn_mask.dim() == 3:
            b, t, _ = attn_mask.shape
            attn_mask = (
                attn_mask.unsqueeze(1)                # => [B, 1, T, T]
                .expand(-1, self.heads, -1, -1)       # => [B, heads, T, T]
                .reshape(b * self.heads, t, t)        # => [B*heads, T, T]
            )

        # Pass through each DecoderOnlyBlock
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Final normalization
        return self.norm(x)



class MAIA2Transformer(nn.Module):
    def __init__(self, output_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_dim = 128
        d_model = cfg.dim_vit

        # === Projections ===
        self.token_projection = nn.Linear(self.token_dim, d_model)
        self.elo_projection = nn.Linear(self.token_dim, d_model)
        self.time_control_embedding = nn.Linear(2, d_model)
        self.time_remaining_embedding = nn.Linear(1, d_model)

        # === Learned Tokens ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.low_elo_token = nn.Parameter(torch.randn(128))
        self.high_elo_token = nn.Parameter(torch.randn(128))

        # === Row/Col embeddings for 64 squares ===
        self.row_embed = nn.Embedding(8, d_model)
        self.col_embed = nn.Embedding(8, d_model)

        # === Small positional offsets ===
        self.castling_turn_pos_embed = nn.Embedding(2, d_model)  # castling=0, turn=1
        self.elo_pos_embed = nn.Embedding(2, d_model)            # self=0, oppo=1

        # === Transformer ===
        self.transformer = CustomTransformerDecoder(
            dim=d_model,
            depth=cfg.num_blocks_vit,
            heads=32,
            dim_head=32,
            mlp_dim=int(d_model * 1.5),
            dropout=0.1
        )

        # === Output heads ===
        self.fc_move = nn.Linear(d_model, output_dim)
        self.fc_side_info = nn.Linear(d_model, output_dim + 6 + 6 + 1 + 64 + 64)
        self.fc_value_hidden = nn.Linear(d_model, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_move_time_hidden = nn.Linear(d_model, 256)
        self.fc_move_time = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(d_model)

    def interpolate_elo(self, elos):
        low_weight = (2800 - elos) / 2300
        high_weight = (elos-500) / 2300
        B = elos.size(0)
        elo_emb_114 = low_weight.unsqueeze(1) * self.low_elo_token + high_weight.unsqueeze(1) * self.high_elo_token
        zeros_14 = torch.zeros((B, 0), device=elo_emb_114.device, dtype=elo_emb_114.dtype)
        elo_emb_128 = torch.cat([zeros_14, elo_emb_114], dim=-1)
        return elo_emb_128

    def forward(self, token_sequence, elos_self, elos_oppo, time_controls, clock_info, attn_mask=None):
        B, T, _ = token_sequence.shape  # token_sequence: [B, 66, 128]
        if random.random() < 0.005:
            print(f"[forward] Runtime batch size: {B}", flush=True)

        squares_128 = token_sequence[:, :64, :]     # [B, 64, 128]
        castling_128 = token_sequence[:, 64:65, :]  # [B, 1, 128]
        turn_128 = token_sequence[:, 65:66, :]      # [B, 1, 128]

        # === [CLS] token ===
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]

        # === ELO tokens ===
        elo_self = self.elo_projection(self.interpolate_elo(elos_self)).unsqueeze(1)  # [B,1,d_model]
        elo_oppo = self.elo_projection(self.interpolate_elo(elos_oppo)).unsqueeze(1)  # [B,1,d_model]

        # Add positional offsets to distinguish ELO self/oppo
        elo_self += self.elo_pos_embed.weight[0].unsqueeze(0).unsqueeze(1)  # [B,1,d_model]
        elo_oppo += self.elo_pos_embed.weight[1].unsqueeze(0).unsqueeze(1)

        # === Time control ===
        time_ctrl_emb = self.time_control_embedding(time_controls).unsqueeze(1)     # [B,1,d_model]
        time_left_emb = self.time_remaining_embedding(clock_info.unsqueeze(1)).unsqueeze(1)  # [B,1,d_model]

        # === Project square tokens + add row/col embeddings ===
        squares_proj = self.token_projection(squares_128)  # [B,64,d_model]
        idx = torch.arange(64, device=squares_proj.device)
        rows = idx // 8
        cols = idx % 8
        squares_proj += (self.row_embed(rows) + self.col_embed(cols)).unsqueeze(0)

        # === Castling and player-turn ===
        castling_proj = self.token_projection(castling_128)  # [B,1,d_model]
        turn_proj     = self.token_projection(turn_128)      # [B,1,d_model]

        # Add positional offsets to castling and turn tokens
        castling_proj += self.castling_turn_pos_embed.weight[0].unsqueeze(0).unsqueeze(1)
        turn_proj     += self.castling_turn_pos_embed.weight[1].unsqueeze(0).unsqueeze(1)

        # === Concatenate full token sequence ===
        x = torch.cat([
            cls_tokens,       # index 0
            elo_self,         # index 1
            elo_oppo,         # index 2
            time_ctrl_emb,    # index 3
            time_left_emb,    # index 4
            squares_proj,     # index 5–68
            castling_proj,    # index 69
            turn_proj         # index 70
        ], dim=1)  # [B, 71, d_model]

        # === Transformer ===
        x = self.dropout(x)
        x = self.transformer(x, attn_mask=attn_mask)
        x_cls = self.last_ln(x[:, 0, :])  # [CLS] output

        # === Final heads ===
        logits_move       = self.fc_move(x_cls)
        logits_side_info  = self.fc_side_info(x_cls)
        logits_value      = self.fc_value(F.relu(self.fc_value_hidden(x_cls))).squeeze(-1)
        logits_move_time  = self.fc_move_time(F.relu(self.fc_move_time_hidden(x_cls))).squeeze(-1)

        return logits_move, logits_side_info, logits_value, logits_move_time




def read_monthly_data_path(cfg):
    print('Training Data:', flush=True)
    pgn_paths = []

    for year in range(cfg.start_year, cfg.end_year + 1):
        start_month = cfg.start_month if year == cfg.start_year else 1
        end_month = cfg.end_month if year == cfg.end_year else 12

        for month in range(start_month, end_month + 1):
            formatted_month = f"{month:02d}"
            pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
            # skip 2019-12
            if year == 2019 and month == 12:
                continue
            print(pgn_path, flush=True)
            pgn_paths.append(pgn_path)

    return pgn_paths


def evaluate(model, dataloader):
    counter = 0
    correct_move = 0

    model.eval()
    with torch.no_grad():
        for boards, labels, elos_self, elos_oppo, legal_moves, side_info in dataloader:
            boards = boards.cuda()
            labels = labels.cuda()
            elos_self = elos_self.cuda()
            elos_oppo = elos_oppo.cuda()
            legal_moves = legal_moves.cuda()

            logits_maia, logits_side_info, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            preds = logits_maia_legal.argmax(dim=-1)
            correct_move += (preds == labels).sum().item()

            counter += len(labels)

    return correct_move, counter


def evaluate_MAIA1_data(model, all_moves_dict, elo_dict, cfg, tiny=False):
    elo_list = range(1000, 2600, 100)

    for i in elo_list:
        start = i
        end = i + 100
        file_path = f"../data/test/KDDTest_{start}-{end}.csv"
        data = pd.read_csv(file_path)
        data = data[data.type == 'Rapid'][['board', 'move', 'active_elo', 'opponent_elo', 'white_active']]
        dataset = MAIA1Dataset(data, all_moves_dict, elo_dict, cfg)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=cfg.num_workers)
        if cfg.verbose:
            dataloader = tqdm.tqdm(dataloader)
        print(f'Testing Elo Range {start}-{end} with MAIA 1 data:', flush=True)
        correct_move, counter = evaluate(model, dataloader)
        print(f'Accuracy Move Prediction: {round(correct_move / counter, 4)}', flush=True)
        if tiny:
            break

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB

    # GPU Utilization percentage (Torch 2.0+)
    utilization = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else "N/A"

    print(f"GPU Memory: Allocated {allocated:.2f} GB | Reserved {reserved:.2f} GB | Utilization: {utilization}%", flush=True)
    sys.stdout.flush()

def print_grad_stats(model, step):
    """
    Prints gradient statistics (mean, max abs grad) for every parameter
    that has a non-None .grad. Typically called after loss.backward()
    and before optimizer.step().
    """
    print(f"== Step {step} gradient stats ==")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"  {name}: mean={grad_mean:.6f}, max={grad_max:.6f}", flush=True)
        else:
            print(f"  {name}: grad is None", flush=True)


def print_time_stats(time_controls, clock_info, clock_deltas, step):
    """
    Prints min, max, mean, and std for time_controls (split into [base, inc]),
    clock_info, and clock_deltas.
    
    Args:
        time_controls: Tensor [batch_size, 2]
        clock_info:    Tensor [batch_size]
        clock_deltas:  Tensor [batch_size]
        step:          The current step (for logging)
    """
    # Detach and move to CPU for easy stats
    tc = time_controls.detach().cpu()
    ci = clock_info.detach().cpu()
    cd = clock_deltas.detach().cpu()

    # Split base/inc from time_controls
    base = tc[:, 0]
    inc  = tc[:, 1]

    print(f"== Step {step} time stats ==", flush=True)
    print(
        f" time_controls base: min={base.min():.3f}, max={base.max():.3f}, "
        f"mean={base.mean():.3f}, std={base.std():.3f}", flush=True
    )
    print(
        f" time_controls inc:  min={inc.min():.3f}, max={inc.max():.3f}, "
        f"mean={inc.mean():.3f}, std={inc.std():.3f}", flush=True
    )
    print(
        f" clock_info:         min={ci.min():.3f}, max={ci.max():.3f}, "
        f"mean={ci.mean():.3f}, std={ci.std():.3f}", flush=True
    )
    print(
        f" clock_deltas:       min={cd.min():.3f}, max={cd.max():.3f}, "
        f"mean={cd.mean():.3f}, std={cd.std():.3f}", flush=True
    )


def train_chunks(
    cfg,
    data_train,
    data_val,  # <-- separate validation data
    model,
    optimizer,
    all_moves_dict,
    criterion_maia,
    criterion_side_info,
    criterion_value,
    criterion_move_time
):
    """
    Train for one epoch on data_train, then run validation on data_val.
    Returns:
        (1) train_loss
        (2) train_loss_maia
        (3) train_loss_side_info
        (4) train_loss_value
        (5) train_loss_move_time
        (6) val_loss                 (overall)
        (7) val_loss_maia
        (8) val_loss_side_info
        (9) val_loss_value
        (10) val_loss_move_time
        (11) val_accuracy
    """
    # if len(data_train) > 0:
    #     print(f"Data size: {len(data_train)}")  # Print length of the data list
    #     print(data_train[0])
    #     with open("fen_nonempty_strings.pkl", "wb") as file:
    #         pickle.dump(data_train[:250000], file)
    #         file.flush()  # Ensure data is flushed to disk
    #         os.fsync(file.fileno())  # Force OS-level flush to disk
    #         print("file saved (hopefully)")

    #     time.sleep(90)  # Allow some time for I/O operations to complete
    #     terminate_all()  # Still crashes, but ensures the pickle file is actually written

    ######################
    # 1) Create Datasets #
    ######################

    print("TOTAL LENGTH OF DATA:", flush = True)
    print(len(data_train), flush=True)

    dataset_train = MAIA2Dataset(data_train, all_moves_dict, cfg)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )

    dataset_val = MAIA2Dataset(data_val, all_moves_dict, cfg)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )

    if cfg.verbose:
        dataloader_train = tqdm.tqdm(dataloader_train, desc="Training")

    #########################
    # 2) Training (One Epoch)
    #########################
    model.train()  # train mode

    avg_loss = 0.0
    avg_loss_maia = 0.0
    avg_loss_side_info = 0.0
    avg_loss_value = 0.0
    avg_loss_move_time = 0.0

    step = 0

    for (
        boards,
        labels,
        elos_self,
        elos_oppo,
        legal_moves,
        side_info,
        wdl,
        time_controls,
        clock_info,
        clock_deltas
    ) in dataloader_train:

        boards = boards.cuda()
        labels = labels.cuda()
        elos_self = elos_self.cuda()
        elos_oppo = elos_oppo.cuda()
        side_info = side_info.cuda()
        wdl = wdl.float().cuda()

        # Convert times
        time_controls = torch.stack(time_controls, dim=1).float().cuda() / 60.0
        clock_info = clock_info.float().cuda() / 60.0
        clock_deltas = clock_deltas.float().cuda() / 60.0

        clock_deltas = torch.clamp_min(clock_deltas, 0.0)
        # 2) log(1 + clock_delta)
        log_clock_deltas = torch.log1p(clock_deltas)  # shape [batch_size]

        if step == 0:

            print("time_controls shape:", time_controls.shape)
            print("clock_info shape:", clock_info.shape)
            print("clock_deltas shape:", clock_deltas.shape)

            # Now you can do something like:
            base_times = time_controls[:, 0].tolist()  # [batch_size]
            increments = time_controls[:, 1].tolist()
            ci = clock_info.tolist()
            cd = clock_deltas.tolist()
            
            # For example, turn the first 100 items into a little table:
            for i in range(min(100, len(base_times))):
                print(
                    f"Sample {i}: base={base_times[i]:.2f}, inc={increments[i]:.2f}, "
                    f"clock_info={ci[i]:.2f}, clock_delta={cd[i]:.2f}", flush=True
                )



        # Forward
        logits_maia, logits_side_info_, logits_value, logits_move_time = model(
            boards, elos_self, elos_oppo, time_controls, clock_info
        )

        # Compute losses
        loss_maia_ = criterion_maia(logits_maia, labels)
        loss = 0
        loss += loss_maia_

        if cfg.side_info:
            loss_side_info_ = criterion_side_info(logits_side_info_, side_info) * cfg.side_info_coefficient
            loss += loss_side_info_
        else:
            loss_side_info_ = 0.0

        if cfg.value:
            loss_value_ = criterion_value(logits_value, wdl) * cfg.value_coefficient
            loss += loss_value_
        else:
            loss_value_ = 0.0

        if cfg.move_time:
            loss_move_time_ = criterion_move_time(logits_move_time, log_clock_deltas) * cfg.move_time_coefficient
            loss += loss_move_time_
        else:
            loss_move_time_ = 0.0

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Print stats
        if step % 500 == 0:
            print_time_stats(time_controls, clock_info, clock_deltas, step)
            print_grad_stats(model, step)
        
        optimizer.step()

        # Accumulate training stats
        avg_loss += loss.item()
        avg_loss_maia += loss_maia_.item()
        if cfg.side_info:
            avg_loss_side_info += loss_side_info_.item()
        if cfg.value:
            avg_loss_value += loss_value_.item()
        if cfg.move_time:
            avg_loss_move_time += loss_move_time_.item()

        step += 1

    # Average training losses
    train_loss = round(avg_loss / step, 3)
    train_loss_maia = round(avg_loss_maia / step, 3)
    train_loss_side_info = round(avg_loss_side_info / step, 3)
    train_loss_value = round(avg_loss_value / step, 3)
    train_loss_move_time = round(avg_loss_move_time / step, 3)

    #######################
    # 3) Validation Phase #
    #######################
    model.eval()

    val_loss_sum = 0.0
    val_loss_maia = 0.0
    val_loss_side_info = 0.0
    val_loss_value = 0.0
    val_loss_move_time = 0.0

    val_steps = 0
    correct = 0
    total = 0

    if cfg.verbose:
        dataloader_val = tqdm.tqdm(dataloader_val, desc="Validation")

    with torch.no_grad():
        for (
            boards,
            labels,
            elos_self,
            elos_oppo,
            legal_moves,
            side_info,
            wdl,
            time_controls,
            clock_info,
            clock_deltas
        ) in dataloader_val:

            boards = boards.cuda()
            labels = labels.cuda()
            elos_self = elos_self.cuda()
            elos_oppo = elos_oppo.cuda()
            side_info = side_info.cuda()
            wdl = wdl.float().cuda()

            time_controls = torch.stack(time_controls, dim=1).float().cuda() / 60.0
            clock_info = clock_info.float().cuda() / 60.0
            clock_deltas = clock_deltas.float().cuda() / 60.0
            
            clock_deltas = torch.clamp_min(clock_deltas, 0.0)
            # 2) log(1 + clock_delta)
            log_clock_deltas = torch.log1p(clock_deltas)  # shape [batch_size]

            # Forward pass
            logits_maia, logits_side_info_, logits_value, logits_move_time = model(
                boards, elos_self, elos_oppo, time_controls, clock_info
            )

            # Individual validation losses
            loss_val_maia = criterion_maia(logits_maia, labels)
            val_loss_maia += loss_val_maia.item()

            this_val_side_info = 0.0
            if cfg.side_info:
                this_val_side_info = (
                    criterion_side_info(logits_side_info_, side_info) * cfg.side_info_coefficient
                )
                val_loss_side_info += this_val_side_info.item()

            this_val_value = 0.0
            if cfg.value:
                this_val_value = criterion_value(logits_value, wdl) * cfg.value_coefficient
                val_loss_value += this_val_value.item()

            this_val_move_time = 0.0
            if cfg.move_time:
                this_val_move_time = (
                    criterion_move_time(logits_move_time, log_clock_deltas) * cfg.move_time_coefficient
                )
                val_loss_move_time += this_val_move_time.item()

            # Sum of partial losses
            total_val_loss = loss_val_maia + this_val_side_info + this_val_value + this_val_move_time
            val_loss_sum += total_val_loss.item()

            # Accuracy (example: top-1 on MAIA output)
            _, predicted = torch.max(logits_maia, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            val_steps += 1

    # Average validation losses
    val_loss = round(val_loss_sum / val_steps, 3)
    val_loss_maia = round(val_loss_maia / val_steps, 3)
    val_loss_side_info = round(val_loss_side_info / val_steps, 3)
    val_loss_value = round(val_loss_value / val_steps, 3)
    val_loss_move_time = round(val_loss_move_time / val_steps, 3)

    val_accuracy = round(correct / total, 3) if total > 0 else 0.0

    ###########################
    # 4) Return All Statistics
    ###########################
    return (
        train_loss,                # 1) train_loss
        train_loss_maia,           # 2) train_loss_maia
        train_loss_side_info,      # 3) train_loss_side_info
        train_loss_value,          # 4) train_loss_value
        train_loss_move_time,      # 5) train_loss_move_time

        val_loss,                  # 6) val_loss (overall)
        val_loss_maia,             # 7) val_loss_maia
        val_loss_side_info,        # 8) val_loss_side_info
        val_loss_value,            # 9) val_loss_value
        val_loss_move_time,        # 10) val_loss_move_time
        val_accuracy               # 11) val_accuracy
    )


def preprocess_thread(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict):
    data, game_count, chunk_count = process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
    queue.put([data, game_count, chunk_count])
    del data


def worker_wrapper(semaphore, *args, **kwargs):
    with semaphore:
        preprocess_thread(*args, **kwargs)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default='../../../../../grace/u/geilender', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=3, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=20000, type=str)
    parser.add_argument('--start_year', default=2019, type=int)
    parser.add_argument('--start_month', default=5, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--end_month', default=5, type=int)
    parser.add_argument('--from_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_epoch', default=0, type=int)
    parser.add_argument('--checkpoint_year', default=2018, type=int)
    parser.add_argument('--checkpoint_month', default=5, type=int)
    parser.add_argument('--num_cpu_left', default=7, type=int)
    parser.add_argument('--queue_length', default=2, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1021, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--token_dim', default = 14, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=4, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=72, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)
    parser.add_argument('--move_time', default=True, type=bool)
    parser.add_argument('--move_time_coefficient', default=1, type=float)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    return parser.parse_args(args)

if __name__ == "__main__":
    cfg = parse_args()
    train_fen_model.run(cfg)
