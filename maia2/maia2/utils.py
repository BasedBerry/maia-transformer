import pdb
import chess
import pickle
import os
import random
import numpy as np
import torch
import torch.nn as nn
from io import StringIO
import time
import requests
import tqdm
import pyzstd
import re


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def delete_file(filename):
    
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Data {filename} has been deleted.")
    else:
        print(f"The file '{filename}' does not exist.")


def readable_num(num):
    
    if num >= 1e9:  # if parameters are in the billions
        return f'{num / 1e9:.2f}B'
    elif num >= 1e6:  # if parameters are in the millions
        return f'{num / 1e6:.2f}M'
    elif num >= 1e3:  # if parameters are in the thousands
        return f'{num / 1e3:.2f}K'
    else:
        return str(num)


def readable_time(elapsed_time):

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def count_parameters(model):
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return readable_num(total_params)


def create_elo_dict():
    
    inteval = 100
    start = 1100
    end = 2000
    
    range_dict = {f"<{start}": 0}
    range_index = 1

    for lower_bound in range(start, end - 1, inteval):
        upper_bound = lower_bound + inteval
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={end}"] = range_index
    
    # print(range_dict, flush=True)
    
    return range_dict


def map_to_category(elo, elo_dict):

    inteval = 100
    start = 1100
    end = 2000
    
    if elo < start:
        return elo_dict[f"<{start}"]
    elif elo >= end:
        return elo_dict[f">={end}"]
    else:
        for lower_bound in range(start, end - 1, inteval):
            upper_bound = lower_bound + inteval
            if lower_bound <= elo < upper_bound:
                return elo_dict[f"{lower_bound}-{upper_bound - 1}"]


def get_side_info(board, move_uci, all_moves_dict):
    move = chess.Move.from_uci(move_uci)
    
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)

    from_square_encoded = torch.zeros(64)
    from_square_encoded[move.from_square] = 1

    to_square_encoded = torch.zeros(64)
    to_square_encoded[move.to_square] = 1
    
    if move_uci == 'e1g1':
        rook_move = chess.Move.from_uci('h1f1')
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1
    
    if move_uci == 'e1c1':
        rook_move = chess.Move.from_uci('a1d1')
        from_square_encoded[rook_move.from_square] = 1
        to_square_encoded[rook_move.to_square] = 1

    board.push(move)
    is_check = board.is_check()
    board.pop()
    
    # Order: Pawn, Knight, Bishop, Rook, Queen, King
    side_info = torch.zeros(6 + 6 + 1)
    side_info[moving_piece.piece_type - 1] = 1
    if move_uci in ['e1g1', 'e1c1']:
        side_info[3] = 1
    if captured_piece:
        side_info[6 + captured_piece.piece_type - 1] = 1
    if is_check:
        side_info[-1] = 1
    
    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor([all_moves_dict[move.uci()] for move in board.legal_moves])
    legal_moves[legal_moves_idx] = 1
    
    side_info = torch.cat([side_info, from_square_encoded, to_square_encoded, legal_moves], dim=0)
    
    return legal_moves, side_info


def extract_clock_time(comment):
    
    match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None
    

def read_or_create_chunks(pgn_path, cfg):

    cache_file = pgn_path.replace('.pgn', '_chunks.pkl')

    if os.path.exists(cache_file):
        print(f"Loading cached chunks from {cache_file}")
        with open(cache_file, 'rb') as f:
            pgn_chunks = pickle.load(f)
    else:
        print(f"Cache not found. Creating chunks for {pgn_path}")
        start_time = time.time()
        pgn_chunks = get_chunks(pgn_path, cfg.chunk_size)
        print(f'Chunking took {readable_time(time.time() - start_time)}', flush=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(pgn_chunks, f)
    
    return pgn_chunks


def board_to_tensor(board):
    
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    num_piece_channels = 12  # 6 piece types * 2 colors
    additional_channels = 6  # 1 for player's turn, 4 for castling rights, 1 for en passant
    tensor = torch.zeros((num_piece_channels + additional_channels, 8, 8), dtype=torch.float32)

    # Precompute indices for each piece type
    piece_indices = {piece: i for i, piece in enumerate(piece_types)}

    # Fill tensor for each piece type
    for piece_type in piece_types:
        for color in [True, False]:  # True is White, False is Black
            piece_map = board.pieces(piece_type, color)
            index = piece_indices[piece_type] + (0 if color else 6)
            for square in piece_map:
                row, col = divmod(square, 8)
                tensor[index, row, col] = 1.0

    # Player's turn channel (White = 1, Black = 0)
    turn_channel = num_piece_channels
    if board.turn == chess.WHITE:
        tensor[turn_channel, :, :] = 1.0

    # Castling rights channels
    castling_rights = [board.has_kingside_castling_rights(chess.WHITE),
                       board.has_queenside_castling_rights(chess.WHITE),
                       board.has_kingside_castling_rights(chess.BLACK),
                       board.has_queenside_castling_rights(chess.BLACK)]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            tensor[num_piece_channels + 1 + i, :, :] = 1.0

    # En passant target channel
    ep_channel = num_piece_channels + 5
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[ep_channel, row, col] = 1.0

    return tensor


def tokenize_board_to_tensor(board: chess.Board, token_dim=14):
    """
    Tokenize the chess board into a tensor compatible with the MAIA2Model.
    Includes square tokens (with en passant info), castling rights token, and player turn token.

    Args:
        board (chess.Board): The chess board to tokenize.
        token_dim (int): Dimension of each token (square, castling rights, player turn).

    Returns:
        torch.Tensor: Tensor of shape (66, token_dim).
    """
    # Define the mapping of piece types to vector indices
    piece_map = {
        None: 0,  # Empty square
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Get the en passant square
    en_passant_square = board.ep_square

    # Create a square token
    def square_token(square):
        piece = board.piece_at(square)
        token = [0] * token_dim
        if piece:
            index = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                index += 6  # Offset for black pieces
        else:
            index = 0  # Empty square
        token[index] = 1
        # Add en passant information
        token[13] = 1 if square == en_passant_square else 0
        return token

    # Tokenize all 64 squares
    square_tokens = [square_token(square) for square in chess.SQUARES]

    # Create castling rights token
    castling_rights = [
        int(board.has_kingside_castling_rights(chess.WHITE)),  # White kingside
        int(board.has_queenside_castling_rights(chess.WHITE)),  # White queenside
        int(board.has_kingside_castling_rights(chess.BLACK)),  # Black kingside
        int(board.has_queenside_castling_rights(chess.BLACK)),  # Black queenside
    ]
    castling_rights_token = castling_rights + [0] * (token_dim - len(castling_rights))  # Pad to token_dim

    # Create player turn token
    player_turn_token = [int(board.turn)] + [0] * (token_dim - 1)  # 1 for white, 0 for black

    # Combine square tokens, castling rights token, and player turn token
    tokens = square_tokens + [castling_rights_token, player_turn_token]

    # Convert to a PyTorch tensor of shape (66, token_dim)
    tensor = torch.tensor(tokens, dtype=torch.float32)
    return tensor

class MoveTokenizer(nn.Module):
    def __init__(self, input_dim=136, output_dim=14):
        """
        A module to tokenize chess moves into a fixed-dimensional embedding space.

        Args:
            input_dim (int): Dimensionality of the raw move representation (e.g., 136).
            output_dim (int): Desired token dimension (e.g., 14, to match square tokens).
        """
        super(MoveTokenizer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, moves):
        """
        Tokenize and project moves to a fixed embedding space.

        Args:
            moves (torch.Tensor): Tensor of raw move features (num_moves, input_dim).

        Returns:
            torch.Tensor: Tokenized moves of shape (num_moves, output_dim).
        """
        return self.linear(moves)

def raw_move_representation(pgn_moves):
    """
    Generate raw move representations (before projection).

    Args:
        pgn_moves (list of chess.Move): List of moves in PGN format.

    Returns:
        torch.Tensor: Raw move features of shape (num_moves, 136).
    """
    def move_features(move):
        # Initialize a raw feature vector (136 dimensions)
        features = [0] * 136

        # Source square (0-63)
        features[move.from_square] = 1

        # Destination square (64-127)
        features[64 + move.to_square] = 1

        # Promotion (128-131)
        promotion_map = {
            chess.QUEEN: 128,
            chess.ROOK: 129,
            chess.BISHOP: 130,
            chess.KNIGHT: 131,
        }
        if move.promotion:
            features[promotion_map[move.promotion]] = 1

        # Castling flags (132-135)
        if move.uci() in ["e1g1", "e1c1"]:  # White castling
            features[132] = 1  # Kingside
            features[133] = 1  # Queenside
        elif move.uci() in ["e8g8", "e8c8"]:  # Black castling
            features[134] = 1  # Kingside
            features[135] = 1  # Queenside

        return features

    # Generate raw features for each move
    raw_features = [move_features(move) for move in pgn_moves]

    # Convert to PyTorch tensor
    return torch.tensor(raw_features, dtype=torch.float32)

# Example Usage

def game_to_tensor(pgn, move_tokenizer, token_dim=14):
    """
    Parse a PGN, randomly pick a move, generate the FEN, insert FEN tokens into the move token sequence,
    and construct a causal mask.

    Args:
        pgn (str): The PGN string of the game.
        move_tokenizer (MoveTokenizer): The tokenizer for move embeddings.
        token_dim (int): The dimension of the tokens (should match square token dimension).

    Returns:
        torch.Tensor: The combined sequence of move tokens and FEN tokens.
        torch.Tensor: The corresponding causal mask.
    """
    # Parse the PGN
    pgn_file = StringIO(pgn)  # Wrap the PGN string in a file-like object
    game = chess.pgn.read_game(pgn_file)
    board = game.board()
    moves = list(game.mainline_moves())

    # Generate move tokens
    raw_move_features = raw_move_representation(moves)  # Shape: (num_moves, 136)
    move_tokens = move_tokenizer(raw_move_features)  # Shape: (num_moves, token_dim)

    # Randomly pick a move index
    random_move_index = random.randint(0, len(moves) - 1)
    selected_move = moves[random_move_index]

    # Replay the game up to the selected move
    board.reset()
    for move in moves[:random_move_index + 1]:
        board.push(move)

    # Generate FEN tokens for the current board state
    fen_tokens = tokenize_board_to_tensor(board, token_dim=token_dim)  # Shape: (66, token_dim)

    # Combine the sequence: moves up to the FEN, FEN tokens, and remaining moves
    move_tokens_before_fen = move_tokens[:random_move_index + 1]  # Moves up to the FEN
    move_tokens_after_fen = move_tokens[random_move_index + 1:]  # Moves after the FEN
    combined_sequence = torch.cat([move_tokens_before_fen, fen_tokens, move_tokens_after_fen], dim=0)

    # Construct the causal mask
    num_moves_before = random_move_index + 1
    num_moves_after = len(moves) - random_move_index - 1
    num_fen_tokens = fen_tokens.size(0)
    total_tokens = num_moves_before + num_fen_tokens + num_moves_after

    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)

    # Moves-to-Moves Causal Mask
    for i in range(num_moves_before):
        mask[i, i + 1:] = True  # Prevent move i from attending to future moves

    # Moves-to-FEN Mask
    for i in range(num_moves_before):
        mask[i, num_moves_before: num_moves_before + num_fen_tokens] = False  # Moves can attend to FEN

    # Moves-after-FEN Causal Mask
    for i in range(num_moves_before + num_fen_tokens, total_tokens):
        mask[i, i + 1:] = True  # Moves after FEN cannot attend to future moves
        mask[i, :num_moves_before] = False  # Moves after FEN can attend to earlier moves
        mask[i, num_moves_before: num_moves_before + num_fen_tokens] = False  # Moves after FEN can attend to FEN

    # FEN-to-FEN: No restrictions
    for i in range(num_moves_before, num_moves_before + num_fen_tokens):
        mask[i, num_moves_before:num_moves_before + num_fen_tokens] = False  # FEN tokens can attend to each other
        mask[i, :num_moves_before] = False  # FEN tokens can attend to all earlier move tokens

    return combined_sequence, mask



def generate_pawn_promotions():
    # Define the promotion rows for both colors and the promotion pieces
    # promotion_rows = {'white': '7', 'black': '2'}
    promotion_rows = {'white': '7'}
    promotion_pieces = ['q', 'r', 'b', 'n']
    promotions = []

    # Iterate over each color
    for color, row in promotion_rows.items():
        # Target rows for promotion (8 for white, 1 for black)
        target_row = '8' if color == 'white' else '1'

        # Each file from 'a' to 'h'
        for file in 'abcdefgh':
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f'{file}{row}{file}{target_row}{piece}')

            # Capturing moves to the left and right (if not on the edges of the board)
            if file != 'a':
                left_file = chr(ord(file) - 1)  # File to the left
                for piece in promotion_pieces:
                    promotions.append(f'{file}{row}{left_file}{target_row}{piece}')

            if file != 'h':
                right_file = chr(ord(file) + 1)  # File to the right
                for piece in promotion_pieces:
                    promotions.append(f'{file}{row}{right_file}{target_row}{piece}')

    return promotions


def mirror_square(square):
    
    file = square[0]
    rank = str(9 - int(square[1]))
    
    return file + rank


def mirror_move(move_uci):
    # Check if the move is a promotion (length of UCI string will be more than 4)
    is_promotion = len(move_uci) > 4

    # Extract the start and end squares, and the promotion piece if applicable
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    # Mirror the start and end squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    # Return the mirrored move, including the promotion piece if applicable
    return mirrored_start + mirrored_end + promotion_piece


def get_chunks(pgn_path, chunk_size):

    chunks = []
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            start_pos = pgn_file.tell()
            game_count = 0
            while game_count < chunk_size:
                line = pgn_file.readline()
                if not line:
                    break
                if line[-4:] == "1-0\n" or line[-4:] == "0-1\n":
                    game_count += 1
                if line[-8:] == "1/2-1/2\n":
                    game_count += 1
                if line[-2:] == "*\n":
                    game_count += 1
            line = pgn_file.readline()
            if line not in ["\n", ""]:
                raise ValueError
            end_pos = pgn_file.tell()
            chunks.append((start_pos, end_pos))
            if not line:
                break

    return chunks


def decompress_zst(file_path, decompressed_path):
    """ Decompress a .zst file using pyzstd """
    with open(file_path, 'rb') as compressed_file, open(decompressed_path, 'wb') as decompressed_file:
        pyzstd.decompress_stream(compressed_file, decompressed_file)


def get_all_possible_moves():
    
    all_moves = []

    for rank in range(8):
        for file in range(8): 
            square = chess.square(file, rank)
            
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)
            
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)
    
    all_moves = [all_moves[i].uci() for i in range(len(all_moves))]
    
    pawn_promotions = generate_pawn_promotions()
    
    return all_moves + pawn_promotions


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

