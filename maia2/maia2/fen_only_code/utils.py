import pdb
import chess
import pickle
import os
import random
import numpy as np
import torch
import time
import requests
import tqdm
import pyzstd
import re
from io import StringIO
import random
import multiprocessing as mp
import sys
from io import StringIO
import chess
import chess.pgn
import numpy as np
import concurrent.futures
import heapq




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
def sort_chunk(chunk):
    """Sorts a given chunk of dataset_train by sequence length."""
    return sorted(chunk, key=lambda x: len(x[0]))

def parallel_sort(data, num_workers):
    """Sorts dataset_train by sequence length using multiple workers."""
    chunk_size = max(1, len(data) // num_workers)  # Prevents zero-size chunks
    chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_workers)]

    # Use multiprocessing to sort chunks in parallel
    with mp.Pool(num_workers) as pool:
        sorted_chunks = pool.map(sort_chunk, chunks)

    # Merge sorted chunks efficiently
    sorted_data = sorted((item for chunk in sorted_chunks for item in chunk), key=lambda x: len(x[0]))
    return sorted_data



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


def extract_elo(elo):
    if elo < 500:
        return 500
    elif elo > 2800:
        return 2800
    
    return elo 

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

def board_from_pgn(pgn_str: str) -> chess.Board:
    """
    Given a PGN string, returns a chess.Board object representing
    the position after the moves in the PGN are played.
    """
    # Create a file-like object from the PGN string
    pgn_io = StringIO(pgn_str)
    
    # Read the game from the file-like object
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("No valid game found in the provided PGN string.")
    
    # Start with the initial board position
    board = game.board()
    
    # Play through all the moves in the main line of the game
    for move in game.mainline_moves():
        board.push(move)
    
    return board


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


def square_token(square, board, token_dim, piece_map, total_dim, en_passant_square):
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
    square_token = token + [0] * (total_dim-token_dim)
    return square_token



def tokenize_board(board: chess.Board, token_dim=14, total_dim=128):
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

    # Tokenize all 64 squares
    square_tokens = [square_token(square, board, token_dim, piece_map, total_dim, en_passant_square) for square in chess.SQUARES]

    # Create castling rights token
    castling_rights = [
        int(board.has_kingside_castling_rights(chess.WHITE)),  # White kingside
        int(board.has_queenside_castling_rights(chess.WHITE)),  # White queenside
        int(board.has_kingside_castling_rights(chess.BLACK)),  # Black kingside
        int(board.has_queenside_castling_rights(chess.BLACK)),  # Black queenside
    ]
    castling_rights_token = castling_rights + [0] * (total_dim - len(castling_rights))  # Pad to token_dim

    # Create player turn token
    player_turn_token = [int(board.turn)] + [0] * (total_dim - 1)  # 1 for white, 0 for black

    # Combine square tokens, castling rights token, and player turn token
    tokens = square_tokens + [castling_rights_token, player_turn_token]

    # Convert to a PyTorch tensor of shape (66, token_dim)
    tensor = torch.tensor(tokens, dtype=torch.float32)
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

def random_move_index_from_pgn(pgn):
    pgn_file = StringIO(pgn)
    game = chess.pgn.read_game(pgn_file)
    moves = list(game.mainline_moves())
    num_moves = len(moves)
    total_indices = num_moves + 1  # Possibility for a pure FEN token selection.
    probabilities = np.full(total_indices, 1, dtype=np.float32)  # Uniform distribution

    fen_bias = 0.15
    move_bias = 0.15
    remaining_prob = 1 - (fen_bias + move_bias)

    probabilities[-1] = fen_bias
    probabilities[-2] = move_bias
    probabilities[:-2] *= remaining_prob / (total_indices - 2)
    probabilities /= probabilities.sum()

    move_index = np.random.choice(range(-1, num_moves), p=probabilities)
    sequence_length = calculate_ultimate_sequence_length(move_index, num_moves)
    return move_index, sequence_length

def calculate_ultimate_sequence_length(move_index, num_moves, n_fen_start=67, n_fen_state=67):
    """
    Calculate the ultimate sequence length for the tokenized game based on the move index and the number of moves.
    
    Args:
        move_index (int): The chosen move index (can be negative for special cases).
        num_moves (int): Total number of moves in the game.
        n_fen_start (int): Number of FEN tokens for the starting board.
        n_fen_state (int): Number of FEN tokens for the board after applying moves up to the selected move index.
    
    Returns:
        int: The computed sequence length.
    
    Cases:
      - Pure FEN Tokens: if move_index == -1 or move_index == num_moves - 1, then
          sequence_length = n_fen_state + 1.
      
      - Pure Move Tokens: if move_index == -2 or move_index == num_moves - 2, then
          sequence_length = n_fen_start + 1 + num_moves.
      
      - Mixed Case: otherwise,
          sequence_length = n_fen_state + 1 + (num_moves - (move_index + 1)).
    """
    if move_index == -1 or move_index == num_moves - 1:
        return n_fen_state + 1
    elif move_index == -2 or move_index == num_moves - 2:
        return n_fen_start + 1 + num_moves
    else:
        remaining_moves = num_moves - (move_index + 1)
        return n_fen_state + 1 + remaining_moves


def process_game_with_fen_insertion_and_mask_weighted(pgn, token_dim_fen=14, token_dim_combined=150, token_dim_total=256, move_index=-1):
    """
    Parse a PGN, tokenize moves and FEN tokens, and align them into a unified embedding space (256 dimensions).
    Now includes 106 empty dimensions at the end for Elo tokens.
    
    Inserts a special token (all zeros with a 1 in the last dimension) to denote the boundary between
    FEN tokens and move tokens. The ordering is always:
        [FEN tokens] + [special token] + [move tokens]
    For a pure FEN sequence, the special token is appended at the end;
    for a pure move sequence, FEN tokens corresponding to chess's starting position are prepended,
    and for a mixed sequence, the special token is inserted after the FEN tokens and before the move tokens.
    
    In the mixed case, all FEN tokens and the special token can attend to each other, while causal masking 
    is applied only among the move tokens.
    
    Args:
        pgn (str): The PGN string of the game.
        token_dim_fen (int): Dimension of FEN tokens (default: 14).
        token_dim_combined (int): Dimension of the move + FEN embedding space (default: 150).
        token_dim_total (int): Final embedding size (default: 256).

    Returns:
        torch.Tensor: Combined token sequence of shape (sequence_length, token_dim_total).
        torch.Tensor: Causal attention mask of shape (sequence_length, sequence_length).
    """
    pgn_file = StringIO(pgn)
    game = chess.pgn.read_game(pgn_file)
    board = board_from_pgn(pgn)
    moves = list(game.mainline_moves())

    num_moves = len(moves)
    # total_indices = num_moves + 1  # Possibility for a pure FEN token selection.
    # probabilities = np.full(total_indices, 1, dtype=np.float32)  # Uniform distribution

    # fen_bias = 0.15
    # move_bias = 0.15
    # remaining_prob = 1 - (fen_bias + move_bias)

    # probabilities[-1] = fen_bias
    # probabilities[-2] = move_bias
    # probabilities[:-2] *= remaining_prob / (total_indices - 2)
    # probabilities /= probabilities.sum()

    # move_index = np.random.choice(range(-1, num_moves), p=probabilities)
    # Example override:

    # Define the special token: zeros with a 1 in the last dimension.
    special_token = torch.zeros(token_dim_total, dtype=torch.float32)
    special_token[255] = 1.0

    try:
        # **Case 1: Pure FEN Tokens**
        # (No move tokens. Ordering: [FEN tokens] + [special token])
        if move_index == num_moves - 1 or move_index == -1:
            fen_tokens = tokenize_board_to_tensor(board, token_dim=token_dim_fen)  # (n_fen, token_dim_fen)
            n_fen = fen_tokens.size(0)
            sequence_length = n_fen + 1  # Append special token at the end.
            combined_tokens = torch.zeros((sequence_length, token_dim_total), dtype=torch.float32)
            # Place FEN tokens in designated dims (e.g., 136:150).
            combined_tokens[:n_fen, 136:150] = fen_tokens
            # Append the special token.
            combined_tokens[n_fen] = special_token
            causal_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
            return combined_tokens, causal_mask

        # **Case 2: Pure Move Tokens**
        # (Previously: [] + [special token] + [move tokens])
        # Now: [FEN tokens for starting position] + [special token] + [move tokens]
        elif move_index == num_moves - 2 or move_index == -2:
            # Generate FEN tokens for the starting position.
            starting_board = chess.Board()  # Starting position board.
            fen_tokens = tokenize_board_to_tensor(starting_board, token_dim=token_dim_fen)  # (n_fen, token_dim_fen)
            n_fen = fen_tokens.size(0)
            raw_move_features = raw_move_representation(moves)  # (num_moves, 136)
            sequence_length = n_fen + 1 + num_moves  # FEN tokens + special token + move tokens.
            combined_tokens = torch.zeros((sequence_length, token_dim_total), dtype=torch.float32)
            # Insert FEN tokens.
            combined_tokens[:n_fen, 136:150] = fen_tokens
            # Insert the special token immediately after FEN tokens.
            combined_tokens[n_fen] = special_token
            # Insert move tokens after the special token.
            combined_tokens[n_fen+1:, :136] = raw_move_features
            causal_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
            if num_moves > 0:
                move_causal_mask = torch.triu(torch.ones((num_moves, num_moves), dtype=torch.bool), diagonal=1)
                causal_mask[n_fen+1:, n_fen+1:] = move_causal_mask
            return combined_tokens, causal_mask

        # **Case 3: Mixed FEN and Move Tokens**
        # (Ordering: [FEN tokens] + [special token] + [move tokens])
        else:
            board = chess.Board()
            for move in moves[:move_index + 1]:
                board.push(move)
            fen_tokens = tokenize_board_to_tensor(board, token_dim=token_dim_fen)  # (n_fen, token_dim_fen)
            n_fen = fen_tokens.size(0)
            remaining_moves = moves[move_index + 1:]
            raw_move_features = raw_move_representation(remaining_moves)  # (n_moves, 136)
            n_moves = len(remaining_moves)
            sequence_length = n_fen + 1 + n_moves
            combined_tokens = torch.zeros((sequence_length, token_dim_total), dtype=torch.float32)
            combined_tokens[:n_fen, 136:150] = fen_tokens
            combined_tokens[n_fen] = special_token
            if n_moves > 0:
                combined_tokens[n_fen+1:, :136] = raw_move_features
            causal_mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
            if n_moves > 0:
                move_causal_mask = torch.triu(torch.ones((n_moves, n_moves), dtype=torch.bool), diagonal=1)
                causal_mask[n_fen+1:, n_fen+1:] = move_causal_mask
            return combined_tokens, causal_mask

    except Exception as e:
        print(e)
        print(pgn)
        print(num_moves)
        print(move_index)
        print(probabilities)
        sys.stdout.flush()

        

def generate_flipped_pawn_promotions():
    # Define the promotion rows for both colors and the promotion pieces
    promotion_rows = {'white': '7'}
    # promotion_rows = {'white': '7'}
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

def generate_pawn_promotions():
    # Define the promotion rows for both colors and the promotion pieces
    promotion_rows = {'white': '7', 'black': '2'}
    # promotion_rows = {'white': '7'}
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




def get_all_possible_moves(flipped=False):
    
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
    
    if not flipped:

        pawn_promotions = generate_pawn_promotions()
    
    else:

        pawn_promotions = generate_flipped_pawn_promotions()
    
    return all_moves + pawn_promotions

def export_partial_pgn(game, move_count):
    """
    Export a PGN string up to a given number of moves.

    Args:
        game (chess.pgn.Game): The original game object.
        move_count (int): The number of moves to include in the partial PGN.

    Returns:
        str: A PGN string with moves up to the specified count.
    """
    partial_game = chess.pgn.Game()
    partial_game.headers = game.headers.copy()  # Copy headers (e.g., player names, event)
    node = partial_game
    for i, move in enumerate(game.mainline_moves()):
        if i >= move_count:
            break
        node = node.add_variation(move)  # Add the move to the partial game
    return str(partial_game)



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def sort_chunk(chunk):
    # Sort a single chunk by the seventh element.
    return sorted(chunk, key=lambda x: x[6])

def parallel_sort(data, num_workers):
    """
    Parallel sorts a list of tuples based on the seventh element (index 6).
    
    Args:
        data (list): The list of tuples to sort.
        num_workers (int): The number of parallel workers to use.
    
    Returns:
        list: The sorted list.
    """
    import concurrent.futures
    import heapq

    n = len(data)
    # If the dataset is small or we have one worker, sort directly.
    if n < 10000 or num_workers <= 1:
        return sorted(data, key=lambda x: x[6])
    
    # Split the data into approximately equal chunks.
    chunk_size = n // num_workers
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_workers)]
    # Add any remainder to the last chunk.
    if n % num_workers:
        chunks[-1].extend(data[num_workers * chunk_size:])
    
    # Sort each chunk in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        sorted_chunks = list(executor.map(sort_chunk, chunks))
    
    # Merge the sorted chunks.
    sorted_data = list(heapq.merge(*sorted_chunks, key=lambda x: x[6]))
    return sorted_data

def parse_time_control(time_control: str) -> tuple[int, int]:
    """
    Parses a chess time control string (e.g., "300+0") into total time and increment in seconds.

    Args:
        time_control (str): A string representing the time control in the format "seconds+increment".

    Returns:
        tuple[int, int]: A tuple containing total time in seconds and increment in seconds.
    """
    total, increment = map(int, time_control.strip().split('+'))
    return total, increment
