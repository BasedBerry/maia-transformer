# Transformer Maia2:

The same as Maia2 repo except...

Important changes: 

transformer_only.py file contains new Maia2Model class and updated process_per_game function 

utils.py contains new tokenize_board_to_tensor function, new MoveTokenizer class, and new game_to_tensor function. 

IMPORTANT: right now the model only works with board tokenization, will integrate move tokenization into training after we finish testing board tokenization

