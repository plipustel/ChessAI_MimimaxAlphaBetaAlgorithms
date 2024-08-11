"""-------------------------------------------------------------------------------------------------------------
@Project Name : Artificial Intelligence Chess AI
@Description  : This project is aimed for educational purpose, implement the Artificial Intelligence algorithms: 
                Minimax and Alpha Beta Pruning. The pretty project example in game theory is building Chess AI. Since 
                the game tree of Chess AI has over 10^40 nodes (According to Stuart Russel), it is impossible to traverse of the tree. 
                The best way to cut off the pruned branch is by defining the Heuristic Evaluation Functions (Reference: Shannon Fano)
                
                The Evaluation function on this project, take the very fundamental eval functions:
                1. Pieces Power
                2. Pieces Positions
                3. Pieces Movement
                4. King safety
                For serious project, the sophistic evaluation functions must be added and improved
                
@Algorithms   : Minimax, Alpha Beta Pruning, Heuristic Evaluation Functions           
@Year         : 2024, Aug
@Author       : Plipus Telaumbanua
-------------------------------------------------------------------------------------------------------------"""
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Customable heuristic evaluation functions
class EvalFunc:
    def __init__(self):
        # pieces weight
        self.pieces_power = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 4, 'queen': 9, 'king': 0}
        
        # position weight
        self.piece_pos = {
            'pawn': [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.05, 0.05, 0.2, 0.3, 0.3, 0.2, 0.05, 0.05],
                [0, 0, 0.15, 0.2, 0.2, 0.15, 0, 0],
                [0.05, -0.05, -0.1, 0, 0, -0.1, -0.05, 0.05],
                [0.05, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.05],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            'knight': [
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5],
                [-0.4, -0.2, 0, 0.1, 0.1, 0, -0.2, -0.4],
                [-0.3, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, -0.3],
                [-0.3, 0, 0.3, 0.4, 0.4, 0.3, 0, -0.3],
                [-0.3, 0, 0.3, 0.4, 0.4, 0.3, 0, -0.3],
                [-0.3, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, -0.3],
                [-0.4, -0.2, 0, 0.1, 0.1, 0, -0.2, -0.4],
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5]
            ],
            'bishop': [
                [-0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0.1, 0.2, 0.2, 0.1, 0, -0.1],
                [-0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, -0.1],
                [-0.1, 0, 0.2, 0.2, 0.2, 0.2, 0, -0.1],
                [-0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, -0.1],
                [-0.1, 0.1, 0, 0, 0, 0, 0.1, -0.1],
                [-0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2]
            ],
            'rook': [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [0, 0, 0, 0.1, 0.1, 0, 0, 0]
            ],
            'queen': [
                [-0.2, -0.1, -0.1, -0.05, -0.05, -0.1, -0.1, -0.2],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.1],
                [-0.05, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.05],
                [0, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.05],
                [-0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0, -0.1],
                [-0.1, 0, 0.05, 0, 0, 0, 0, -0.1],
                [-0.2, -0.1, -0.1, -0.05, -0.05, -0.1, -0.1, -0.2]
            ],
            'king': [
                [2, 3, 1, 0, 0, 1, 3, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [2, 3, 1, 0, 0, 1, 3, 2]
            ]
        }
        
        
class ChessPieces:
    def __init__(self, color, pieceType):
        #load pieces
        self.color = color
        self.pieceType = pieceType
        image_path = f'D:/TheCoder/ChessAI/ChessPieces/{color}_{pieceType}.png'
        self.image = ImageTk.PhotoImage(Image.open(image_path).resize((60, 60)))
class ChessAI:
    
    def __init__(self, root):
        self.root = root
        self.root.title('Plipus Chess AI')
        self.n = 8 # chess board 8x8
        self.board = [[None for _ in range(self.n)] for _ in range(self.n)]        
        self.pieces_ref = [[None for _ in range(self.n)] for _ in range(self.n)]
        self.init_board()
        self.selected_piece = None
        self.selected_piece_from = None # coordinate variable
        self.selected_piece_to = None # coordinate variable as well
        self.captured_piece_stack = []
        self.eval = EvalFunc()
        self.AI = 'black'
        self.turn = True
        self.over = False
        
    def init_board(self):
        colors = ['#f5e0ce', '#b06f37']
        for i in range(self.n):
            for j in range(self.n):
                color = colors[(i+j) % 2]
                square = tk.Frame(self.root, bg=color, height=80, width=80)
                square.grid(row=i, column=j)
                self.board[i][j] = square # store the board box
                
                label = tk.Label(square, text=None, bg=color)
                label.place(relx=0.5, rely=0.5, anchor='center')
                square.bind('<Button-1>', lambda e, x=i, y=j: self.on_square_click(x, y))
    
    def init_pieces(self):
        start_pieces_pos = [
            (0, 0, ChessPieces('black', 'rook')), (0, 1, ChessPieces('black', 'knight')), (0, 2, ChessPieces('black', 'bishop')),
            (0, 3, ChessPieces('black', 'queen')), (0, 4, ChessPieces('black', 'king')), (0, 5, ChessPieces('black', 'bishop')),
            (0, 6, ChessPieces('black', 'knight')), (0, 7, ChessPieces('black', 'rook')), 
            (1, 0, ChessPieces('black', 'pawn')),(1, 1, ChessPieces('black', 'pawn')),(1, 2, ChessPieces('black', 'pawn')),
            (1, 3, ChessPieces('black', 'pawn')),(1, 4, ChessPieces('black', 'pawn')),(1, 5, ChessPieces('black', 'pawn')),
            (1, 6, ChessPieces('black', 'pawn')),(1, 7, ChessPieces('black', 'pawn')),
            
            # white pieces
            (7, 0, ChessPieces('white', 'rook')), (7, 1, ChessPieces('white', 'knight')), (7, 2, ChessPieces('white', 'bishop')),
            (7, 3, ChessPieces('white', 'queen')), (7, 4, ChessPieces('white', 'king')), (7, 5, ChessPieces('white', 'bishop')),
            (7, 6, ChessPieces('white', 'knight')), (7, 7, ChessPieces('white', 'rook')), 
            (6, 0, ChessPieces('white', 'pawn')),(6, 1, ChessPieces('white', 'pawn')),(6, 2, ChessPieces('white', 'pawn')),
            (6, 3, ChessPieces('white', 'pawn')),(6, 4, ChessPieces('white', 'pawn')),(6, 5, ChessPieces('white', 'pawn')),
            (6, 6, ChessPieces('white', 'pawn')),(6, 7, ChessPieces('white', 'pawn')),
      
        ]   
         
        for i, j, piece in start_pieces_pos:
            frame = self.board[i][j]
            label = tk.Label(frame, image=piece.image, bg=frame.cget('bg'))
            label.place(x=5, y=5)
            label.bind('<Button-1>', lambda e, x=i, y=j: self.on_square_click(x, y))
            self.pieces_ref[i][j] = piece
               
    def on_square_click(self, i, j):
        if self.selected_piece is None:
            if self.pieces_ref[i][j] is not None:
                self.selected_piece = self.pieces_ref[i][j]
                self.selected_piece_from = (i, j)
        else:
            self.selected_piece_to = (i, j)
            p, q = self.selected_piece_from
            x, y = self.selected_piece_to
            if self.valid_move(p, q, x, y): # check for valid movements
                self.movement()
                self.AI_Call(False)
                
    def AI_Call(self, turn):
        if turn is False and not self.over:
            self.AI_BestMove(self.AI)
            self.movement()
            
    def _move_piece(self, p, q, c, d):
        frame = self.board[c][d]
        #handle pawn promotion later
        color = self.selected_piece.color
        ptype = self.selected_piece.pieceType
        promoted = self.pawn_promote(color, ptype, c)
        piece = promoted if promoted is not None else self.selected_piece
        # end pawn promotion handler
        label = tk.Label(frame, image=piece.image, bg=frame.cget('bg'))
        label.place(x=5, y=5)
        label.bind('<Button-1>', lambda e, x=c, y=d: self.on_square_click(x, y))
        self.pieces_ref[c][d] = piece
        self.pieces_ref[p][q] = None
        self._reset_pieces_value()
        # test Alpha Beta algorithm
        #self.MinimaxAlphaBeta(2, float('-inf'), float('inf'), False)
        for widget in self.board[p][q].winfo_children():
            widget.destroy()
               
    def movement(self):
        p, q = self.selected_piece_from
        x, y = self.selected_piece_to
        color = self.selected_piece.color
        opponent = 'white' if color == 'black' else 'black'
        if self.valid_move(p, q, x, y):
            self._move_piece(p, q, x, y)
            self.is_king_dead(opponent)
            if self.king_under_attack(opponent):
                self.check_king(opponent)
                if self.checkmate(opponent):
                    self.game_over(opponent)
            elif self.stalemate(opponent):
                self.drawn()
                
    # handle obstacles for bishop and rook, will useful for queen
    def bishop_obstacles_clear(self, p, q, x, y):
        # check for negative/positive diagonal slope
        if abs(p - x) == abs(q - y):
            p_step = 1 if p < x else -1
            q_step = 1 if q < y else -1
            row, col = p + p_step, q + q_step
            while row != x and col != y:
                if self.pieces_ref[row][col] is not None:
                    self._reset_pieces_value()
                    return False
                row += p_step
                col += q_step
            return True
        self._reset_pieces_value()
        return False
    
    def rook_obstacles_clear(self, p, q, x, y):
        if q == y:
            for i in range(min(p, x) + 1, max(p, x)): # trying to fly (illegal)
                if self.pieces_ref[i][q] is not None:
                    self._reset_pieces_value()
                    return False
            return True     
        elif p == x:
            for i in range(min(q, y) + 1, max(q, y)):
                if self.pieces_ref[p][i] is not None:
                    self._reset_pieces_value()
                    return False
            return True
        self._reset_pieces_value()
        return False
    
    # movement roles for AI & Human
    def valid_move(self, p, q, x, y):
        piece = self.pieces_ref[p][q]
        if not piece:
            return False

        if piece.pieceType == 'pawn':
            if piece.color == 'white':
                # pawn's initial double move with no obstacles between and on the target
                if p == 6 and x == p - 2 and q == y and not self.pieces_ref[x][y] and not self.pieces_ref[p-1][q]:
                    return True
                if q == y and x == p - 1 and not self.pieces_ref[x][y]: # for single move
                    return True
                if x == p - 1 and 1 == abs(y - q) and self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'black': # enable opponent capturing, for negative and positive diagonal slope
                    return True
            else:
                # pawn's initial double move with no obstacles between and on the target
                if p == 1 and x == p + 2 and q == y and not self.pieces_ref[x][y] and not self.pieces_ref[p+1][q]:
                    return True
                if q == y and x == p + 1 and not self.pieces_ref[x][y]: # for single move
                    return True
                if x == p + 1 and 1 == abs(y - q) and self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'white': # enable opponent capturing, for negative and positive diagonal slope
                    return True
        elif piece.pieceType == 'knight':
            if piece.color == 'white':
                # letter 'L' flip > transform 90 degree and letter 'L'
                if(abs(p - x) == 1 and abs(q - y) == 2) or (abs(p - x) == 2 and abs(q - y) == 1):
                    if self.pieces_ref[x][y]:
                        if self.pieces_ref[x][y].color == 'black': # capture opponent
                            return True
                    else:
                        return True
            else:
                 # letter 'L' flip > transform 90 degree and letter 'L'
                if(abs(p - x) == 1 and abs(q - y) == 2) or (abs(p - x) == 2 and abs(q - y) == 1):
                    if self.pieces_ref[x][y]:
                        if self.pieces_ref[x][y].color == 'white': # capture opponent
                            return True
                    else:
                        return True
        elif piece.pieceType == 'bishop':
            if piece.color == 'white':
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'white': # trying capture tim mate
                    self._reset_pieces_value()
                    return False
            else:
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'black': # trying capture tim mate
                    self._reset_pieces_value()
                    return False 
            return self.bishop_obstacles_clear(p, q, x, y)
        
        elif piece.pieceType == 'rook':
            if piece.color == 'white':
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'white': 
                    self._reset_pieces_value()
                    return False
            else:
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'black':
                    self._reset_pieces_value()
                    return False
                
            return self.rook_obstacles_clear(p, q, x, y)
            
        elif piece.pieceType == 'queen':
            if piece.color == 'white':
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'white': 
                    self._reset_pieces_value()
                    return False
            else:
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'black':
                    self._reset_pieces_value()
                    return False
            # check obstacles
            if q == y or p == x:
                return self.rook_obstacles_clear(p, q, x, y)
            if abs(p - x) == abs(q - y):
                return self.bishop_obstacles_clear(p, q, x, y)
         
        elif piece.pieceType == 'king':
            if piece.color == 'white':
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'white':
                    self._reset_pieces_value()
                    return False
                if self.pieces_ref[x][y] == 'black':
                    return True
            else:
                if self.pieces_ref[x][y] and self.pieces_ref[x][y].color == 'black':
                    self._reset_pieces_value()
                    return False
                if self.pieces_ref[x][y] == 'white':
                    return True
            if abs(p - x) <= 1 and abs(q - y) <= 1:
                return True
    
    def pawn_promote(self, color, ptype, x):
        if(x == 0 or x == 7) and ptype == 'pawn':
            return ChessPieces(color, 'queen') # as default, highest piece value
        return None
    
    def _reset_pieces_value(self):
        self.selected_piece = None
        self.selected_piece_from = None # coordinate variable
        self.selected_piece_to = None # coordinate variable as well
    
    def possible_moves(self, color):
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                piece = self.pieces_ref[i][j]
                if piece is not None and piece.color == color:
                    for x in range(self.n):
                        for y in range(self.n):
                            if self.valid_move(i, j, x, y):
                                moves.append([(i, j), (x, y)])
        return moves
    
    def king_under_attack(self, color):
        king_pos = self.find_king_pos('king', color)
        if king_pos is None:
            return False
        x_king, y_king = king_pos
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces_ref[i][j] and self.pieces_ref[i][j].color != color:
                    if self.valid_move(i, j, x_king, y_king):
                        return True
        return False
                
    def protect_king_attempts(self, color):
        moves = []
        for i in range(self.n):
            for j in range(self.n):
                piece = self.pieces_ref[i][j]
                if piece is not None and piece.color == color:
                    for x in range(self.n):
                        for y in range(self.n):
                            if self.valid_move(i, j, x, y):
                                moves.append([(i, j),(x, y)])
                                
        return moves
               
    def king_escape(self, color):
        moves = self.protect_king_attempts(color)
        for coord in moves:
            start, end = coord
            current = self.pieces_ref[start[0]][start[1]]
            next = self.pieces_ref[end[0]][end[1]]
            self.pieces_ref[end[0]][end[1]] = current
            self.pieces_ref[start[0]][start[1]] = None
            if not self.king_under_attack(color):
                # revert
                self.pieces_ref[start[0]][start[1]] = current
                self.pieces_ref[end[0]][end[1]] = next
                return True
            
            self.pieces_ref[start[0]][start[1]] = current
            self.pieces_ref[end[0]][end[1]] = next
        return False
    
    def checkmate(self, opponent):
        if self.king_under_attack(opponent):
            if not self.king_escape(opponent):
                return True
        return False

    # drawn
    def stalemate(self, opponent):
        if not self.king_under_attack(opponent):
            if not self.king_escape(opponent):
                return True
        return False
        
    def game_over(self, opponent):
        winner = 'Congart, you win !' if opponent == self.AI else ':-D AI Win !'
        messagebox.showinfo('Info', f'{winner}')
        self.over = True
        
        # reset game
        for i in range(self.n):
            for j in range(self.n):
                self.pieces_ref[i][j] = None
        self.init_pieces()
        return 
     
    def drawn(self):
        messagebox.showinfo('Infor', 'Drawn ! Both of you are the best')
        for i in range(self.n):
            for j in range(self.n):
                self.pieces_ref[i][j] = None
        self.init_pieces()
        return 
    
    def check_king(self, opponent):
        messagebox.showinfo('Warning', f'Check {opponent}')
    
    def is_king_dead(self, opponent):
        if self.find_king_pos('king', opponent) is None:
            self.game_over(opponent)
        
    def find_king_pos(self, pieceType, color):
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces_ref[i][j] is not None:
                    piece = self.pieces_ref[i][j]
                    if piece.pieceType == pieceType and piece.color == color:
                        return (i, j)
        return None
    
    # heuristic evaluation functions
    def eval_pieces_move_pos(self, color):
        # material balanced (mb) variables
        piece_power_mb = 0 # material balanced from AI's perspective. DISADVANTAGE <= n <= ADVANTAGE
        piece_pos_mb = 0
        piece_mov_mb = 0
        king_safety = 0
        moves = len(self.protect_king_attempts(color))
        # calculate material balanced
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces_ref[i][j] is not None:
                    piece = self.pieces_ref[i][j]
                    power = self.eval.pieces_power[piece.pieceType]
                    position = self.eval.piece_pos[piece.pieceType][i][j]
                    if piece.color == color:
                        piece_power_mb += power
                        piece_pos_mb += position
                        piece_mov_mb += moves
                        king_safety += self._eval_king_safety(self.find_king_pos('king', piece.color))
                    else:
                        piece_power_mb -= power
                        piece_pos_mb -= position
                        piece_mov_mb -= moves
                        king_safety -= self._eval_king_safety(self.find_king_pos('king', piece.color))
        strategy = piece_pos_mb + piece_pos_mb + 0.1 * piece_mov_mb + king_safety
        print(f'{color} pieces power: {piece_power_mb}')
        print(f'{color} strategy power: {piece_pos_mb}')
        print(f'{color} attack power: {piece_mov_mb}')
        print(f'{color} king safety: {king_safety}')
        return strategy
    
    # priority king's safe
    def _eval_king_safety(self, king_pos):
        if king_pos is None:
            return 0
        x, y = king_pos
        safety = 0
        penalty = 1.0
        rewards = 0.5
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n and self.pieces_ref[nx][ny] is not None:
                pieces = self.pieces_ref[nx][ny]
                if pieces.color !=  self.pieces_ref[x][y].color:
                    safety -= penalty # penalty for enemy pieces near the king
                else:
                    safety += rewards # reward for friendly pieces near the king
        if self.king_under_attack(self.pieces_ref[x][y].color):
            safety -= 50
            
        return safety
        
    # Implement AI minimax and Alpha Beta Pruning Algorithms
    def MinimaxAlphaBeta(self, depth, alpha, beta, max_player):
        color = 'black' if max_player else 'white'
        
        # stop recursion for depth == 0
        if depth == 0 or self.checkmate(color):
            return self.eval_pieces_move_pos(color)
            
        if max_player: #AI, search maximum value of MIN White
            eval = float('-inf') #infinite negative numbers
            for moves in self.possible_moves(color):
                print(f'AI Thinking....{color}')
                # test possible moves later here (simulation)
                self.AI_MovesSimulation(moves)
                eval = max(eval, self.MinimaxAlphaBeta(depth - 1, alpha, beta, False)) # recursive call MAX
                # backtrack
                self.AI_BacktrackingMoves(moves)
                if alpha >= beta: # alpha beta pruning
                    print(f'{color} alpha:{alpha} >= beta{beta} pruned')
                    break
                alpha = max(eval, alpha)
            print(f'MAX {color} piece: {eval}')
            return eval
        else:
            eval = float('inf')
            for moves in self.possible_moves(color):
                print(f'AI Thinking....{color}')
                 # test possible moves later here
                self.AI_MovesSimulation(moves)
                eval = min(eval, self.MinimaxAlphaBeta(depth - 1, alpha, beta, True)) # MIN
                # backtrack
                self.AI_BacktrackingMoves(moves)
                if alpha >= beta: # alpha beta pruning
                    print(f'{color} alpha:{alpha} >= beta{beta} pruned')
                    break
                beta = min(eval, beta) 
            print(f'MIN {color} piece: {eval}')
            return eval
    
    # Assume AI is Black/MAX, it's time to implement AI make black capable to think :-)
    def AI_BestMove(self, color):
        alpha = float('-inf')    
        max_val = float('-inf')
        beta = float('inf')
        best_pos = None
        for moves in self.possible_moves(color):
            self.AI_MovesSimulation(moves) # if black move, what happen?
            min_val = self.MinimaxAlphaBeta(2, alpha, beta, False) # then simulate the possible moves
            self.AI_BacktrackingMoves(moves) # backtrack
            if min_val > max_val: # always true for first init
                max_val = min_val
                best_pos = moves # switch
        if best_pos is not None:
            start, end = best_pos
            self.selected_piece = self.pieces_ref[start[0]][start[1]]
            self.selected_piece_from = (start[0], start[1])
            self.selected_piece_to = (end[0], end[1])
        print(f'AI choose {color} move : {best_pos}, with evaluation val: {max_val}')  
        
    # make AI movement simulation
    def AI_MovesSimulation(self, moves):
        x, y = moves
        piece = self.pieces_ref[x[0]][x[1]]
        # store captured piece if any, and restore it later in backtracking algorithm
        captured_piece = self.pieces_ref[y[0]][y[1]]
        self.captured_piece_stack.append((captured_piece, y))
        self.pieces_ref[y[0]][y[1]] = piece
        self.pieces_ref[x[0]][x[1]] = None
    
    # backtracking algorithm to restore the captured pieces in any
    def AI_BacktrackingMoves(self, moves):
        x, y = moves
        piece = self.pieces_ref[y[0]][y[1]]
        # restore the captured pieces to its original position
        self.pieces_ref[x[0]][x[1]] = piece
        captured_piece, position = self.captured_piece_stack.pop()
        self.pieces_ref[y[0]][y[1]] = captured_piece
        
  
if __name__ == '__main__':
    root = tk.Tk()
    Chess = ChessAI(root)
    Chess.init_pieces()
    root.mainloop()
