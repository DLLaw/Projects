from typing import List, Optional
from position import Position
from player_colors import PlayerColors
from game_piece import GamePiece
from game_player import GamePlayer

class GoModel: # starting over lol
    """
    Represents the Go game model including board state, player turns, and game rules.

    Handles the board state,players, turn transitions, move history, and validation of moves according to Go
    game rules. It provides functionality to place pieces, switch turns, pass turns,
    check validity of moves, verify game-ending conditions, and obeys?
    the game's Ko rule. The class maintains the board as a grid of positions, tracks
    player information, and manages the game's state throughout its lifecycle.

    """
    valid_sizes = {6, 9, 11, 13, 19}

    def __init__(self, rows: int = 6, cols: int = 6):
        """
        Initializes a game board, players, and move tracking functionalities for the board game. This class
        maintains the current state of the board, the players' data, and handles specifics such as
        undoing moves, validation of board dimensions, and initialization of a starting game state. The board
        must be square and of valid dimensions provided in the allowed size list.

        Args:
            rows (int, optional): Number of rows for the game board. Must be an integer in valid sizes and equal to `cols`. Defaults to 6.
            cols (int, optional): Number of columns for the game board. Must be an integer in valid sizes and equal to `rows`. Defaults to 6.

        Raises:
            TypeError: If `rows` or `cols` is not of type `int`.
            ValueError: If `rows` or `cols` is not in the valid sizes list or if the board is not square.
        """
        if not isinstance(rows, int):
            raise TypeError("rows must be an integer")

        if not isinstance(cols, int):
            raise TypeError("cols must be an integer")

        if rows not in self.valid_sizes or cols not in self.valid_sizes:
            raise ValueError("board size must be a valid size")

        if cols != rows:
            raise ValueError("board must be square")

        # board stuff
        self.__nrows = rows
        self.__ncols = cols
        self.__board: List[List[Optional[GamePiece]]] = [[None for _ in range(cols)] for _ in range(rows)]
        # self.__current_player = GamePlayer(PlayerColors.BLACK)
        self.__message = ""

        # player stuff
        self.__black_player = GamePlayer(PlayerColors.BLACK)
        self.__white_player = GamePlayer(PlayerColors.WHITE)
        self.__current_player = self.__black_player # black goes first

        # move history for undo
        self.__move_history = [] # stores previous game states for undo
        self.__previous_board_states = [] # tracks past boards for ko role

        #Saves the empty board state
        board = []

        for i in range(self.nrows):
            lst = []
            for j in range(self.ncols):
                lst.append(None)
            board.append(lst)

        self.__move_history.append({
            "board": board,
            "player": self.__current_player,
            "captures": {p.player_color: p.capture_count for p in [self.__black_player, self.__white_player]},
            "skips": {p.player_color: p.skip_count for p in [self.__black_player, self.__white_player]}
        })

        # for pass turn
        self.consecutive_skips = 0

        #These are from Cooper's code.  They are placeholder pieces for set_piece.
        self.white = GamePiece(PlayerColors.WHITE)
        self.black = GamePiece(PlayerColors.BLACK)

    @property # readable
    def nrows(self) -> int:
        """
        A property that represents the number of rows.

        Returns:
            nrows (int): The number of rows.
        """
        return self.__nrows # number of rows

    @property
    def black_player(self):
        """
        A property that represents the black player.

        Returns:
            black_player: The black player.
        """
        return self.__black_player


    @property # readable
    def ncols(self) -> int:
        """
        A property that represents the number of columns.

        Returns:
            ncols (int): The number of columns.
        """
        return self.__ncols # number of columns

    @property # readable
    def current_player(self) -> GamePlayer:
        """
        Gets the current player of the game.

        Returns:
            The player currently taking their turn in the game.
        """
        return self.__current_player # current player, obviously

    @property # readable
    def board(self) -> List[List[Optional[GamePiece]]]:
        """
        Provides an accessor for the current state of the game board.



        Returns:
            The current state of the game board as a nested list. Each element is either a `GamePiece` instance
            or `None`.
        """
        return self.__board # current board state

    @property # readable
    def message(self) -> str:
        """
        Represents a property to access the current game message in a read-only manner.

        Returns:
            message (str): The current game message stored in the instance.
        """
        return self.__message # current game message

    @message.setter # writable
    def message(self, msg: str):
        """
        Sets a new value for the `message` attribute. This method ensures that the
        provided value is a string. If the type of the value is not a string, it
        raises a TypeError. This setter method updates the private attribute
        `__message` with the given value.

        Args:
            msg (str): The new value to set for the `message` attribute.

        Raises:
            TypeError: If the provided value is not a string.
        """
        if not isinstance(msg, str):
            raise TypeError("message must be a string")
        self.__message = msg # sets a new message

    def piece_at(self, pos: Position) -> Optional[GamePiece]:
        """
        Returns the game piece located at a specified position on the board. The position
        must be within the bounds of the board and should be an instance of the
        `Position` class. If the position is out of bounds or the provided input
        is not a `Position`, an error is raised.

        Args:
            pos (Position): The position on the board for which the game piece is to be retrieved.

        Raises:
            TypeError: If the given `pos` is not an instance of `Position`.
            ValueError: If the given `pos` is out of the board's valid boundaries.

        Returns:
            board (Optional[GamePiece]): The game piece located at the specified `pos`, or `None` if no piece is present.
        """
        if not isinstance(pos, Position):
            raise TypeError("pos must be a Position")

        if not (0 <= pos.row < self.__nrows and 0 <= pos.col < self.__ncols):
            raise ValueError("out of bounds")
        return self.__board[pos.row][pos.col] # return a game piece at a given position

    def set_piece(self, pos: Position, piece: Optional[GamePiece] = None) -> None:
        """
        Sets a game piece at the specified position on the board. This method performs validation
        on the input parameters to ensure the position is within bounds and assigns the piece to
        the specified position. It updates the current board state, resets consecutive skip count,
        and stores the move history for potential undo functionality.

        Args:
            pos (Position): The position on the board where the piece is to be placed.
            piece (Optional[GamePiece], optional): The game piece to place at the specified position, or None to clear the position. Defaults to None.
        """
        if not isinstance(pos, Position):
            raise TypeError("pos must be a Position")

        if not isinstance(piece, GamePiece) and piece is not None:
            raise TypeError("Piece must be a gamepiece or None")

        if not (0 <= pos.row < self.__nrows and 0 <= pos.col < self.__ncols): # thank you prof
            raise ValueError("position is out of bounds")

        board = []
        for k in range(0, self.ncols):
            board.append([])

        for i in range(self.nrows):
            board[i] = self.board[i].copy()

        if piece is None:
            pass
        elif piece == self.white:
            piece = self.white
        elif piece == self.black:
            piece = self.black

        self.message = f"{self.current_player} {pos.__str__()}"

        self.__board[pos.row][pos.col] = piece # places a game piece at a given position
        self.consecutive_skips = 0

        # saves current board state
        self.__move_history.append({
            "board": board,
            "player": self.__current_player,
            "captures": {p.player_color: p.capture_count for p in [self.__black_player, self.__white_player]},
            "skips": {p.player_color: p.skip_count for p in [self.__black_player, self.__white_player]}
            })

        self.__previous_board_states.append(row[:] for row in self.__board)

    def set_next_player(self) -> None:
        """
        Switches the turn to the next player in the game.

        Returns:
            None
        """
        print("Switching player...")
        self.__current_player = (self.__black_player if self.__current_player == self.__white_player else self.__white_player)
        print(f"Next player: {self.__current_player.player_color.name}")

    def pass_turn(self) -> None:
        """
        Increments the skip count for the current player, updates the consecutive skip count,
        and switches the turn to the next player. This method ensures proper handling of
        skipped turns in a game.

        Returns:
            None
        """
        self.__current_player.skip_count += 1
        self.consecutive_skips += 1
        self.set_next_player() # switches turn to next player

    def is_game_over(self) -> bool:
        '''
        is_game_over checks if the game is over.

        Returns:
            True if the game is over, False otherwise.
        '''
        if self.consecutive_skips >= 2:
            return True
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i][j] is None:
                    piece = None
                    piece_color = self.current_player.player_color
                    if piece_color == self.black.color:
                        piece = self.black
                    if piece_color == self.white.color:
                        piece = self.white
                    if self.is_valid_placement(Position(i, j), piece):
                        return False
        return True

    def is_valid_placement(self, pos: Position, piece: GamePiece) -> bool: # checks if given position is valid before placing the actual piece
        """
        Validates if a given position is appropriate for placing a game piece on the board.

        Args:
            pos (Position): The position to be checked.
            piece (GamePiece): The game piece to be placed.

        Returns:
            True if the position is valid for placement; False otherwise.

        Raises:
            TypeError: If `pos` is not an instance of Position or `piece` is not an instance of GamePiece.
        """
        if not isinstance(pos, Position):
            raise TypeError("pos must be a Position")

        if not isinstance(piece, GamePiece):
            raise TypeError("piece must be a gamepiece or None")

        if not (0 <= pos.row < self.__nrows and 0 <= pos.col < self.__ncols): # checks out of bounds
            self.__message = "out of bounds"
            return False

        if self.__board[pos.row][pos.col] is not None: # if that position is already taken
            self.__message = "position already taken"
            return False


        if not piece.is_valid_placement(pos, self.__board): # is_valid_placement from game_piece.py
            self.__message = "no liberties (is_valid_placement failed)"
            return False

        if self.__check_ko_rule(pos, piece): # ko rule is loop of being captured over and over again
            self.__message = "ko rule violated"
            return False
        return True

    def __check_ko_rule(self, pos: Position, piece: GamePiece) -> bool: # used with is_valid_placement, ko rile is loop of being captured over and over again
        """
        Checks the Ko rule in Go, which prevents a player from making a move that would
        recreate a previous board state. This ensures that the game does not fall into
        an infinite loop of captures and recaptures of the same stones.

        Args:
            pos: The position on the board where the piece is placed.
            piece: The game piece being placed on the Go board.

        Returns:
            True if the Ko rule is violated, False otherwise.
        """
        simulated_board = [row[:] for row in self.__board]  # copy board state
        simulated_board[pos.row][pos.col] = piece  # place the piece

        return simulated_board in self.__previous_board_states  # compare past states

    def __has_liberty(self, pos: Position, visited: set) -> bool: # capture helper method, a set is a unorganized collection of elements that is mutable
        """
        Determine if the given position has liberty on the game board. A position
        has liberty if there is at least one adjacent empty space or if connected
        stones of the same color have liberty.

        Liberty in this context refers to an empty space directly adjacent to the
        given position or any connected same-colored positions.

        Args:
            pos: The position to check for liberty.
            visited: A set of positions already visited during the recursive check to avoid processing the same position multiple times.

        Returns:
            True if the given position has liberty, False otherwise.
        """
        if pos in visited:
            return False  # already checked position
        visited.add(pos)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up down left right

        for dr, dc in directions: # moves to each of the 4 neighboring positions
            new_row, new_col = pos.row + dr, pos.col + dc

            if 0 <= new_row < self.__nrows and 0 <= new_col < self.__ncols: # checks if new position is within board boundaries
                neighbor = self.__board[new_row][new_col]
                if neighbor is None:  # found a liberty
                    return True
                elif neighbor.color == self.__board[pos.row][pos.col].color: # if the neighbor color is the same color
                    if self.__has_liberty(Position(new_row, new_col), visited): # recursively check that stone
                        return True
                else:
                    return False

        return False # if all adjacent positions have been checked and no liberties are found return false

    def capture(self) -> None:
        """
        Identifies and removes opponent's captured stones (no liberties) from the board, updates the current
        player's capture count, and sets an appropriate game message. Captures refer to stones fully surrounded
        and without any liberties (adjacent empty spaces). This method checks every position on the board and
        removes any captured stones, incrementing the player's capture count and updating the game state.

        Raises:
            RuntimeError: If the current player's state is invalid during capture operation.
            ValueError: If the board configuration or dimensions are inconsistent.
        """
        opponent_color = self.__current_player.player_color.opponent() # gets opponent color
        captured_positions = [] # list for captures positions (stones with no liberties)
        # finds all pieces that are surrounded
        for row in range(self.__nrows):
            for col in range(self.__ncols): # loops through every position on the board
                piece = self.__board[row][col]
                if piece and piece.color == opponent_color: # if a piece is found and is opponent color it will be checked for liberties
                    if not self.__has_liberty(Position(row, col), set()): # calls __has_liberty to check if the opponents piece has at least 1 empty adjacent space, if no liberties are found false is returned and the position is added to captured_positions
                        captured_positions.append(Position(row, col))

        # remove captured stones and update capture count
        for pos in captured_positions: # iterates through all captured positions
            self.__board[pos.row][pos.col] = None # removes them from the board by setting their position to none
            self.__current_player.capture_count += 1 # capture count goes up for current player

        # update game message
        if captured_positions:
            self.__message = f"{self.__current_player.player_color.name} captured {len(captured_positions)} pieces"
        else:
            self.__message = f"no captures this turn"

#-----------------------------------------------------------------------------------------------------------------------

    def calculate_score(self) -> list:
        """
        Calculates and returns the scores for two players based on the capture count, game pieces
        on the board, and the territories controlled by each player. This function also accounts
        for the additional points for the WHITE player as per the game rules.

        Returns:
            scores: A list containing the scores of both players, where the first index represents the BLACK player's score and the second index represents the WHITE player's score.
        """
        visited_positions = []

        #Starts each player's score with their capture count, and adds WHITE player's extra points.
        scores = [self.__black_player.capture_count, self.__white_player.capture_count + 6.5]
        players = [self.__black_player, self.__white_player]

        #Loops through the board, and if a game piece is there, add it to the respective player's score.
        for row in range(self.nrows):
            for col in range(self.ncols):

                #Makes sure that every position is only visited once.
                if (Position(row, col) in visited_positions):
                    continue

                    #Makes sure that the board at the Position is a game piece
                if not isinstance(self.board[row][col], GamePiece):
                    continue

                if self.board[row][col].color == PlayerColors.BLACK:
                    scores[0] += 1

                elif self.board[row][col].color == PlayerColors.WHITE:
                    scores[1] += 1

                x, black_loop = self.surrounded(PlayerColors.BLACK, Position(row, col))
                y, white_loop = self.surrounded(PlayerColors.WHITE, Position(row, col))
                add_black_territory = True
                add_white_territoty = True
                for i in black_loop:
                    if i in visited_positions:
                        add_black_territory = False

                for i in white_loop:
                    if i in visited_positions:
                        add_white_territoty = False

                if add_black_territory:
                    scores[0] += self.territory(black_loop)[2]
                    visited_positions.extend(black_loop)

                if add_white_territoty:
                    scores[1] += self.territory(white_loop)[2]
                    visited_positions.extend(white_loop)
                visited_positions.append(Position(row, col))
        return scores

#-----------------------------------------------------------------------------------------------------------------------

    def surrounded(self, color, pos):
        """
        Determines if a set of pieces of a given color is surrounded in a grid. If surrounded, it returns a flag
        indicating truth and the list of pieces forming the closed boundary. The method utilizes the positioning
        and layout of a two-dimensional array to evaluate if elements of the given color form a fully enclosed
        loop inside the grid's boundaries.

        Args:
            color: The color of the pieces to check for surrounding.
            pos: The starting position from which the check begins.

        Returns:
            True if the set is surrounded, False otherwise.
            return_list: A list of Position objects that form the boundary of the surrounded region if it exists.
        """
        pieces = []

        for i in range(pos.row, self.nrows):
            for j in range(pos.col, self.ncols):
                if self.board[i][j]:
                    if self.board[i][j].color == color:
                        pieces.append(Position(i, j))

        row_dictionary = {}
        col_dictionary = {}
        for i in pieces:
            if not i.row in row_dictionary:
                row_dictionary[i.row] = []
            if not i.col in col_dictionary:
                col_dictionary[i.col] = []
            row_dictionary[i.row].append(i.col)
            col_dictionary[i.col].append(i.row)

        if not row_dictionary:
            return False, []

        top = min(row_dictionary.keys())
        bottom = max(row_dictionary.keys())

        while len(row_dictionary[bottom]) != len(row_dictionary[top]):
            if len(row_dictionary[bottom]) > len(row_dictionary[top]):
                top += 1
                while top not in row_dictionary:
                    top += 1
            elif len(row_dictionary[top]) > len(row_dictionary[bottom]):
                bottom -= 1
                while bottom not in row_dictionary:
                    bottom -= 1

        left = len(row_dictionary[top]) - 2
        right = len(row_dictionary[top]) + 2

        while left not in col_dictionary:
            left += 1
            if left >= self.ncols // right:
                return False, []

        while right not in col_dictionary:
            right -= 1
            if right <= self.ncols // left:
                return False, []

        top_row = []
        bottom_row = []

        left_col = []
        right_col = []

        for i in pieces:
            if i.row == top and left <= i.col <= right:
                top_row.append(i)
            elif i.row == bottom and left <= i.col <= right:
                bottom_row.append(i)

            if i.col == left and top <= i.row <= bottom:
                left_col.append(i)
            elif i.col == right and top <= i.row <= bottom:
                right_col.append(i)

        return_list = []
        for i in top_row:
            if i not in return_list:
                return_list.append(i)

        for i in right_col:
            if i not in return_list:
                return_list.append(i)

        for i in bottom_row:
            if i not in return_list:
                return_list.append(i)

        for i in left_col:
            if i not in return_list:
                return_list.append(i)

        loop = 0
        while loop < len(top_row) and loop < len(return_list) - 1:
            if return_list[loop].col + 1 != return_list[loop + 1].col and return_list[loop].col + 1 != return_list[loop + 1].col + 1:
                return False, return_list
            loop += 1

        while len(top_row) <= loop < len(right_col) and loop < len(return_list) - 1:
            if return_list[loop].row + 1 != return_list[loop + 1].row and return_list[loop].row + 1 != return_list[loop + 1].row + 1:
                return False, return_list
            loop += 1

        while len(right_col) <= loop < len(bottom_row) and loop < len(return_list) - 1:
            if return_list[loop].col + 1 != return_list[loop + 1].col and return_list[loop].col + 1 != return_list[loop + 1].col + 1:
                return False, return_list
            loop += 1

        while len(bottom_row) <= loop < len(left_col) and loop < len(return_list) - 1:
            if return_list[loop].row + 1 != return_list[loop + 1].row and return_list[loop].row + 1 != return_list[loop + 1].row + 1:
                return False, return_list
            loop += 1
        return True, return_list

    def territory(self, lst: list[Position]):
        """
        Computes the territory details based on given positions in the board. This method identifies the central
        positions of a provided group on the board, maps the rows and their respective column indices in a dictionary,
        and determines the count of central positions that differ from the initial group composition.
        Args:
            lst: A list of Position objects representing specific positions to analyze on the board.

        Returns:
            A list containing the center positions of the territory, a dictionary mapping rows to their column indices, and the number of central positions.
        """
        center = []
        dictionary = {}
        for j in lst:
            if not j.row in dictionary:
                dictionary[j.row] = []
            if not j.col in dictionary[j.row]:
                dictionary[j.row].append(j.col)

        for i in range(self.nrows):
            if i in dictionary.keys():
                for j in range(self.ncols):
                    if dictionary[i][0] < j < dictionary[i][-1]:
                        if self.board[i][j] != self.board[lst[0].row][lst[0].col]:
                            center.append(Position(i, j))
        return [center, dictionary, len(center)]

    def in_confines(self, pos: Position):
        """
        Check if a given position is within the confines of a grid.

        This function determines if a given position lies within the
        valid bounds of a grid defined by the attributes `nrows` (number
        of rows) and `ncols` (number of columns). If the position is
        none, it is considered to be within the confines by default.
        Args:
            pos: The position to check for confines validity.

        Returns:
            true or false

        Raises:
            TypeError: If the `pos` parameter is not an instance of the `Position` class.
        """
        if pos is None:
            return True
        if not isinstance(pos, Position):
            raise TypeError("Position must be of the position class.")
        if pos.row >= self.nrows or pos.col >= self.ncols or pos.row < 0 or pos.col < 0:
            return False
        return True

#-----------------------------------------------------------------------------------------------------------------------

    def __flood_fill_territory(self, pos: Position, territory_positions: set, surrounding_colors: set, visited: set) -> None: # helper method for calculate score
        """
        Performs a flood fill algorithm on the board starting from a given position to determine the territory
        of empty spaces and the surrounding stone colors. This is a helper method used to calculate the score
        in the game.

        Args:
            pos: The starting position of the flood fill represented as a ``Position`` object with row and col attributes.
            territory_positions: A set of tuples, with each tuple representing the row and column of a position
            surrounding_colors: A set used to keep track of the distinct colors of stones surrounding the territory.
            visited: A set of tuples representing positions that have already been checked during the flood fill

        Returns:
            Does not return anything. It returns only to prevent infinite loops.
        """
        if (pos.row, pos.col) in visited: # already checked the position
            return  # prevents infinite loops

        visited.add((pos.row, pos.col))  # mark starting position visited
        queue = [pos]  # list of empty positions on the board being explored as part of a territory

        while queue:
            current = queue.pop(0) # removes position from queue...
            territory_positions.add((current.row, current.col))  # and adds it to the territory

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up down left right
            for dr, dc in directions:
                new_row, new_col = current.row + dr, current.col + dc
                if 0 <= new_row < self.__nrows and 0 <= new_col < self.__ncols: # out of bounds

                    neighbor = self.__board[new_row][new_col]

                    if neighbor is None:  # if neighbor is empty space
                        if (new_row, new_col) not in visited: # if neighbor is empty, add it to territory
                            queue.append(Position(new_row, new_col)) # add it to queue
                            visited.add((new_row, new_col))
                    else:
                        surrounding_colors.add(neighbor.color)  # add surrounding stone color

    def undo(self):
        """
        Undoes the last move performed in the game. The method restores the
        game state to the most recent state saved in the move history. If the
        move history contains only one entry, the operation will fail as there
        are no moves to undo.

        Raises:
            UndoException: Raised when the move history contains only one entry,
        """
        if len(self.__move_history) == 1: # if there is nothing in the move history we cant undo
            raise UndoException("nothing in move history")

        # undoes the game
        last_state = self.__move_history.pop()
        self.__board = [row[:] for row in last_state["board"]]  # Ensure deep copy
        self.__message = "undo"

        self.__current_player.capture_count = last_state["captures"][self.__current_player.player_color]
        self.__current_player.skip_count = last_state["skips"][self.__current_player.player_color]

        if self.__previous_board_states: # prevents 2 of the same board in previous_board_states list
            self.__previous_board_states.pop()

    def __str__(self):
        """
        Converts the board object into a human-readable string. The
        string contains each row of the board with its columns read as spaces.
        Each row is separated by a newline character. If a cell on the board is empty,
        it will print 'None' for that position; otherwise, it will call the `__str__`
        method of the object stored at that position.

        Returns:
            return_string: A string representation of the board.
        """
        return_string = ''
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i][j] is None:
                    return_string += "None "
                else:
                    return_string += self.board[i][j].__str__() + " "
            return_string += "\n"
        return return_string

class UndoException(Exception):
    """
    Represents a custom exception for errors related to undo operations.

    This exception can be used to specifically handle cases where an undo
    operation fails or is not possible.

    """
    pass

if __name__ == "__main__":
    x = GoModel(6, 6)
    x.set_piece(Position(0, 4), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(0, 2), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(0, 3), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(1, 5), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(2, 4), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(2, 2), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(2, 3), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(1, 1), GamePiece(PlayerColors.BLACK))
    x.set_piece(Position(5, 5), GamePiece(PlayerColors.BLACK))
    y = x.board[0][4]
    """x.set_piece(Position(1, 2), GamePiece(PlayerColors.WHITE))
    x.set_piece(Position(1, 3), GamePiece(PlayerColors.WHITE))
    x.set_piece(Position(1, 4), GamePiece(PlayerColors.WHITE))"""
    print(x)
    print(x.surrounded(PlayerColors.BLACK, Position(0, 0)))
    print(x.black_player.capture_count)
    print(x.calculate_score())
