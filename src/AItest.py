import multiprocessing
import os

import numpy as np

from AIPlayer import Dispatcher, CoreAILogic
from BoardMover import gen_new_num, decode_board, encode_board, s_move_board

from ai_and_sort import ai_core


class AItest:
    def __init__(self, path):
        self.ai_logic = CoreAILogic()
        self.board_encoded, self.board, self.ai_player, self.score, self.history = self.new_game()
        self.ai_dispatcher = Dispatcher(self.board, self.board_encoded)
        self.died = False
        self.path = path
        self.endgame_65k = 0
        self.step = 0
        self.ai_logic.time_limit_ratio = 1.25

    def ai_step(self, counts):
        # 将计算完全委托给 ai_logic
        best_move = self.ai_logic.calculate_step(self.ai_player, self.board, counts)
        return {1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}.get(best_move, '')

    @staticmethod
    def new_game():
        board_encoded = np.uint64(0x0011212426ac17bd)  # np.uint64(gen_new_num(gen_new_num(np.uint64(0))[0])[0])
        board = decode_board(board_encoded)
        ai_player = ai_core.AIPlayer(board_encoded)
        ai_player.max_threads = 4
        score = 0
        history = [(board_encoded, score, 0)]
        return board_encoded, board, ai_player, score, history

    def one_step(self):
        self.ai_dispatcher.reset(self.board, self.board_encoded)
        best_move = self.ai_dispatcher.dispatcher()
        if best_move == 'AI':
            self.ai_player.board = self.board_encoded
            ai_move = self.ai_step(self.ai_dispatcher.counts)
            if ai_move:
                self.do_move(ai_move.capitalize())
            else:
                self.died = True
        else:
            self.do_move(best_move.capitalize())

    def do_move(self, direction):
        # print(direction)
        direction = {'Left':1, 'Right':2, 'Up':3, 'Down':4}[direction.capitalize()]
        board_encoded_new, new_score = s_move_board(self.board_encoded, direction)
        board_encoded_new = np.uint64(board_encoded_new)
        if board_encoded_new != self.board_encoded:
            self.board_encoded = board_encoded_new
            self.board = decode_board(self.board_encoded)
            self.score += new_score
            self.gen_new_num()
        else:
            self.died = True

    def gen_new_num(self):
        self.board_encoded = np.uint64(gen_new_num(self.board_encoded)[0])
        self.board = decode_board(self.board_encoded)
        self.history.append((self.board_encoded, self.score, self.ai_dispatcher.last_operator))

    def play(self):
        while not self.died:
            self.step += 1
            if self.step > 8192:
                self.died = True
                continue
            self.one_step()

            target_tile = 32768 >> self.endgame_65k
            if self.score > 840000 and np.sum((self.board == target_tile)) == 2:
                positions = np.where(self.board == target_tile)
                first_position = (positions[0][0], positions[1][0])
                second_position = (positions[0][1], positions[1][1])

                if positions[0][0] == positions[0][1] and abs(positions[1][0] - positions[1][1]) == 1:
                    self.board[first_position] = target_tile >> 1
                    self.board[second_position] = target_tile >> 1
                    self.board_encoded = np.uint64(encode_board(self.board))
                    self.score += target_tile
                    self.do_move('Right')
                    self.died = False
                    self.endgame_65k += 1

                elif positions[1][0] == positions[1][1] and abs(positions[0][0] - positions[0][1]) == 1:
                    self.board[first_position] = target_tile >> 1
                    self.board[second_position] = target_tile >> 1
                    self.board_encoded = np.uint64(encode_board(self.board))
                    self.score += target_tile
                    self.do_move('Down')
                    self.died = False
                    self.endgame_65k += 1

            if self.step % 931==128:
                print(self.score)
                print(self.board)
                print(os.path.basename(self.path))
                if self.score > 900000:
                    print('**************************************************')
                else:
                    print('                                                  ')

        print(self.score)
        print(self.board)
        print(os.path.basename(self.path), 'died')
        print('--------------------------------------------------')
        record = np.array(self.history, dtype='uint64,uint32,uint8')
        record.tofile(self.path)


def run_test(index):
    ai_test = AItest(fr"C:\Users\Administrator\Desktop\record\record1\{index}")
    ai_test.play()


def main():
    multiprocessing.freeze_support()
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(run_test, range(0, 400), chunksize=1)


if __name__ == "__main__":
    main()
