#!/usr/bin/python3
#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
'''
The Sudoku Bot
'''
import logging
import argparse
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from solver import Board
from tools import draw_matrix, solution_img
from cv2 import imread, IMREAD_GRAYSCALE
from keras.models import model_from_json
from image_recognition import get_grid


# Create bot entity with API

def start(bot, update):
    '''
    Define the /start command
    '''
    bot.send_message(chat_id=update.message.chat_id, text="""
/generate <difficulty> to receive a sudoku puzzle as an image
send me an image of the puzzle and I'll try to solve it""")

def error(bot, update, error):
    '''
    Print warnings in case of errors
    '''
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"',
                   update,
                   error)

def generate(bot, update, args):
    '''
    Generate random message starting from words given (/with command)
    '''
    argument = ' '.join(args)
    B = Board(9)
    B.generate_puzzle(1, to_remove=30)
    draw_matrix(B.grid, name='puzzle.png')
    bot.send_photo(chat_id=update.message.chat_id, photo=open('puzzle.png', 'rb'))

def solve(bot, update):
    file_id = update.message.photo[-1]
    newFile = bot.getFile(file_id)
    newFile.download('download.jpg')
    bot.sendMessage(chat_id=update.message.chat_id, text="Processing...")
    img = imread('download.jpg', IMREAD_GRAYSCALE)
    json_file = open(args.model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(args.model +'.h5')
    grid = get_grid(img, model, method='cnn')
    B = Board(9, grid)
    # print(B.grid)
    solution_img(B.solution().T, B.grid.T, 'solution.png')
    bot.send_photo(chat_id=update.message.chat_id, photo=open('solution.png', 'rb'))


def start_bot(token):
    updater = Updater(token=token)
    dispatcher = updater.dispatcher
    # Config the logger
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Add all the handlers
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('generate', generate, pass_args=True))
    photo_handler = MessageHandler(Filters.photo, solve)
    dispatcher.add_handler(photo_handler)
    dispatcher.add_error_handler(error)
    # Start the bot
    updater.start_polling()
    # idle is better than just polling, because of Ctrl+c
    updater.idle()

if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('model', help='Location of the model.json and model.h5 files for CNN')
    parser.add_argument("token", help="your bot API token", type=str)
    args = parser.parse_args()
    # MODEL
    # json_file = open(args.model + '.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(args.model +'.h5')
    start_bot(token=args.token)
