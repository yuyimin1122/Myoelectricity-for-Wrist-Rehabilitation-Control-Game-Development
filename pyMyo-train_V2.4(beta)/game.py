from main import *
from pyMyoMain import *
import sys
from PyQt5.QtWidgets import QApplication


def main():

    while True:

        AUDIO['start'].play()
        IMAGES['bgpic'] = IMAGES['day']
        color = random.choice(['red', 'yellow', 'blue'])
        IMAGES['birds'] = [IMAGES[color+'-up'], IMAGES[color+'-mid'], IMAGES[color+'-down']]
        pipe = IMAGES[random.choice(['green-pipe', 'red-pipe'])]
        IMAGES['pipes'] = [pipe, pygame.transform.flip(pipe, False, True)]
        menu_window()

        result = game_window()
        end_window(result)


if __name__ == '__main__':
    main()