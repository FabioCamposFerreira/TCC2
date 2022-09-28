import sys
import os
import subprocess
import time


class ProgressBar:
    def __init__(self, total: float):
        self.total = total
        self.time_before = time.time()
        self.line = ""
        self.time_delta = 0
        self.percentage_before = 0
        self.percentage_now = 0
        self.percentage_delta = 0

    def line_up(self):
        """Print in line up"""
        sys.stdout.write('\x1b[1A')
        sys.stdout.flush()

    def line_down(self):
        """Print in line down"""
        # I could use '\x1b[1B' here, but newline is faster and easier
        sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, actual: float):
        """Print and update progress bar"""
        self.percentage_now = actual/self.total*100
        self.percentage_delta = self.percentage_now - self.percentage_before
        self.percentage_before = self.percentage_now
        time_now = time.time()
        self.time_delta = time_now - self.time_before
        self.time_before = time_now

    def make_line(self, text: str):
        """Construct line"""
        line_width = int(subprocess.check_output("tput cols", shell=True))
        try:
            time_left = int(self.time_delta/self.percentage_delta*(100-self.percentage_now))
            minutes = int(time_left/60)
            seconds = int(time_left - minutes*60)
        except ZeroDivisionError:
            minutes = 0
            seconds = 0
        self.line = text+" [*] {}% {} min {} s".format(int(self.percentage_now), minutes, seconds)
        bar_len = (line_width-len(self.line))
        hash_quantity = int(self.percentage_now*bar_len/100)
        hyphen_quantity = bar_len-hash_quantity
        self.line = self.line.replace("*", "#"*hash_quantity+"-"*hyphen_quantity)

    def print(self, text: str, actual: float, line=0):
        """Print line

        Parameters
        ----------
        text : str
            _description_
        actual : float
            _description_
        line : int, optional
            -1 print in acima , 0 print in line actual, 1 print in line below , by default 0
        """
        self.update(actual)
        self.make_line(text)
        self.enable_print()
        if line == -1:
            self.line_up()
            self.line_up()
            print(self.line, end="\r")
            self.line_down()
        elif line == 0:
            print(self.line, end="\r")
        elif line == 1:
            self.line_down()
            print(self.line, end="\r")
            self.line_up()
        self.block_print()

    def block_print(self):
        """Not run prints"""
        sys.stdout = open(os.devnull, 'w')

    def enable_print(self):
        """Cancel function block_print"""
        sys.stdout = sys.__stdout__

    def end(self):
        """Always call after loop with progress_bar()"""
        self.enable_print()
        # print(end='\x1b[2K') # clear line
        print()
