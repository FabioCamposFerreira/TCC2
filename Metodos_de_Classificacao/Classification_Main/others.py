import os
import subprocess
import sys
import time


def images_sort(e: str):
    return float(".".join(e.split(".")[0:2]))


class TimeConversion():
    def __init__(self, seconds) -> None:
        self.minutes = int(seconds/60)
        self.hours = int(self.minutes/60)
        self.minutes = self.minutes - self.hours*60
        self.seconds = int(seconds - self.minutes*60 - self.hours*60)
        if self.seconds > 30:
            self.seconds = 60
        elif self.seconds <= 10:
            self.seconds = 10
        elif self.seconds < 30:
            self.seconds = 30
        self.time_formatted = ""
        if self.hours != 0:
            self.time_formatted += " {} h".format(self.hours)
        if self.minutes != 0:
            self.time_formatted += " {} min".format(self.minutes)
        self.time_formatted += " {} s".format(self.seconds)

    def __repr__(self):
        return self.time_formatted


class ProgressBar:
    def __init__(self, text: str, total: float, line_position: int):
        self.total = total
        self.time_before = time.time()
        self.line = ""
        self.line_position = line_position
        self.text = text
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

    def make_line(self):
        """Construct line"""
        try:
            line_width = int(subprocess.check_output("tput cols", shell=True))
        except:
            line_width=100
        try:
            time_formated = TimeConversion(int(self.time_delta / self.percentage_delta*(100-self.percentage_now)))
        except ZeroDivisionError:
            time_formated = "0 s"
        self.line = self.text+" [*] {}% {}".format(int(self.percentage_now), time_formated)
        bar_len = (line_width-len(self.line))
        hash_quantity = int(self.percentage_now*bar_len/100)
        hyphen_quantity = bar_len-hash_quantity
        self.line = self.line.replace("*", "#"*hash_quantity+"-"*hyphen_quantity)

    def print(self, actual: float):
        """Print line

        Parameters
        ----------
        actual : float
            _description_
        line : int, optional
            -1 print in acima , 0 print in line actual, 1 print in line below , by default 0
        """
        self.update(actual)
        self.make_line()
        # self.enable_print()
        if self.line_position == -1:
            self.line_up()
            self.line_up()
            print(self.line, end="\r")
            self.line_down()
        elif self.line_position == 0:
            print(self.line, end="\r")
        elif self.line_position == 1:
            self.line_down()
            print(self.line, end="\r")
            self.line_up()
        # self.block_print()

    def block_print(self):
        """Not run prints"""
        sys.stdout = open(os.devnull, 'w')

    def enable_print(self):
        """Cancel function block_print"""
        sys.stdout = sys.__stdout__

    def end(self):
        """Always call after loop with progress_bar()"""
        self.print(self.total)
        # self.enable_print()
        # print(end='\x1b[2K') # clear line
        print()
