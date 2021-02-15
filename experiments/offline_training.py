import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " test_race.py >res1.txt " +
          "' &")
