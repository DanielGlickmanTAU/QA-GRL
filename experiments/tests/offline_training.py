import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " question_generation_test.py >res1.txt " +
          "' &")
