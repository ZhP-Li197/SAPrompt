import sys

from run import main


if __name__ == "__main__":
    sys.argv.insert(1, "pretrain")
    main()
