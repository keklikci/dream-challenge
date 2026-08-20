"""Script companion for the root exploratory notebook.

The reusable pipeline lives in ``dream.py``. This companion keeps notebook
execution discoverable without importing notebook state into the package.
"""

from dream import main


if __name__ == "__main__":
    main()
