from __future__ import annotations

import sys

from ringdown.experiments import PAPER_FIG10_PROFILE_GEOMETRY, run_registered_paper_fig10_script


def main() -> None:
    run_registered_paper_fig10_script(PAPER_FIG10_PROFILE_GEOMETRY, sys.argv[1:])


if __name__ == "__main__":
    main()
