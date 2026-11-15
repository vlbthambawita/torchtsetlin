"""Job-queue worker with the HTM arms registered (``registers`` adds them to ``runner.ARMS``).

    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.htm.worker jobs.json [...]
"""
import sys

from experiments.transformer_concepts.code import jobqueue
from experiments.transformer_concepts.code.htm import registers  # noqa: F401  (registers the arms)

if __name__ == "__main__":
    jobqueue.work(sys.argv[1:])
