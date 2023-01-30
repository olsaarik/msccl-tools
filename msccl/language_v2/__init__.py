

# Context manager for for_range loops
class for_range:
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step

    def __enter__(self):
        self.iter = range(self.start, self.end, self.step)
        return self.iter.__iter__()

    def __exit__(self, type, value, traceback):
        pass

# Class for representing runtime indices, such as the local ran and the number of ranks
class IndexVar:
    def __init__(self, name):
        self.name = name

RANK = IndexVar("rank")
RANKS = IndexVar("ranks")

# Decorator for triggering compilation
def compile():
    def decorator(func):

    return decorator

def test():
    def allreduce_ring():
        chunks = Input().chunk_up(RANKS)
        next_rank = (RANK + 1) % RANKS
        prev_rank = (RANK - 1) % RANKS
        steps = RANKS - 1
        # Do the allreduce step
        with for_range(steps) as step:
            send_index = (rank - step) % RANKS
            chunks[send_index].send(next_rank)
            recv_index = (rank - step - 1) % RANKS
            chunks[recv_index].recv_reduce(prev_rank)
        # Do the allgather around the same ring, but starting one rank earlier
        with for_range(steps) as step:
            send_index = (rank - step + 1) % RANKS
            chunks[send_index].send(next_rank)
            recv_index = (rank - step) % RANKS
            chunks[recv_index].recv(prev_rank)