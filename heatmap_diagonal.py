import time

from mpi4py import MPI


def write_data(rank_val, n_bytes, delay_time):
    # write n_bytes to a file from
    # rank rank_val after delay_time seconds
    time.sleep(delay_time)
    filename = f"rank_{rank_val}_write_{n_bytes}_bytes"
    bytes_to_write = n_bytes * "x"
    bytes_to_write = bytes_to_write.encode("utf8")
    with open(filename, "wb") as outfile:
        outfile.write(bytes_to_write)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # each rank should write data
    # at a specific offset time to
    # produce diagonal heatmap IO
    # activity like this:
    # x
    #  x
    #   x
    #    x
    # The objective is to produce a darshan
    # log file with a testable ground truth
    # IO activity data structure that is sufficiently
    # exaggerated to produce a similar pattern for
    # both runtime and DXT heatmaps (we want to be
    # able to ensure that both heatmap data structures
    # are consistent with each other)

    delay_time = rank * 0.1
    for rank_val in range(size):
        if rank == rank_val:
            write_data(rank_val=rank_val,
                       n_bytes=1,
                       delay_time=delay_time)


if __name__ == "__main__":
    main()
