from queue import Queue
from typing import List

def bfs(map: list, i: int, j: int):
    """Makes Breadth First Search and marks island visited.

    Args:
        map (list): Matrix M*N
        i (int): Starting row index
        j (int): Starting column index
    """
    dir_x = (0, 0, -1, 1)
    dir_y = (-1, 1, 0, 0)
    q = Queue()
    q.put((i, j))
    while(not q.empty()):
        x, y = q.get()
        check_border = x >= 0 and x < len(map)
        check_border = check_border and y >= 0 and y < len(map[x])
        if check_border and map[x][y] == 1:
            map[x][y] = 2
            for dx, dy in zip(dir_x, dir_y):
                q.put((x+dx, y+dy))


def counting_islands(map: list) -> int:
    """Takes Matrix M*N as an input and counts the number of islands.

    The function uses BFS algorithm to find the solution.
    There are 3 possible states on the map:
    0 - the ocean, 1 - unvisited islands, 2 - visited islands
    At the end of the algorith all the islands will be visited.

    Args:
        map (List): Matrix M*N size
    Returns:
        int: Number of islands
    """
    num_islands = 0
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] != 1:
                continue
            num_islands += 1
            bfs(map, i, j)
    return num_islands


def read_row() -> List:
    """Reads a row and raise error if there is non-integer value in a row

    Returns:
        List: row of cells
    """
    inp = input()
    try:
        row = [int(x) for x in inp.split()]
    except Exception as e:
        raise ValueError("Input dimensions must be integers")
    return row


def read_input() -> List:
    """Reads M,N dimentions of Matrix and then full map.

    Returns:
        List: All the map with islands.
    """
    map = []
    first_row = read_row()
    if len(first_row) != 2:
        raise ValueError("Matrix must be 2D")
    M, N = first_row

    for i in range(M):
        row = read_row()
        if len(row) != N:
            raise ValueError(f"Row length must be equal N={N}")
        map.append(row)

    return map


if __name__ == '__main__':
    map = read_input()
    num_islands = counting_islands(map)
    print(num_islands)
