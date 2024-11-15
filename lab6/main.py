from matchmaking import matchmaking
from data import graph


def main() -> None:
    L: int = 3
    R: int = 3

    par = matchmaking(graph, 0, L + R + 1)
    print(f'Max matchmaking: {par[0]}\n')
    g = par[1]

    print('Match makings:\n')

    for i in range(L + 1, L + R + 1):
        for j in range(1, L + 1):
            if g[i][j] == 1:
                print('[', j, ',', i - L, ']\n')
                break


if __name__ == '__main__':
    main()