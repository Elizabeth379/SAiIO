import backpack
from data import weights, values, capacity
def main() -> None:
    result = backpack.backpack(weights, values, capacity)

    print("Вместимость рюкзака:", capacity)
    print("Веса предметов:", weights)
    print("Ценности предметов:", values)
    print("Максимальная ценность в рюкзаке:", result)


if __name__ == '__main__':
    main()