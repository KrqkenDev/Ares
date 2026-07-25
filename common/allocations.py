class Allocation:

    STEP = 0.05
    

    @classmethod
    def values(cls):
        return[ round(i * cls.STEP, 2) for i in range(1, 21)]

    @classmethod
    def count(cls):
        return len(cls.values())

    @classmethod
    def get(cls, index: int):
        values = cls.values()

        if index < 0 or index >= len(values):
            raise ValueError(f"Invalid allocation index: {index}")

        return values[index]