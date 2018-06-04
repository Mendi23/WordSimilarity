from abc import abstractmethod


class BaseContext:
    def __init__(self, filePath):
        self.filePath = filePath

    def __enter__(self):
        self.fileobj = open(self.filePath, "w", encoding="utf8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fileobj.close()

    @abstractmethod
    def process(self, index, cur):
        pass
