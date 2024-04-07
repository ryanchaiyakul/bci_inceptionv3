import pathlib
import numpy as np
import sklearn.model_selection

class ECGDataSet:
    """
    Transforms raw ECG data for preprocessing
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, person:np.ndarray):
        self.__x = x
        self.__y = y
        self.__person = person

    @property
    def raw(self) -> np.ndarray:
        """
        Raw np.ndarray input
        """
        return self.__x, self.__y

    def get_person(self, n: int):
        """
        Returns all the samples related to person n
        """
        idx = np.nonzero(self.__person == n)[0]
        return self.__x[idx], self.__y[idx]


    def train_valid_split(self, test_size: float, random_state: int = 0, person: int = -1):
        x_train, x_valid, y_train, y_valid, person_train, person_valid = sklearn.model_selection.train_test_split(self.__x, self.__y, self.__person, test_size = test_size, random_state = random_state)
        return ECGDataSet(x_train, y_train, person_train), ECGDataSet(x_valid, y_valid, person_valid)


class PreProcess:
    
    def __init__(self, root_dir: str = ""):
        self.__root_dir = pathlib.Path(root_dir)
        self.__test = ECGDataSet(np.load(self.__root_dir / "X_test.npy"), np.load(self.__root_dir / "y_test.npy"), np.load(self.__root_dir / "person_test.npy"))
        self.__train_valid = ECGDataSet(np.load(self.__root_dir / "X_train_valid.npy"), np.load(self.__root_dir / "y_train_valid.npy"), np.load(self.__root_dir / "person_train_valid.npy"))
        
    @property
    def train_valid(self) -> tuple:
        return self.__train_valid

    @property
    def test(self):
        return self.__test