import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector, dtype=int)
    n = x.shape[0]

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]

    diffs = xs[1:] - xs[:-1]
    mask = diffs > 0
    if not np.any(mask):
        # константный признак -> возвращаем пустой результат
        return np.array([]), np.array([]), None, -np.inf

    thresholds_all = (xs[:-1] + xs[1:]) / 2.0
    thresholds = thresholds_all[mask]

    cumsum_ones = np.cumsum(ys)
    ones_left = cumsum_ones[:-1][mask]
    n_left = np.arange(1, n)[mask]
    n_right = n - n_left
    ones_right = cumsum_ones[-1] - ones_left

    # энтропия Джини
    p1_l = ones_left / n_left
    p0_l = 1.0 - p1_l
    H_l = 1.0 - p1_l ** 2 - p0_l ** 2

    p1_r = ones_right / n_right
    p0_r = 1.0 - p1_r
    H_r = 1.0 - p1_r ** 2 - p0_r ** 2

    # Q(R)
    ginis = -(n_left / n) * H_l - (n_right / n) * H_r

    best_idx = int(np.argmax(ginis))
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(
            self,
            feature_types,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
    ):
        if np.any(
                list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, f"_{key}", value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # критерий останова: все метки одинаковые
        if np.all(sub_y == sub_y[0]):  # FIX: было !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # доп. критерии остановки
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None  # FIX: для сохранения map лучшего признака

        for feature in range(sub_X.shape[1]):  # FIX: начинали с 1
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)  # FIX: astype
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # FIX: было count / click

                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))  # FIX: x[0], не x[1]
                )
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]], dtype=float)
            else:
                raise ValueError

            # FIX: проверка на константный признак
            if np.unique(feature_vector).size < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                    categories_map_best = None
                elif feature_type == "categorical":  # FIX: было "Categorical"
                    threshold_best = threshold # FIX: храним числовой порог
                    categories_map_best = categories_map.copy() # сохраняем мапу
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # FIX: [0][0], не кортеж
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            # FIX: используем сохранённую map, иначе берётся последняя из цикла
            if categories_map_best is not None:
                node["categories_split"] = [k for k, v in categories_map_best.items() if v < threshold_best]
            else:
                node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(
            sub_X[np.logical_not(split)],
            sub_y[np.logical_not(split)],  # FIX: было sub_y[split]
            node["right_child"],
            depth + 1
        )

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        f = node["feature_split"]
        if "threshold" in node:
            return self._predict_node(
                x,
                node["left_child"] if x[f] < node["threshold"] else node["right_child"],
            )
        else:
            cats_left = set(node["categories_split"])
            return self._predict_node(
                x, node["left_child"] if x[f] in cats_left else node["right_child"]
            )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
