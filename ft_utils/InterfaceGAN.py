from tqdm import tqdm
from ft_utils.utils import BATCH_SIZE
from ft_utils.BatchImageClassifier import BatchImageClassifier
from ft_utils.BatchImageGenerator import BatchImageGenerator

import numpy as np
from sklearn import svm

class InterfaceGAN():
    def __init__(self):
        self.__boundary = None
        return

    def __find_boundary(self, latent_vectors, scores):
        self.__boundary = self.__interfacegan__train_boundary(latent_codes=latent_vectors, scores=scores)
        return

    def __move_into_direction(self, latent_vector):
        if self.__boundary is None: raise ValueError("No Boundary exists. Run find_boundary first")

        return self.__interfacegan__linear_interpolate(latent_code=latent_vector, boundary=self.__boundary, start_distance=0, steps=64)

    def latent_walk(self, attributes, latent_vector):
        batch_classifier = BatchImageClassifier("out_batch_transfer")
        batch_generator = BatchImageGenerator("out_batch_transfer", True)

        def classify_young(start_seed, batch_size, text):
            return batch_classifier.classify_from_batch(start_seed, batch_size, text)

        scores = [] # np.array([])
        latent_vectors_list = []

        text_features = batch_classifier.tokenize_attributes(attributes)

        print("Scoring images")
        for i in tqdm(range(0, round(200_000 / BATCH_SIZE))):
            probs = classify_young(i*BATCH_SIZE, BATCH_SIZE, text_features)
            scores.extend([t[0,0].item() for t in probs]) # Use extend for efficiency
            latent_vectors_list.append(batch_generator.load_w_batch(i*BATCH_SIZE, BATCH_SIZE))

        latent_vectors = np.concatenate(latent_vectors_list, axis=0)
        scores = np.array(scores).reshape(-1, 1)

        top_k_count = 3_000
        sorted_indices = np.argsort(scores.squeeze())
        worst_indices = sorted_indices[:top_k_count]
        best_indices = sorted_indices[-top_k_count:]
        combined_indices = np.concatenate([best_indices, worst_indices])

        latent_vectors_best_and_worst = latent_vectors[combined_indices]
        print(latent_vectors_best_and_worst)

        self.train_boundary(latent_vectors_best_and_worst, top_k_count)

        # self.__find_boundary(np.squeeze(latent_vectors), scores)
        return self.__move_into_direction(latent_vector)

    def train_boundary(self, latent_vectors, top_k_count):
        """
        Takes latent_vectors that have 10% best and then 10% worst
        performing latent vectors in terms of their CLIP score
        """
        latent_space_dim = latent_vectors.shape[1]

        train_label = np.concatenate([np.ones(top_k_count, dtype=np.int32),
                                    np.zeros(top_k_count, dtype=np.int32)], axis=0)

        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(latent_vectors, train_label)
        print(f'Finish training.')

        a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
        self.__boundary = a / np.linalg.norm(a)
        return


    """ORIGINAL INTERFACEGAN FUNCTIONS"""
    def __interfacegan__train_boundary(self, latent_codes,
                    scores,
                    chosen_num_or_ratio=0.02,
                    split_ratio=0.7,
                    invalid_value=None,
                    logger=None):
        """Trains boundary in latent space with offline predicted attribute scores.

        Given a collection of latent codes and the attribute scores predicted from the
        corresponding images, this function will train a linear SVM by treating it as
        a bi-classification problem. Basically, the samples with highest attribute
        scores are treated as positive samples, while those with lowest scores as
        negative. For now, the latent code can ONLY be with 1 dimension.

        NOTE: The returned boundary is with shape (1, latent_space_dim), and also
        normalized with unit norm.

        Args:
            latent_codes: Input latent codes as training data.
            scores: Input attribute scores used to generate training labels.
            chosen_num_or_ratio: How many samples will be chosen as positive (negative)
            samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
            latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
            0.5 * latent_codes_num)` will be used. (default: 0.02)
            split_ratio: Ratio to split training and validation sets. (default: 0.7)
            invalid_value: This field is used to filter out data. (default: None)
            logger: Logger for recording log messages. If set as `None`, a default
            logger, which prints messages from all levels to screen, will be created.
            (default: None)

        Returns:
            A decision boundary with type `numpy.ndarray`.

        Raises:
            ValueError: If the input `latent_codes` or `scores` are with invalid format.
        """

        if (not isinstance(latent_codes, np.ndarray) or
            not len(latent_codes.shape) == 2):
            raise ValueError(f'Input `latent_codes` should be with type'
                            f'`numpy.ndarray`, and shape [num_samples, '
                            f'latent_space_dim]!')
        num_samples = latent_codes.shape[0]
        latent_space_dim = latent_codes.shape[1]
        if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
            not scores.shape[0] == num_samples or not scores.shape[1] == 1):
            raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                            f'shape [num_samples, 1], where `num_samples` should be '
                            f'exactly same as that of input `latent_codes`!')
        if chosen_num_or_ratio <= 0:
            raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                            f'but {chosen_num_or_ratio} received!')

        print(f'Filtering training data.')
        if invalid_value is not None:
            latent_codes = latent_codes[scores[:, 0] != invalid_value]
            scores = scores[scores[:, 0] != invalid_value]

        print(f'Sorting scores to get positive and negative samples.')
        sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
        latent_codes = latent_codes[sorted_idx]
        scores = scores[sorted_idx]
        num_samples = latent_codes.shape[0]
        if 0 < chosen_num_or_ratio <= 1:
            chosen_num = int(num_samples * chosen_num_or_ratio)
        else:
            chosen_num = int(chosen_num_or_ratio)
        chosen_num = min(chosen_num, num_samples // 2)

        print(f'Spliting training and validation sets:')
        train_num = int(chosen_num * split_ratio)
        val_num = chosen_num - train_num
        # Positive samples.
        positive_idx = np.arange(chosen_num)
        np.random.shuffle(positive_idx)
        positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
        positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
        # Negative samples.
        negative_idx = np.arange(chosen_num)
        np.random.shuffle(negative_idx)
        negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
        negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
        # Training set.
        train_data = np.concatenate([positive_train, negative_train], axis=0)
        train_label = np.concatenate([np.ones(train_num, dtype=np.int32),
                                        np.zeros(train_num, dtype=np.int32)], axis=0)
        print(f'  Training: {train_num} positive, {train_num} negative.')
        # Validation set.
        val_data = np.concatenate([positive_val, negative_val], axis=0)
        val_label = np.concatenate([np.ones(val_num, dtype=np.int32),
                                    np.zeros(val_num, dtype=np.int32)], axis=0)
        print(f'  Validation: {val_num} positive, {val_num} negative.')
        # Remaining set.
        remaining_num = num_samples - chosen_num * 2
        remaining_data = latent_codes[chosen_num:-chosen_num]
        remaining_scores = scores[chosen_num:-chosen_num]
        decision_value = (scores[0] + scores[-1]) / 2
        remaining_label = np.ones(remaining_num, dtype=np.int32)
        remaining_label[remaining_scores.ravel() < decision_value] = 0
        remaining_positive_num = np.sum(remaining_label == 1)
        remaining_negative_num = np.sum(remaining_label == 0)
        print(f'  Remaining: {remaining_positive_num} positive, '
                    f'{remaining_negative_num} negative.')

        print(f'Training boundary.')
        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(train_data, train_label)
        print(f'Finish training.')

        if val_num:
            val_prediction = classifier.predict(val_data)
            correct_num = np.sum(val_label == val_prediction)
            print(f'Accuracy for validation set: '
                        f'{correct_num} / {val_num * 2} = '
                        f'{correct_num / (val_num * 2):.6f}')

        if remaining_num:
            remaining_prediction = classifier.predict(remaining_data)
            correct_num = np.sum(remaining_label == remaining_prediction)
            print(f'Accuracy for remaining set: '
                        f'{correct_num} / {remaining_num} = '
                        f'{correct_num / remaining_num:.6f}')

        a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
        return a / np.linalg.norm(a)


    def __interfacegan__project_boundary(self, primal, *args):
        """Projects the primal boundary onto condition boundaries.

        The function is used for conditional manipulation, where the projected vector
        will be subscribed from the normal direction of the original boundary. Here,
        all input boundaries are supposed to have already been normalized to unit
        norm, and with same shape [1, latent_space_dim].

        Args:
            primal: The primal boundary.
            *args: Other boundaries as conditions.

        Returns:
            A projected boundary (also normalized to unit norm), which is orthogonal to
            all condition boundaries.

        Raises:
            LinAlgError: If there are more than two condition boundaries and the method fails 
                        to find a projected boundary orthogonal to all condition boundaries.
        """
        assert len(primal.shape) == 2 and primal.shape[0] == 1

        if not args:
            return primal
        if len(args) == 1:
            cond = args[0]
            assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
                    cond.shape[1] == primal.shape[1])
            new = primal - primal.dot(cond.T) * cond
            return new / np.linalg.norm(new)
        elif len(args) == 2:
            cond_1 = args[0]
            cond_2 = args[1]
            assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
                    cond_1.shape[1] == primal.shape[1])
            assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
                    cond_2.shape[1] == primal.shape[1])
            primal_cond_1 = primal.dot(cond_1.T)
            primal_cond_2 = primal.dot(cond_2.T)
            cond_1_cond_2 = cond_1.dot(cond_2.T)
            alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
                1 - cond_1_cond_2 ** 2 + 1e-8)
            beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
                1 - cond_1_cond_2 ** 2 + 1e-8)
            new = primal - alpha * cond_1 - beta * cond_2
            return new / np.linalg.norm(new)
        else:
            for cond_boundary in args:
                assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
                        cond_boundary.shape[1] == primal.shape[1])
                cond_boundaries = np.squeeze(np.asarray(args))
                A = np.matmul(cond_boundaries, cond_boundaries.T)
                B = np.matmul(cond_boundaries, primal.T)
                x = np.linalg.solve(A, B)
                new = primal - (np.matmul(x.T, cond_boundaries))
            return new / np.linalg.norm(new)


    def __interfacegan__linear_interpolate(self, latent_code,
                        boundary,
                        start_distance=-3.0,
                        end_distance=3.0,
                        steps=10):
        """Manipulates the given latent code with respect to a particular boundary.

        Basically, this function takes a latent code and a boundary as inputs, and
        outputs a collection of manipulated latent codes. For example, let `steps` to
        be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
        `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
        with shape [10, latent_space_dim]. The first output latent code is
        `start_distance` away from the given `boundary`, while the last output latent
        code is `end_distance` away from the given `boundary`. Remaining latent codes
        are linearly interpolated.

        Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
        to support W+ space in Style GAN. In this case, all features in W+ space will
        be manipulated same as each other. Accordingly, the output will be with shape
        [10, num_layers, latent_space_dim].

        NOTE: Distance is sign sensitive.

        Args:
            latent_code: The input latent code for manipulation.
            boundary: The semantic boundary as reference.
            start_distance: The distance to the boundary where the manipulation starts.
            (default: -3.0)
            end_distance: The distance to the boundary where the manipulation ends.
            (default: 3.0)
            steps: Number of steps to move the latent code from start position to end
            position. (default: 10)
        """
        assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
                len(boundary.shape) == 2 and
                boundary.shape[1] == latent_code.shape[-1])

        linspace = np.linspace(start_distance, end_distance, steps)
        if len(latent_code.shape) == 2:
            linspace = linspace - latent_code.dot(boundary.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            return latent_code + linspace * boundary
        if len(latent_code.shape) == 3:
            linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
            return latent_code + linspace * boundary.reshape(1, 1, -1)
        raise ValueError(f'Input `latent_code` should be with shape '
                        f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                        f'W+ space in Style GAN!\n'
                        f'But {latent_code.shape} is received.')
