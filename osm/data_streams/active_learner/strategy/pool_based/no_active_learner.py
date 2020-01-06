from osm.data_streams.active_learner.strategy.abstract_strategy import AbstractActiveLearningStrategy


class NoActiveLearner(AbstractActiveLearningStrategy):
    def below_threshold(self, gain):
        """
        Just returns true acting as if no active learning takes place
        :param gain: the information gain if the instance is used
        :return: True
        """

        return True

    def get_name(self):
        return "no_active_learner"
