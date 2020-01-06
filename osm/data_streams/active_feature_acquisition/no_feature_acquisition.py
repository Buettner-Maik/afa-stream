from osm.data_streams.active_feature_acquisition.abstract_feature_acquisition import AbstractActiveFeatureAcquisitionStrategy

class NoActiveFeatureAcquisition(AbstractActiveFeatureAcquisitionStrategy):
    """
    immediately returns the incoming data
    """
    def get_data(self, data):
        return data

    def get_name(self):
        return "no_active_feature_acquisition"