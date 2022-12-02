class LabelQuery:
    def __init__(self, *args, **kwargs):
        self.config = self.get_query_config()
        self.base_query = self.get_base_query()
        self.info = self.config['labeler_info']
        
    def get_query_config(self):
        raise NotImplementedError
        
    def get_base_query(self):
        raise NotImplementedError