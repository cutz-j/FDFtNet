from .base import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--source_dataset', type=str)
        parser.add_argument('--target_dataset', type=str)
        parser.add_argument('--resume', type=str)

        self.isTrain = False
        return parser