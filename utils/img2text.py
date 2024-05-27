from omegaconf import DictConfig


class Img2Text:
    def __init__(self, config: DictConfig):
        self.api_key = config.img2text.api_key
        self.api_url = config.img2text.api_url

    def get_text(self):
        # some code to convert image to text
        return "Hello World"
