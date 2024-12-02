from pyjamas.rplugins.base import PJSPluginABC


class PJSPlugin(PJSPluginABC):
    def name(self) -> str:
        return "Sample plugin"

    def run(self, parameters: dict) -> bool:
        print("Hello my friends!")
        return True
